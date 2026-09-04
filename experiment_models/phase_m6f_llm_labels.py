# phase_m6f_llm_labels.py
# Смысловой анализ кластеров M6e через Qwen3.5-4B.
#
# Для каждой модели: загружаем HDBSCAN-метки + эмбеддинги →
# для каждого кластера (кроме шума) → top-5 документов, ближайших к центроиду →
# LLM (Qwen3.5-4B) → тематическая метка + уверенность.
#
# Запуск: python phase_m6f_llm_labels.py

import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

BASE = Path(__file__).resolve().parent
EMB_DIR = BASE / 'embeddings'
REPORTS_DIR = BASE / 'reports'
SAMPLE_CSV = BASE / 'sample_50k.csv'

LLM_MODEL_PATH = Path(cfg.HF_MODEL_PATH)
LLM_DEVICE = getattr(cfg, 'HF_DEVICE', 'cuda:0')

ORDER = ['nomic_v15', 'nomic_v2', 'e5_large', 'rubert_tiny2', 'sbert_ru',
         'qwen3e_4b', 'qwen3e_06b']

MAX_DIM = {'qwen3e_4b': 512, 'qwen3e_06b': 512}  # MRL для Qwen


def load_embeddings(model_key):
    parts = sorted(glob.glob(str(EMB_DIR / f"{model_key}__shard*.npy")))
    parts = [p for p in parts if not p.endswith('_ids.npy')]
    emb = np.concatenate([np.load(p) for p in parts])
    ids = np.concatenate([np.load(p.replace('.npy', '_ids.npy')) for p in parts])
    pos = {int(i): k for k, i in enumerate(ids)}
    order = [pos[int(i)] for i in df['id']]
    X = emb[order]
    d = MAX_DIM.get(model_key, 0)
    if d and d < X.shape[1]:
        X = X[:, :d]
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def top5_per_cluster(X, labels, subject_texts):
    """(list of dict: cluster_id, top5_texts, top5_sims, cluster_size)"""
    clusters = sorted(c for c in np.unique(labels) if c != -1)
    results = []
    for c in clusters:
        mask = labels == c
        centroid = X[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = X[mask] @ centroid
        top_idx = np.argsort(-sims)[:5]
        idxs = np.where(mask)[0][top_idx]
        results.append({
            'cluster': int(c),
            'size': int(mask.sum()),
            'top5_texts': [subject_texts[i] for i in idxs],
            'top5_sims': [float(sims[top_idx[k]]) for k in range(5)],
        })
    return results


def build_messages(cluster_texts):
    texts = '\n---\n'.join(cluster_texts)
    return [
        {'role': 'system', 'content': 'Ты определяешь общую тему пяти документов. '
         'Ответь одним-двумя словами.'},
        {'role': 'user', 'content': f'Определи общую тему:\n\n{texts}\n\nТЕМА:'},
    ]


def main():
    global df
    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    subject_texts = df['subject_text'].fillna('').tolist()
    print(f'загрузка LLM {LLM_MODEL_PATH} на {LLM_DEVICE}...', flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(LLM_MODEL_PATH))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(LLM_MODEL_PATH), torch_dtype=torch.bfloat16).to(LLM_DEVICE)
    model.eval()
    print(f'  LLM загружена за {time.time()-t0:.1f}с', flush=True)

    all_results = {}
    for key in ORDER:
        lab_path = REPORTS_DIR / f'm6e_labels_{key}.npy'
        if not lab_path.exists():
            print(f'  {key}: нет меток — пропуск', flush=True)
            continue

        print(f'\n=== {key} ===', flush=True)
        t0 = time.time()
        X = load_embeddings(key)
        labels = np.load(str(lab_path))
        clusters = top5_per_cluster(X, labels, subject_texts)
        print(f'  {len(clusters)} кластеров, загрузка эмбеддингов {time.time()-t0:.0f}с',
              flush=True)

        cluster_labels = []
        for cl in clusters:
            messages = build_messages(cl['top5_texts'])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt').to(LLM_DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=24, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()
            # очистка от остаточных <think> (если enable_thinking не сработал)
            if '<think>' in response:
                response = response.split('</think>')[-1].strip()
            response = response.split('\n')[0].strip().rstrip('.')
            if not response or len(response) > 80:
                response = '(неопределено)'
            cluster_labels.append({
                'cluster': cl['cluster'],
                'size': cl['size'],
                'llm_label': response,
                'top5_texts': cl['top5_texts'],
                'top5_sims': [round(s, 4) for s in cl['top5_sims']],
            })
            print(f'    кластер {cl["cluster"]:>3} ({cl["size"]:>5} док.): {response}',
                  flush=True)

        all_results[key] = cluster_labels
        print(f'  done ({time.time()-t0:.0f}с)', flush=True)

    # Сохранение
    p = REPORTS_DIR / 'm6f_llm_labels.json'
    p.write_text(json.dumps(all_results, ensure_ascii=False, indent=2,
                            default=lambda o: int(o) if isinstance(o, np.integer)
                            else float(o) if isinstance(o, np.floating)
                            else str(o)),
                 encoding='utf-8')

    # Markdown отчёт
    L = ['# M6f: смысловой анализ кластеров через LLM', '',
         'Для каждой модели HDBSCAN-кластеры, для каждого кластера '
         'топ-5 документов → Qwen3.5-4B → тематическая метка.', '',
         '| Модель | Кластер | Размер | Топ-5 sim | LLM-метка |',
         '|---|---|---|---|---|']
    for key in ORDER:
        cls = all_results.get(key)
        if not cls:
            continue
        for cl in cls:
            sims = ', '.join(f'{s:.3f}' for s in cl['top5_sims'])
            L.append(f"| {key} | {cl['cluster']} | {cl['size']} "
                     f"| {sims} | {cl['llm_label']} |")
        L.append('')

    # Итоговая таблица по моделям
    L.append('## Сводка: сколько кластеров и какие темы')
    L.append('')
    for key in ORDER:
        cls = all_results.get(key)
        if not cls:
            continue
        labels = [f"{c['cluster']}: «{c['llm_label']}» ({c['size']} док.)"
                  for c in cls]
        L.append(f'- **{key}** ({len(cls)} кластеров): ' + '; '.join(labels))

    out_md = REPORTS_DIR / 'm6f_llm_labels.md'
    out_md.write_text('\n'.join(L), encoding='utf-8')
    print(f'\nwritten {p}')
    print(f'written {out_md}')


if __name__ == '__main__':
    main()
