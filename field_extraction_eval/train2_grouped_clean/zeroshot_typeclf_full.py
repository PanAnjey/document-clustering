# -*- coding: utf-8 -*-
# zeroshot_typeclf_full.py
# Zero-shot Qwen3.5-4B (локальная, 4 млрд) на ВСЁМ texts_cache (2277 док.,
# грязные метки папок) -- тот же протокол, что у строк таблицы «вторая
# задача» в статье (стандартная accuracy, без dedup). Нужен один сопоставимый
# столбец рядом с облачными praxis-large / Qwen3.6-27B.

import csv
import json
from collections import Counter

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common import LABELS, MODEL_PATH, PROMPT, TEXTS_CACHE, extract_json

OUT = 'reports/zeroshot_typeclf_full.jsonl'


def main():
    rows = [r for r in csv.DictReader(open(TEXTS_CACHE, encoding='utf-8'))
            if r['cls'] in LABELS]
    print(f'{len(rows)} документов', flush=True)

    done = set()
    try:
        for l in open(OUT, encoding='utf-8'):
            done.add(json.loads(l)['i'])
    except FileNotFoundError:
        pass

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager')
    model.eval()

    cor = tot = 0
    with open(OUT, 'a', encoding='utf-8') as f:
        for i, r in enumerate(rows):
            if i in done:
                continue
            enc = tok.apply_chat_template(
                [{'role': 'user', 'content': PROMPT.format(text=r['text'][:2500])}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            p = extract_json(gen)
            pred = p.get('type') if p else None
            tot += 1
            cor += int(pred == r['cls'])
            f.write(json.dumps({'i': i, 'true': r['cls'], 'pred': pred}, ensure_ascii=False) + '\n')
            f.flush()
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(rows)}  acc~{cor / max(tot, 1):.3f}', flush=True)

    # финальный счёт по всему файлу
    allr = [json.loads(l) for l in open(OUT, encoding='utf-8')]
    n = len(allr)
    acc = sum(x['pred'] == x['true'] for x in allr) / n
    tp = Counter(); fp = Counter(); cnt = Counter()
    for x in allr:
        cnt[x['true']] += 1
        if x['pred'] == x['true']:
            tp[x['true']] += 1
        elif x['pred']:
            fp[x['pred']] += 1
    f1s = []
    for c in LABELS:
        if not cnt[c]:
            continue
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0
        rc = tp[c] / cnt[c]
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    res = {'n': n, 'accuracy': acc, 'macro_f1': sum(f1s) / len(f1s)}
    json.dump(res, open('reports/zeroshot_typeclf_full.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\nDONE  accuracy {acc:.4f}  macro-F1 {res["macro_f1"]:.4f}  (n={n})', flush=True)


if __name__ == '__main__':
    main()
