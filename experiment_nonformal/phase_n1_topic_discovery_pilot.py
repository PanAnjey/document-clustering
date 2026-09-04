# phase_n1_topic_discovery_pilot.py
# Пилот: открытие тем "с нуля" для НЕформализованных документов
# (PDF_Text + PDF_Tables + Excel_Xlsx + Word), БЕЗ каких-либо заранее
# известных тем/каталогов (в отличие от X8, где был каталог из 56 тем).
#
# Метод:
#   1. Стратифицированная выборка файлов по 4 группам форматов (Sorted/*).
#   2. Извлечение текста (формат-специфичное: fitz / Aspose.Cells / Aspose.Words).
#   3. Qwen3.5-4B СВОБОДНО придумывает короткую тему документа (2-5 слов),
#      без списка тем на выбор — zero-shot open-ended.
#   4. Эмбеддинг тем (nomic-embed) -> HDBSCAN -> слияние близких центроидов
#      (merge_by_centroids, sim>=0.92) -> c-TF-IDF топ-слова консолидированных
#      тем. Метод консолидации идентичен experiment_excel/phase_2d_llm_themes.py.
#
# Запуск:
#   python phase_n1_topic_discovery_pilot.py [--n-per-group 125] [--seed 42]

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXCEL_EXP = PROJECT_ROOT / 'experiment_excel'
if str(EXCEL_EXP) not in sys.path:
    sys.path.insert(0, str(EXCEL_EXP))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import fitz  # noqa: E402
import hdbscan  # noqa: E402

from embed_helper import encode_texts  # noqa: E402  (experiment_excel)
from phase_1_split_and_extract import normalize_tsv  # noqa: E402  (experiment_excel)
from phase_2c_themes import c_tfidf_top_words  # noqa: E402  (experiment_excel)
from phase_2d_llm_themes import merge_by_centroids  # noqa: E402  (experiment_excel)
from extractors.cells_aspose_extractor import extract_text_direct  # noqa: E402
from extractors.aspose_extractor import extract_word_aspose  # noqa: E402

MODEL_PATH = r"D:\MODELS\Transformers\Qwen_Qwen3.5-4B"
SORTED_ROOT = Path(r"D:\FileOrganizer\Sorted")
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_CSV = OUT_DIR / "n1_topic_discovery_pilot.csv"
OUT_MD = OUT_DIR / "n1_topic_discovery_pilot.md"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 24
TEXT_CHARS = 2000
MAX_LENGTH = 3000
MERGE_SIM = 0.92
# Выше, чем 0.01 в phase_2d (там N ~десятки тысяч) — компенсирует малый N пилота.
MIN_CLUSTER_FACTOR = 0.02
MIN_CLUSTER_FLOOR = 5

GROUPS = {
    'pdf_text':   [('PDF_Text', '*.pdf')],
    'pdf_tables': [('PDF_Tables', '**/*.pdf')],
    'excel_xlsx': [('Excel_Xlsx', '*.xlsx'), ('Excel_Xlsx', '*.xls')],
    'word':       [('Word_Docx', '*.docx'), ('Word_Doc', '*.doc'),
                   ('Word_Rtf', '*.rtf'), ('Word_Odt', '*.odt')],
}

SYS_PROMPT = (
    "Ты анализируешь деловой документ (счёт, акт, договор, письмо, отчёт и т.п.). "
    "Определи ГЛАВНУЮ ТЕМУ документа — о чём он по существу (предмет, услуга, "
    "товар, процесс), а НЕ формальный тип документа (не пиши 'счёт-фактура', "
    "'акт', 'договор' как тему саму по себе, если это не единственное, что "
    "можно понять из текста). "
    "Ответь КОРОТКОЙ фразой из 2-5 слов на русском языке, без пояснений, без "
    "кавычек, без точки в конце."
)


def sample_files(n_per_group, seed):
    rng = random.Random(seed)
    sample = []  # (group, path)
    for group, patterns in GROUPS.items():
        pool = []
        for subdir, pattern in patterns:
            pool.extend((SORTED_ROOT / subdir).glob(pattern))
        pool = sorted(set(pool))  # детерминированный порядок перед сэмплингом
        rng.shuffle(pool)
        picked = pool[:n_per_group]
        print(f"  {group}: пул {len(pool)}, взято {len(picked)}", flush=True)
        sample.extend((group, p) for p in picked)
    return sample


def extract_text(group, path):
    try:
        if group in ('pdf_text', 'pdf_tables'):
            d = fitz.open(str(path))
            try:
                return d[0].get_text("text") if d.page_count else ""
            finally:
                d.close()
        if group == 'excel_xlsx':
            t = extract_text_direct(path)
            return normalize_tsv(t) if t else None
        if group == 'word':
            res = extract_word_aspose(path)
            return res.get('text')
    except Exception as e:
        print(f"    extract fail {path.name}: {e}", flush=True)
        return None
    return None


def make_prompt(tok, text):
    t = (text or "").replace("\n", " ")[:TEXT_CHARS]
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Текст документа:\n{t}\n\nТЕМА:"},
    ]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-group', type=int, default=125)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print("сэмплирование файлов...", flush=True)
    sample = sample_files(args.n_per_group, args.seed)
    print(f"всего в выборке: {len(sample)}", flush=True)

    print("извлечение текста...", flush=True)
    items = []  # (group, path, text)
    t_ext0 = time.time()
    for group, path in sample:
        txt = extract_text(group, path)
        if txt and len(txt.strip()) >= 30:
            items.append((group, path, txt))
    print(f"успешно извлечено: {len(items)}/{len(sample)} ({time.time()-t_ext0:.0f}с)", flush=True)
    if not items:
        sys.exit("нет текста для классификации")

    print("загрузка Qwen3.5-4B...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"модель загружена, VRAM={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    labels_raw = []
    t0 = time.time()
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        prompts = [make_prompt(tok, it[2]) for it in batch]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LENGTH).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        for j in range(len(batch)):
            gen = out[j][inp.input_ids.shape[1]:]
            ans = tok.decode(gen, skip_special_tokens=True).strip()
            ans = ans.splitlines()[0].strip(' "\'.-—').strip()
            labels_raw.append(ans[:120])
        done = min(i + BATCH_SIZE, len(items))
        if done % (BATCH_SIZE * 5) < BATCH_SIZE or done == len(items):
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  {done}/{len(items)}  elapsed={el:.0f}s  rate={rate:.1f}/s", flush=True)

    del model
    torch.cuda.empty_cache()

    print("эмбеддинг тем...", flush=True)
    emb = encode_texts(labels_raw, batch_size=32)

    n = len(emb)
    min_cluster = max(MIN_CLUSTER_FLOOR, int(MIN_CLUSTER_FACTOR * n))
    print(f"HDBSCAN: n={n}, min_cluster_size={min_cluster}", flush=True)
    hlabels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=min_cluster, core_dist_n_jobs=-1,
    ).fit_predict(emb)
    n_raw = len(set(hlabels.tolist()) - {-1})

    hlabels = merge_by_centroids(hlabels, emb, sim_threshold=MERGE_SIM)
    real = [l for l in np.unique(hlabels) if l != -1]
    noise = int((hlabels == -1).sum())
    print(f"HDBSCAN: {n_raw} сырых -> {len(real)} тем после слияния (sim>={MERGE_SIM}), "
          f"шум {noise} ({noise/n:.1%})", flush=True)

    top_words_map = {}
    if real:
        class_texts, order = [], []
        for lab in real:
            idx = np.where(hlabels == lab)[0]
            class_texts.append(' '.join(labels_raw[i] for i in idx))
            order.append(lab)
        tops = c_tfidf_top_words(class_texts, top_n=8)
        top_words_map = dict(zip(order, tops))

    # ── CSV: по документам ──────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "file", "llm_topic", "theme_id", "top_words"])
        w.writeheader()
        for (group, path, _text), label, hl in zip(items, labels_raw, hlabels):
            w.writerow({
                "group": group, "file": path.name, "llm_topic": label,
                "theme_id": int(hl), "top_words": top_words_map.get(hl, ""),
            })

    # ── MD: сводный отчёт ────────────────────────────────────────────
    groups_arr = np.array([it[0] for it in items])
    L = [
        "# N1: открытие тем с нуля для неформализованных документов (пилот)",
        "",
        f"Выборка: {n} документов ({args.n_per_group}/группу, seed={args.seed}) "
        "из PDF_Text + PDF_Tables + Excel_Xlsx + Word (docx/doc/rtf/odt).",
        "Метод: Qwen3.5-4B свободно придумывает тему документа (zero-shot, "
        "без каталога) -> nomic-embed -> HDBSCAN -> слияние центроидов "
        f"(sim>={MERGE_SIM}) -> c-TF-IDF топ-слова.",
        "",
        "## Состав выборки по форматам",
        "",
        "| Группа | N |",
        "|---|---|",
    ]
    for g in GROUPS:
        L.append(f"| {g} | {int((groups_arr == g).sum())} |")
    L += [
        "",
        f"## Итог кластеризации",
        "",
        f"- Сырых кластеров HDBSCAN: {n_raw}",
        f"- После слияния близких центроидов (sim>={MERGE_SIM}): **{len(real)} тем**",
        f"- Шум (не вошли ни в одну тему): {noise} ({noise/n:.1%})",
        "",
        "## Темы (по убыванию размера) и состав по форматам",
        "",
        "| Тема | N | Топ-слова (LLM-фраз) | pdf_text | pdf_tables | excel_xlsx | word |",
        "|---|---|---|---|---|---|---|",
    ]
    for lab in sorted(real, key=lambda l: -int((hlabels == l).sum())):
        idx = np.where(hlabels == lab)[0]
        cnt = len(idx)
        comp = {g: int((groups_arr[idx] == g).sum()) for g in GROUPS}
        L.append(f"| {lab} | {cnt} | {top_words_map.get(lab, '')} | "
                 f"{comp['pdf_text']} | {comp['pdf_tables']} | {comp['excel_xlsx']} | {comp['word']} |")

    L += [
        "",
        "## Примеры LLM-тем внутри крупнейших кластеров",
        "",
    ]
    for lab in sorted(real, key=lambda l: -int((hlabels == l).sum()))[:10]:
        idx = np.where(hlabels == lab)[0]
        examples = [labels_raw[i] for i in idx[:6]]
        L.append(f"- **тема {lab}** ({len(idx)} док.): " + "; ".join(examples))

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"written {OUT_CSV}", flush=True)
    print(f"written {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
