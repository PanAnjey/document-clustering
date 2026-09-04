# phase_o1_run.py
# Фаза O1: рендер 1-й страницы документов + прогон OCR-движков.
#
# Выборка: N_PDF_TEXT документов pdf_text (есть эталонный текст fitz)
# + N_PDF_SCAN реальных сканов pdf_scan (без эталона).
#
# Запуск (venv_ocr):
#   python phase_o1_run.py                  # все движки
#   python phase_o1_run.py --engine paddle  # только PaddleOCR
#   python phase_o1_run.py --limit 20       # отладка
#
# Выход: results/o1_results.jsonl (resume: обработанные пропускаются),
#        renders/*.png

import argparse
import io
import json
import sys
import time
from pathlib import Path

import fitz
import psycopg2

from oc_config import (  # noqa: E402
    DB_URL, DPI, ENGINES, MIN_GT_CHARS, N_PDF_SCAN, N_PDF_TEXT,
    RENDER_DIR, RESULTS_DIR, RESULTS_JSONL,
)

DOCS_SET = RESULTS_DIR / 'docs_set.json'


def fetch_docs():
    """Единый набор документов для ВСЕХ движков (парный дизайн):
    при первом запуске — случайная выборка, сохраняется в results/docs_set.json;
    последующие запуски (новые движки) берут тот же набор из файла."""
    if DOCS_SET.exists():
        docs = json.loads(io.open(DOCS_SET, encoding='utf-8').read())
        print(f"набор из {DOCS_SET.name}: {len(docs)} документов")
        return docs
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    docs = []
    for source, n in (('pdf_text', N_PDF_TEXT), ('pdf_scan', N_PDF_SCAN)):
        cur.execute("""
            SELECT id, file_path FROM documents
            WHERE format_type = %s
            ORDER BY random() LIMIT %s
        """, (source, n * 2))
        rows = [(i, p) for i, p in cur.fetchall() if p and Path(p).exists()]
        docs.extend({'doc_id': i, 'file_path': p, 'source': source}
                    for i, p in rows[:n])
    conn.close()
    with io.open(DOCS_SET, 'w', encoding='utf-8') as f:
        f.write(json.dumps(docs, ensure_ascii=False))
    print(f"новый набор сохранён в {DOCS_SET.name}: {len(docs)} документов")
    return docs


def render_page1(pdf_path: str, out_png: Path) -> str:
    """Рендер 1-й страницы + возврат эталонного текста fitz."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        gt = page.get_text().strip()
        pix = page.get_pixmap(dpi=DPI)
        pix.save(str(out_png))
        return gt
    finally:
        doc.close()


def load_done():
    done = set()
    if RESULTS_JSONL.exists():
        with io.open(RESULTS_JSONL, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for eng in rec.get('outputs', {}):
                        done.add((rec['doc_id'], eng))
                except Exception:  # noqa: BLE001
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='all', choices=['all'] + ENGINES)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    engines = ENGINES if args.engine == 'all' else [args.engine]

    RENDER_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    docs = fetch_docs()
    if args.limit:
        docs = docs[:args.limit]
    print(f"документов: {len(docs)} "
          f"(pdf_text {sum(d['source'] == 'pdf_text' for d in docs)}, "
          f"pdf_scan {sum(d['source'] == 'pdf_scan' for d in docs)}), "
          f"движки: {engines}")

    done = load_done()
    engine_objs = {}

    def get_eng(name):
        if name not in engine_objs:
            print(f"  init engine: {name}", flush=True)
            if name == 'tesseract':
                from engines.oc_tesseract import get_engine
            elif name == 'paddle':
                from engines.oc_paddle import get_engine
            elif name == 'deepseek':
                from engines.oc_deepseek import get_engine
            elif name == 'deepseek2':
                from engines.oc_deepseek2 import get_engine
            elif name == 'nanonets':
                from engines.oc_nanonets import get_engine
            elif name == 'qwen3vl':
                from engines.oc_qwen3vl import get_engine
            elif name == 'qwen3vl4b':
                from engines.oc_qwen3vl4b import get_engine
            elif name == 'gemma4':
                from engines.oc_gemma4 import get_engine
            else:
                raise ValueError(name)
            engine_objs[name] = get_engine()
            print(f"  engine ready: {name}", flush=True)
        return engine_objs[name]

    n_new = 0
    with io.open(RESULTS_JSONL, 'a', encoding='utf-8') as out:
        for k, d in enumerate(docs):
            png = RENDER_DIR / f"{d['source']}_{d['doc_id']}.png"
            try:
                gt = render_page1(d['file_path'], png)
            except Exception as e:  # noqa: BLE001
                print(f"  [{k + 1}/{len(docs)}] doc {d['doc_id']}: "
                      f"render failed: {e}")
                continue
            if d['source'] == 'pdf_text' and len(gt) < MIN_GT_CHARS:
                continue

            outputs = {}
            for eng in engines:
                if (d['doc_id'], eng) in done:
                    continue
                t0 = time.time()
                try:
                    text = get_eng(eng).ocr(str(png))
                    outputs[eng] = {'text': text,
                                    'sec': round(time.time() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    outputs[eng] = {'text': '', 'sec': round(time.time() - t0, 2),
                                    'error': f"{type(e).__name__}: {e}"}
            if not outputs:
                continue
            rec = {'doc_id': d['doc_id'], 'source': d['source'],
                   'file_name': Path(d['file_path']).name,
                   'gt_text': gt, 'outputs': outputs}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            n_new += 1
            if n_new % 10 == 0 or n_new == 1:
                print(f"  [{k + 1}/{len(docs)}] записано {n_new}", flush=True)

    print(f"done. новых записей: {n_new} → {RESULTS_JSONL}")


if __name__ == '__main__':
    main()
