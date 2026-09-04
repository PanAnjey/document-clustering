# phase_m5_ocr.py
# Фаза M5: устойчивость моделей к OCR-шуму (сценарий сканов).
#
# Берём N документов pdf_text из БД (documents), рендерим 1-ю страницу
# (fitz, 200 dpi) → Tesseract OCR (rus+eng) → сравниваем эмбеддинг
# OCR-текста с эмбеддингом исходного текста той же страницы:
#   - cos(orig, ocr) — насколько OCR-версия близка к оригиналу;
#   - identity top-1/top-5 — находит ли OCR-эмбеддинг свой документ
#     среди всех N оригиналов (метрика «документ всё ещё узнаваем»).
#
# Запуск: python phase_m5_ocr.py [--model KEY] [--device cuda:0] [--limit 50]

import argparse
import io
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import psycopg2
import pytesseract
from PIL import Image

from em_config import DB_URL, MODELS, OCR_DPI, OCR_N_DOCS, REPORTS_DIR  # noqa: E402
from em_model_loader import EmbeddingModel  # noqa: E402
from config import cfg  # noqa: E402

pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_PATH
OCR_LANG = getattr(cfg, 'TESSERACT_LANG', 'rus+eng')


def fetch_docs(n: int) -> pd.DataFrame:
    # NB: колонка documents.text для pdf_text пуста (текст в БД не хранится) —
    # фильтрация по длине текста выполняется ниже через fitz (orig >= 100).
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql("""
        SELECT id, file_path FROM documents
        WHERE format_type = 'pdf_text'
        ORDER BY random() LIMIT %s
    """, conn, params=(n * 2,))  # запас на битые/отсутствующие файлы
    conn.close()
    # В БД file_path указывает на D:\FileOrganizer\Sorted\... — эта папка
    # больше не существует, файлы физически лежат в Sorted_golden (см.
    # 2026-08-27: переименование/заморозка датасета). Подмена только для
    # этого OCR-теста, БД/остальной пайплайн не трогаем.
    df['file_path'] = df['file_path'].apply(
        lambda p: p.replace('\\Sorted\\', '\\Sorted_golden\\') if p else p)
    df = df[df['file_path'].apply(lambda p: p and Path(p).exists())]
    return df.head(n)


def page1_text_and_ocr(path: str):
    """(текст 1-й страницы через fitz, OCR 1-й страницы через Tesseract)."""
    doc = fitz.open(path)
    try:
        page = doc[0]
        orig = page.get_text().strip()
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        ocr = pytesseract.image_to_string(img, lang=OCR_LANG).strip()
        return orig, ocr
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    n = args.limit or OCR_N_DOCS
    print(f"fetching {n} pdf_text документов...")
    df = fetch_docs(n)
    print(f"  {len(df)} существующих файлов")

    pairs, kept_ids = [], []
    for k, row in enumerate(df.itertuples()):
        try:
            orig, ocr = page1_text_and_ocr(row.file_path)
            if len(orig) >= 100 and len(ocr) >= 50:
                pairs.append((orig, ocr))
                kept_ids.append(int(row.id))
        except Exception as e:  # noqa: BLE001
            print(f"  doc {row.id}: {type(e).__name__}: {e}")
        if (k + 1) % 25 == 0:
            print(f"  OCR: {k + 1}/{len(df)} (годных {len(pairs)})")
    print(f"годных пар (orig>=100, ocr>=50 символов): {len(pairs)}")
    if not pairs:
        print("нет годных пар — выход")
        return

    keys = [args.model] if args.model else list(MODELS)
    results = []
    for key in keys:
        print(f"\n=== {key} ===")
        with EmbeddingModel(key, device=args.device) as em:
            orig_emb = em.encode([p[0] for p in pairs])
            ocr_emb = em.encode([p[1] for p in pairs])
        sims = ocr_emb @ orig_emb.T  # (N, N), строки — OCR, столбцы — orig
        diag = np.diag(sims)
        order = np.argsort(-sims, axis=1)
        top1 = float(np.mean([i in order[i, :1] for i in range(len(pairs))]))
        top5 = float(np.mean([i in order[i, :5] for i in range(len(pairs))]))
        res = {'model': key, 'dim': MODELS[key]['dim'], 'n': len(pairs),
               'cos_mean': float(diag.mean()), 'cos_median': float(np.median(diag)),
               'cos_min': float(diag.min()), 'id_top1': top1, 'id_top5': top5}
        results.append(res)
        print(f"  cos(orig,ocr) avg {res['cos_mean']:.3f} median {res['cos_median']:.3f} "
              f"min {res['cos_min']:.3f} | identity top-1 {top1:.1%} top-5 {top5:.1%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm5_ocr.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# M5: устойчивость к OCR-шуму (pdf_text → рендер → Tesseract rus+eng)', '']
    lines.append(f'Пар документов: {len(pairs)} (текст 1-й страницы vs OCR 1-й страницы)')
    lines.append('')
    lines.append('| Модель | dim | cos avg | cos median | cos min | identity top-1 | top-5 |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in results:
        lines.append(f"| {r['model']} | {r['dim']} | {r['cos_mean']:.3f} | "
                     f"{r['cos_median']:.3f} | {r['cos_min']:.3f} | "
                     f"{r['id_top1']:.1%} | {r['id_top5']:.1%} |")
    out = REPORTS_DIR / 'm5_ocr.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nwritten {out}")


if __name__ == '__main__':
    main()
