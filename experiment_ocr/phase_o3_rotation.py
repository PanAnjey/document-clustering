# phase_o3_rotation.py
# Фаза O3: устойчивость OCR-движков к повороту скана (90°/180°).
#
# Мотивация: визуальный энкодер Nanonets-OCR2-3B — Qwen2.5-VL (ViT),
# который в экспериментах проекта по ориентации (TRAIN/SESSION_CONTENT.md)
# показал слабость на классе 180° (81.7% после LoRA; Qwen3-VL с SigLIP-2 —
# 100%). Проверяем эмпирически: те же 20 документов, повёрнутые на 90°
# и 180°, BoW-F1 против эталона fitz.
#
# Запуск (по процессу на движок):
#   venv_ocr:   python phase_o3_rotation.py --engine nanonets (paddle/tesseract)
#   venv_dsocr: python phase_o3_rotation.py --engine deepseek
#
# Выход: results/rotation_test.jsonl (resume).

import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image

from oc_config import RENDER_DIR, RESULTS_DIR  # noqa: E402
from oc_metrics import bow_f1  # noqa: E402
from oc_textnorm import normalize  # noqa: E402

N_DOCS = 20
ANGLES = [90, 180]
ROT_DIR = RENDER_DIR / 'rot'
OUT_JSONL = RESULTS_DIR / 'rotation_test.jsonl'


def load_gt():
    """GT из основных результатов (первые N_DOCS pdf_text с длиной ≥300)."""
    src = RESULTS_DIR / 'o1_results.jsonl'
    gt = {}
    for line in io.open(src, encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'pdf_text' and len(normalize(r['gt_text'])) >= 300:
            gt[r['doc_id']] = r['gt_text']
        if len(gt) >= N_DOCS:
            break
    return gt


def make_rotations(doc_id):
    """Создаёт повёрнутые PNG из renders/pdf_text_{doc_id}.png."""
    base = RENDER_DIR / f'pdf_text_{doc_id}.png'
    outs = {}
    ROT_DIR.mkdir(exist_ok=True)
    img = Image.open(base)
    for ang in ANGLES:
        out = ROT_DIR / f'pdf_text_{doc_id}_rot{ang}.png'
        if not out.exists():
            img.rotate(-ang, expand=True).save(out)
        outs[ang] = str(out)
    return outs


def load_done():
    done = set()
    if OUT_JSONL.exists():
        for line in io.open(OUT_JSONL, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['doc_id'], r['angle'], r['engine']))
            except Exception:  # noqa: BLE001
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True,
                    choices=['tesseract', 'paddle', 'deepseek', 'nanonets'])
    args = ap.parse_args()

    gt = load_gt()
    print(f"документов с GT: {len(gt)}, углы: {ANGLES}, движок: {args.engine}")

    if args.engine == 'tesseract':
        from engines.oc_tesseract import get_engine
    elif args.engine == 'paddle':
        from engines.oc_paddle import get_engine
    elif args.engine == 'deepseek':
        from engines.oc_deepseek import get_engine
    else:
        from engines.oc_nanonets import get_engine
    eng = get_engine()

    done = load_done()
    n_new = 0
    with io.open(OUT_JSONL, 'a', encoding='utf-8') as out:
        for doc_id, gt_text in gt.items():
            for ang, png in make_rotations(doc_id).items():
                if (doc_id, ang, args.engine) in done:
                    continue
                t0 = time.time()
                try:
                    text = eng.ocr(png)
                    rec = {'doc_id': doc_id, 'angle': ang,
                           'engine': args.engine, 'f1': bow_f1(gt_text, text),
                           'chars': len(text), 'sec': round(time.time() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    rec = {'doc_id': doc_id, 'angle': ang,
                           'engine': args.engine, 'f1': 0.0, 'chars': 0,
                           'sec': round(time.time() - t0, 2),
                           'error': f"{type(e).__name__}: {e}"}
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                n_new += 1
                print(f"  {doc_id} rot{ang}: F1 {rec['f1']:.3f} "
                      f"({rec['chars']} chars, {rec['sec']}s)", flush=True)
    print(f"done. новых: {n_new} → {OUT_JSONL}")


if __name__ == '__main__':
    main()
