# phase_o4b_orientation_large.py
# Фаза O4b: РЕПРЕЗЕНТАТИВНЫЙ парный тест классификаторов ориентации.
#
# Набор: 500 случайных pdf_text × 4 угла (0/90/180/270, PIL rotate, рендер
# 150 dpi — для ориентации достаточно) = 2000 предсказаний на классификатор.
# Gemma 3 4B (zero-shot) — парная подвыборка первых 100 документов.
#
# Классификаторы: vit_large / effnet / pp_lcnet (из phase_o4) + gemma3.
#
# Запуск:
#   base python (timm 1.0.27):  python phase_o4b_orientation_large.py --cls vit_large|effnet
#   venv_ocr:                   python phase_o4b_orientation_large.py --cls pp_lcnet|gemma3
#
# Итог: results/orientation_cls_large.jsonl (resume).

import argparse
import io
import json
import re
import time
from pathlib import Path

import fitz
import psycopg2
from PIL import Image

from oc_config import DB_URL, RENDER_DIR, RESULTS_DIR  # noqa: E402

N_DOCS = 500
GEMMA_DOCS = 100
ANGLES = [0, 90, 180, 270]
DPI = 150
EVAL_SET = RESULTS_DIR / 'orientation_eval_set.json'
IMG_DIR = RENDER_DIR / 'o4_large'
OUT_JSONL = RESULTS_DIR / 'orientation_cls_large.jsonl'

GEMMA_PATH = r"D:\MODELS\Transformers\google_gemma-3-4b-it"
GEMMA_PROMPT = (
    "You are an expert document image analyst. Determine the clockwise "
    "rotation angle of this document page: 0, 90, 180, or 270 degrees. "
    "Answer with a single number only."
)


def fetch_eval_set():
    if EVAL_SET.exists():
        docs = json.loads(io.open(EVAL_SET, encoding='utf-8').read())
        print(f"набор из {EVAL_SET.name}: {len(docs)} документов")
        return docs
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, file_path FROM documents
        WHERE format_type = 'pdf_text'
        ORDER BY random() LIMIT %s
    """, (N_DOCS * 2,))
    rows = [(i, p) for i, p in cur.fetchall() if p and Path(p).exists()]
    conn.close()
    docs = [{'doc_id': i, 'file_path': p} for i, p in rows[:N_DOCS]]
    with io.open(EVAL_SET, 'w', encoding='utf-8') as f:
        f.write(json.dumps(docs, ensure_ascii=False))
    print(f"новый набор сохранён: {len(docs)} документов")
    return docs


def ensure_images(doc_id, file_path):
    """doc_id → {угол: путь}. Рендер 150 dpi + 3 поворота (по требованию)."""
    IMG_DIR.mkdir(exist_ok=True)
    outs = {}
    base = IMG_DIR / f"{doc_id}_0.png"
    if not base.exists():
        doc = fitz.open(file_path)
        try:
            pix = doc[0].get_pixmap(dpi=DPI)
            pix.save(str(base))
        finally:
            doc.close()
    img = Image.open(base)
    outs[0] = str(base)
    for ang in (90, 180, 270):
        out = IMG_DIR / f"{doc_id}_{ang}.png"
        if not out.exists():
            img.rotate(-ang, expand=True).save(out)
        outs[ang] = str(out)
    return outs


class Gemma3Classifier:
    name = 'gemma3'

    def __init__(self):
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        _ = torch.tensor([0], device='cuda:0')
        torch.cuda.synchronize(0)
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(GEMMA_PATH)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            GEMMA_PATH, torch_dtype=torch.bfloat16).to('cuda:0').eval()

    @staticmethod
    def parse_angle(resp: str) -> int:
        m = re.search(r'\b(270|180|90)\b', resp)
        if m:
            return int(m.group(1))
        m = re.search(r'\b0\b', resp)
        return int(m.group(0)) if m else -1

    def predict(self, png_path):
        img = Image.open(png_path).convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': GEMMA_PROMPT}]}]
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img],
                           return_tensors='pt').to('cuda:0')
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=16,
                                      do_sample=False)
        resp = self.proc.batch_decode(
            out[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)[0]
        return self.parse_angle(resp), 0.5


def load_done():
    done = set()
    if OUT_JSONL.exists():
        for line in io.open(OUT_JSONL, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['doc_id'], r['angle'], r['cls']))
            except Exception:  # noqa: BLE001
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', required=True,
                    choices=['vit_large', 'effnet', 'pp_lcnet', 'gemma3'])
    args = ap.parse_args()

    docs = fetch_eval_set()
    if args.cls == 'gemma3':
        docs = docs[:GEMMA_DOCS]
        clf = Gemma3Classifier()
    elif args.cls == 'vit_large':
        from phase_o4_orientation import VitClassifier
        clf = VitClassifier()
    elif args.cls == 'effnet':
        from phase_o4_orientation import EffnetClassifier
        clf = EffnetClassifier()
    else:
        from phase_o4_orientation import PpLcnetClassifier
        clf = PpLcnetClassifier()

    print(f"документов: {len(docs)}, классификатор: {args.cls}")
    done = load_done()
    n_new = n_ok = 0
    t_start = time.time()
    with io.open(OUT_JSONL, 'a', encoding='utf-8') as out:
        for k, d in enumerate(docs):
            try:
                images = ensure_images(d['doc_id'], d['file_path'])
            except Exception as e:  # noqa: BLE001
                print(f"  doc {d['doc_id']}: render failed: {e}")
                continue
            for true_ang, png in images.items():
                if (d['doc_id'], true_ang, args.cls) in done:
                    continue
                t0 = time.time()
                try:
                    pred, conf = clf.predict(png)
                    rec = {'doc_id': d['doc_id'], 'angle': true_ang,
                           'cls': args.cls, 'pred': pred,
                           'conf': round(conf, 4),
                           'sec': round(time.time() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    rec = {'doc_id': d['doc_id'], 'angle': true_ang,
                           'cls': args.cls, 'pred': -1, 'conf': 0.0,
                           'sec': round(time.time() - t0, 2),
                           'error': f"{type(e).__name__}: {e}"}
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                n_new += 1
                n_ok += int(rec['pred'] == true_ang)
                if n_new % 100 == 0:
                    dt = time.time() - t_start
                    print(f"  {n_new} прогонов, acc {n_ok / n_new:.1%}, "
                          f"{n_new / dt:.1f} прогонов/с", flush=True)
    print(f"done. новых: {n_new}, acc {n_ok / max(n_new, 1):.2%} → {OUT_JSONL}")


if __name__ == '__main__':
    main()
