r"""Гейт паспортов (ПДн) — комбинированный: MRZ (детерминированно) + visual (nomic-embed-vision).

Стратегия для гарантии (~100% recall):
  1. MRZ (OCR на 4 поворотах) — жёсткий детерминированный признак заграна.
  2. Visual (nomic-embed-vision) — эталонные эмбеддинги подтверждённых паспортов,
     косинусная близость с порогом (по умолч. 0.80).
  3. Если сработал ЛЮБОЙ — карантин + ручное подтверждение (объём мал, <10/2мес).
  4. Порог visual настроен на recall: лучше ложное срабатывание, чем пропуск.

Создание эталонов visual:
    python passport_gate.py --build-refs

Прогон:
    python passport_gate.py --scan D:\path           # dry-run
    python passport_gate.py --scan D:\path --move     # карантин + review
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ---- Конфиг ----
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_LANG = "rus+eng"
MODEL_DIR = Path(r"D:\MODELS\Transformers\nomic-ai_nomic-embed-vision-v1.5")
DEVICE = "cuda:1"
REFS_DIR = Path(r"D:\FileOrganizer\TRAIN\passport_refs")
REFS_EMB_FILE = REFS_DIR / "embeddings.npy"
REFS_META_FILE = REFS_DIR / "metadata.json"
QUARANTINE_DIR = Path(r"D:\FileOrganizer\TRAIN\quarantine_passport")
QUARANTINE_REVIEW_DIR = Path(r"D:\FileOrganizer\TRAIN\quarantine_passport_review")
OCR_MAXSIDE = 3000
SIMILARITY_THRESHOLD = 0.80  # high recall; ложные срабатывания отсеются при ручном подтверждении
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif")


# ========== MRZ (OCR) ==========
_MRZ_PRUS = re.compile(r"p[<ckск]\s?rus", re.IGNORECASE)


def _ocr_text(img, upscale=True):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    if upscale:
        w, h = img.size
        fac = min(3.0, OCR_MAXSIDE / float(max(w, h)))
        if fac > 1.0:
            img = img.resize((int(w * fac), int(h * fac)), Image.LANCZOS)
    return pytesseract.image_to_string(img, lang=TESSERACT_LANG)


def detect_mrz(path):
    """MRZ-детектор на 4 поворотах. Возвращает (is_mrz: bool, evidence: str)."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return False, ""
    for rot in (0, 90, 180, 270):
        im = img if rot == 0 else img.rotate(-rot, expand=True)
        txt = _ocr_text(im)
        if _MRZ_PRUS.search(txt):
            return True, "mrz_prus"
        for ln in txt.splitlines():
            s = ln.strip()
            if s.count("<") >= 8:
                return True, "mrz_chevrons"
            up = re.sub(r"[^A-Z0-9<]", "", s.upper())
            if len(up) >= 40 and "RUS" in s.upper():
                return True, "mrz_block"
    return False, ""


# ========== VISUAL (nomic-embed-vision) ==========
_visual_model = None
_visual_processor = None
_is_patched = False


def _apply_patch():
    global _is_patched
    if _is_patched:
        return
    from transformers import PreTrainedModel
    _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
    def _patched_mark(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_mark(self, loading_info)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark
    _orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device
    def _patched_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
            self.all_tied_weights_keys = {}
        return _orig_move(self, *args, **kwargs)
    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
    _is_patched = True


def _ensure_visual_model():
    global _visual_model, _visual_processor
    if _visual_model is not None:
        return
    import torch
    from transformers import AutoModel, CLIPImageProcessor
    _apply_patch()
    _visual_model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(DEVICE)
    _visual_model.eval()
    _visual_processor = CLIPImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    torch.cuda.empty_cache()


def _image_embedding(img):
    _ensure_visual_model()
    import torch
    inputs = _visual_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _visual_model(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output.cpu().numpy()[0]
    else:
        emb = out.last_hidden_state[:, 0].cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / np.maximum(norm, 1e-8)


def _load_refs():
    if not REFS_EMB_FILE.exists():
        return None, []
    return np.load(REFS_EMB_FILE), (json.loads(REFS_META_FILE.read_text("utf-8"))
                                     if REFS_META_FILE.exists() else [])


def detect_visual(path):
    refs, meta = _load_refs()
    if refs is None or len(refs) == 0:
        return 0.0, None
    img = Image.open(path).convert("RGB")
    emb = _image_embedding(img)
    sims = refs @ emb
    best = int(np.argmax(sims))
    return float(sims[best]), meta[best]["name"] if meta and best < len(meta) else None


# ========== КОМБИНИРОВАННЫЙ ДЕТЕКТОР ==========
def detect(path, threshold=None):
    """Комбинированный детектор. Возвращает dict с вердиктом и evidence."""
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    fn = os.path.basename(path)
    # 1) MRZ
    mrz, ev_mrz = detect_mrz(path)
    if mrz:
        return {"file": fn, "path": path, "verdict": "passport",
                "method": "mrz", "similarity": 1.0, "best_match": ev_mrz}
    # 2) Visual
    try:
        sim, match = detect_visual(path)
    except Exception as e:
        return {"file": fn, "path": path, "verdict": "error", "error": str(e)}
    if sim >= threshold:
        return {"file": fn, "path": path, "verdict": "passport",
                "method": "visual", "similarity": round(sim, 4), "best_match": match}
    if sim >= threshold - 0.12:
        return {"file": fn, "path": path, "verdict": "maybe",
                "method": "visual_low", "similarity": round(sim, 4), "best_match": match}
    return {"file": fn, "path": path, "verdict": "no", "method": "none",
            "similarity": round(sim, 4)}


# ========== BUILD REFS ==========
def find_passport_files(folder):
    folder = Path(folder)
    names = [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
             and ("паспорт" in p.stem.lower() or "pasport" in p.stem.lower())]
    seen, uniq = set(), []
    for n in names:
        b = os.path.basename(n)
        if b not in seen:
            seen.add(b); uniq.append(Path(n))
    return uniq


def build_refs(folder=None):
    if folder is None:
        import config as C
        folder = C.DISTRIB_DIR
    files = find_passport_files(folder)
    if not files:
        print(f"[ERROR] не найдено паспортов в {folder}")
        return
    print(f"Найдено паспортов: {len(files)}")
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    embs, meta = [], []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  SKIP {f.name}: {e}"); continue
        emb = _image_embedding(img)
        embs.append(emb)
        meta.append({"path": str(f), "name": f.name, "size": img.size})
        print(f"  OK {f.name}")
    np.save(REFS_EMB_FILE, np.array(embs))
    REFS_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nСохранено эталонов: {len(embs)}")


# ========== WORKER (для Pool) ==========
def _worker(path):
    return detect(path)


# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser(description="Гейт паспортов: MRZ + visual (nomic-embed-vision).")
    ap.add_argument("--build-refs", action="store_true")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.build_refs:
        build_refs(args.folder)
        return

    if args.scan:
        folder = args.folder
        if not folder:
            print("[ERROR] укажите --folder", file=sys.stderr)
            sys.exit(1)
        fp = Path(folder)
        files = [str(f) for f in fp.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            print(f"[WARN] нет изображений в {folder}"); return
        print(f"Сканирование {folder} ({len(files)} файлов) threshold={args.threshold} workers={args.workers}")
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        n_p = n_m = n_no = n_e = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(_worker, files), 1):
                v = res["verdict"]
                if v == "passport":
                    n_p += 1
                    print(f"  [ПАСПОРТ] {res['method']} sim={res.get('similarity','-')} | {res['file'][:40]}")
                    dst = QUARANTINE_DIR / res["file"]
                elif v == "maybe":
                    n_m += 1
                    print(f"  [maybe]   {res['method']} sim={res.get('similarity','-')} | {res['file'][:40]}")
                    dst = QUARANTINE_REVIEW_DIR / res["file"]
                elif v == "error":
                    n_e += 1
                    print(f"  [ERROR]   {res.get('error','')} | {res['file'][:40]}")
                    continue
                else:
                    n_no += 1; continue
                if args.move:
                    shutil.move(res["path"], dst)
                elif args.copy:
                    shutil.copy2(res["path"], dst)
        print(f"\nИтог: паспорт={n_p}  возможно={n_m}  не паспорт={n_no}  ошибок={n_e}")
        print(f"  карантин:        {QUARANTINE_DIR}")
        print(f"  карантин/review: {QUARANTINE_REVIEW_DIR}")


if __name__ == "__main__":
    main()
