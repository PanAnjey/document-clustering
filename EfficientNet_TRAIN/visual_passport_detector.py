r"""Визуальный детектор паспортов РФ на основе nomic-embed-vision.

Принцип: паспорта имеют характерную визуальную структуру (фото + поля + раскладка),
которую nomic-embed-vision фиксирует в 768-dim embedding'е. Для новых документов
вычисляется embedding и сравнивается по косинусной близости с эталонными embedding'ами
подтверждённых паспортов. При близости выше порога -> карантин + ручное подтверждение.

Использует ту же модель, что и основной пайплайн (nomic-embed-vision-v1.5 на cuda:1,
CLIPImageProcessor, 768-dim, L2-normalized), поэтому работает без загрузки дублирующей
модели при наличии EmbeddingEngine.

Создание эталонов:
    python visual_passport_detector.py --build-refs    # сканирует distrib на паспорта
    python visual_passport_detector.py --build-refs --folder D:\path\to\passports

Прогон:
    python visual_passport_detector.py --scan D:\FileOrganizer\Sorted  # dry-run отчёт
    python visual_passport_detector.py --scan --move                    # +карантин
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# импорты torch/transformers — ленивые (не загружать при импорте модуля)

# ---- Конфиг (вынести в config.py при интеграции) ----
MODEL_DIR = Path(r"D:\MODELS\Transformers\nomic-ai_nomic-embed-vision-v1.5")
DEVICE = "cuda:1"
EMBEDDING_DIM = 768
REFS_DIR = Path(r"D:\FileOrganizer\TRAIN\passport_refs")
REFS_EMB_FILE = REFS_DIR / "embeddings.npy"
REFS_META_FILE = REFS_DIR / "metadata.json"
QUARANTINE_DIR = Path(r"D:\FileOrganizer\TRAIN\quarantine_passport")
QUARANTINE_REVIEW_DIR = Path(r"D:\FileOrganizer\TRAIN\quarantine_passport_review")
SIMILARITY_THRESHOLD = 0.78   # высокий recall (низкий порог), ложные срабатывания отсеются
# при ручном подтверждении
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif")


# ---------- model (lazy) ----------
_model = None
_processor = None


def _ensure_model():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoModel, CLIPImageProcessor, PreTrainedModel
    # monkey-patch для nomic-embed-vision (идентично embeddings_engine.py)
    _orig_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized
    def _patched_mark_tied(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_mark_tied(self, loading_info)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied
    _orig_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device
    def _patched_move_missing(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
            self.all_tied_weights_keys = {}
        return _orig_move_missing(self, *args, **kwargs)
    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move_missing

    _model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(DEVICE)
    _model.eval()
    _processor = CLIPImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # прогреваем модель (1 forward pass)
    _ = _model(**_processor(images=[Image.new("RGB", (224, 224))], return_tensors="pt").to(DEVICE))
    print(f"[visual_passport] модель загружена на {DEVICE}")


def _normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, keepdims=True)
    return emb / np.maximum(n, 1e-8)


# ---------- embedding ----------
def image_embedding(img: Image.Image) -> np.ndarray:
    """Единичный эмбеддинг изображения. 768-dim, L2-нормализован."""
    _ensure_model()
    import torch
    inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output.cpu().numpy()[0]
    else:
        emb = out.last_hidden_state[:, 0].cpu().numpy()[0]
    return _normalize(emb)


# ---------- reference set ----------
def find_passport_files(folder: str | Path) -> list[Path]:
    """Найти файлы с «паспорт»/«pasport» в имени во всех поддиректориях."""
    folder = Path(folder)
    names = [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
             and ("паспорт" in p.stem.lower() or "pasport" in p.stem.lower())]
    # удалим дубликаты (одно имя файла может встретиться в distrib и retrain)
    seen = set()
    uniq = []
    for n in names:
        base = os.path.basename(n)
        if base not in seen:
            seen.add(base)
            uniq.append(Path(n))
    return uniq


def build_references(folder: str | Path | None = None):
    """Найти все паспорта в folder (или в distrib) и сохранить embedding'и."""
    if folder is None:
        import config as C
        folder = C.DISTRIB_DIR
    files = find_passport_files(folder)
    if not files:
        print(f"[ERROR] не найдено паспортов в {folder} (ищи по имени 'паспорт'/*pasport*)")
        return
    print(f"Найдено паспортов: {len(files)}")
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    embs, meta = [], []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        emb = image_embedding(img)
        embs.append(emb)
        meta.append({"path": str(f), "name": f.name, "size": img.size})
        print(f"  OK {f.name}")
    np.save(REFS_EMB_FILE, np.array(embs))
    REFS_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено эталонов: {len(embs)}")
    print(f"  {REFS_EMB_FILE}")
    print(f"  {REFS_META_FILE}")


def load_references() -> tuple[np.ndarray, list[dict]]:
    """Загрузить эталонные embedding'и и метаданные."""
    if not REFS_EMB_FILE.exists():
        print(f"[ERROR] эталонов нет. Запустите --build-refs", file=sys.stderr)
        sys.exit(1)
    embs = np.load(REFS_EMB_FILE)
    meta = json.loads(REFS_META_FILE.read_text(encoding="utf-8")) if REFS_META_FILE.exists() else []
    return embs, meta


# ---------- detection ----------
def detect_passport_visual(path: str, threshold: float | None = None) -> dict:
    """Проверить, похож ли документ визуально на паспорт.

    Returns:
        {"file": str, "verdict": "passport"|"maybe"|"no",
         "similarity": float, "best_match": str|null}
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return {"file": Path(path).name, "verdict": "error", "error": str(e)}

    emb = image_embedding(img)
    refs, meta = load_references()
    sims = refs @ emb  # L2-norm -> dot == cosine
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_name = meta[best_idx]["name"] if meta and best_idx < len(meta) else None

    if best_sim >= threshold:
        verdict = "passport"
    elif best_sim >= threshold - 0.12:
        verdict = "maybe"
    else:
        verdict = "no"

    return {
        "file": Path(path).name,
        "path": str(path),
        "verdict": verdict,
        "similarity": round(best_sim, 4),
        "best_match": best_name,
    }


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Визуальный детектор паспортов (nomic-embed-vision).")
    ap.add_argument("--build-refs", action="store_true", help="построить эталонные embedding'и")
    ap.add_argument("--folder", default=None, help="папка для --build-refs или --scan")
    ap.add_argument("--scan", action="store_true", help="сканировать папку на паспорта")
    ap.add_argument("--move", action="store_true", help="переместить детектированные в карантин")
    ap.add_argument("--copy", action="store_true", help="копировать в карантин")
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = ap.parse_args()

    if args.build_refs:
        build_references(args.folder)
        return

    if args.scan:
        folder = args.folder
        if not folder:
            print("[ERROR] укажите --folder для --scan", file=sys.stderr)
            sys.exit(1)
        files = [f for f in Path(folder).iterdir()
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            print(f"[WARN] нет изображений в {folder}")
            return
        print(f"Сканирование {folder} ({len(files)} файлов) threshold={args.threshold}")

        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

        n_p = n_m = n_no = n_e = 0
        for f in files:
            res = detect_passport_visual(str(f), args.threshold)
            v = res["verdict"]
            if v == "passport":
                n_p += 1
                dst = QUARANTINE_DIR / f.name
            elif v == "maybe":
                n_m += 1
                dst = QUARANTINE_REVIEW_DIR / f.name
            else:
                n_no += 1
                continue
            print(f"  [{v:9s}] sim={res['similarity']:.3f} match={res['best_match'][:30] if res['best_match'] else '?'} | {f.name[:42]}")
            if args.move:
                shutil.move(res["path"], dst)
            elif args.copy:
                shutil.copy2(res["path"], dst)

        print(f"\nИтог: паспорт={n_p}  возможно={n_m}  не_паспорт={n_no}  ошибок={n_e}")
        return


if __name__ == "__main__":
    main()
