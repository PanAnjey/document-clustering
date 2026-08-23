"""Тестовый прогон дообученной (retrain) модели на массиве изображений.

- Загружает последнюю дообученную модель из MODELS_DIR.
- Обрабатывает все изображения из INFERENCE_INPUT_DIR.
- Сохраняет результаты в inference_results_retrain.json для сравнения с базовой моделью.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import config as C
import image_postprocess as PP
import text_crop

Image.MAX_IMAGE_PIXELS = None  # отключить DecompressionBombError для больших сканов


def get_device():
    """Определяет устройство (GPU/CPU) для инференса."""
    import torch
    pref = C.INFERENCE_DEVICE.strip().lower()
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_retrain_model(device):
    """Загружает последнюю дообученную модель."""
    import torch
    import torch.nn as nn
    import timm
    
    # Находим последнюю retrain модель по имени
    model_prefix = C.RETRAIN_MODEL_PREFIX if hasattr(C, 'RETRAIN_MODEL_PREFIX') else "orientation_vit_base_retrain"
    
    models_dir = Path(C.MODELS_DIR)
    retrain_models = list(models_dir.glob(f"{model_prefix}_*.pth"))
    
    if not retrain_models:
        print(f"[ERROR] Дообученная модель не найдена в {models_dir}")
        print(f"Ожидалось имя с префиксом: {model_prefix}_*.pth")
        sys.exit(1)
    
    # Сортируем по имени (последняя версия будет последней в списке)
    retrain_models.sort(key=lambda x: x.name)
    model_path = retrain_models[-1]  # Берём последнюю
    
    print(f"[INFO] Загружена дообученная модель: {model_path.name}")
    
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    # Определяем размер входа ИЗ ИМЕНИ ФАЙЛА (а не из config.py!)
    model_name_str = model_path.name.lower()
    if "384" in model_name_str:
        input_size = 384
    elif "224" in model_name_str:
        input_size = 224
    else:
        # По умолчанию используем IMAGE_SIZE из config.py
        input_size = C.IMAGE_SIZE
    
    print(f"[INFO] Размер входа определён из имени файла: {input_size}x{input_size}")
    
    # Определяем архитектуру модели по имени в конфиге
    if "vit_large" in C.MODEL_NAME.lower():
        model = timm.create_model(f'vit_large_patch16_{input_size}', pretrained=False, num_classes=0)
    else:
        model = timm.create_model(f'vit_base_patch16_{input_size}', pretrained=False, num_classes=0)

    # Загружаем веса с игнорированием mismatch для positional embeddings
    try:
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[INFO] Веса загружены (strict=False из-за размера входа)")
    except Exception as e:
        print(f"[WARN] Ошибка загрузки: {e}")
        print("[INFO] Продолжаем обучение с частичной загрузкой весов")
    
    model.to(device).eval()

    class_to_idx = ckpt.get("class_to_idx")
    idx_to_class = ckpt.get("idx_to_class")
    
    if idx_to_class is not None:
        idx_to_angle = {int(k): int(v) for k, v in idx_to_class.items()}
    elif class_to_idx is not None:
        idx_to_angle = {int(v): int(k) for k, v in class_to_idx.items()}
    else:
        idx_to_angle = {i: C.CLASS_NAMES[i] for i in range(len(C.CLASS_NAMES))}
    
    return model, idx_to_angle


def _letterbox_white(img: Image.Image) -> Image.Image:
    """Вписать в квадрат IMAGE_SIZE с сохранением пропорций + белые поля."""
    size = C.IMAGE_SIZE
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_transform():
    """Строит трансформацию для инференса."""
    from torchvision import transforms
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Вход ОБЯЗАН совпадать с обучением (config.INPUT_MODE / RESIZE_MODE).
    if getattr(C, "INPUT_MODE", "full") == "crop":
        resize = transforms.Lambda(lambda im: text_crop.crop_text_region(im, jitter=0))
    elif getattr(C, "RESIZE_MODE", "letterbox") == "letterbox":
        resize = transforms.Lambda(_letterbox_white)
    else:
        resize = transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE))
    
    return transforms.Compose([resize, transforms.ToTensor(), norm])


def correct_angle(orientation: int) -> int:
    """Угол для PIL.Image.rotate (положительное = против часовой)."""
    return orientation % 360


# ---------------- Фаза 1: GPU-классификация ----------------

def _decode_one(args):
    """Поток: декод файла + трансформация в тензор."""
    idx, path, tfm = args
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
        return idx, tfm(img), None
    except Exception as e:  # noqa: BLE001
        return idx, None, str(e)


def run_inference_phase(files, model, device, tfm):
    """Батчевая GPU-классификация с параллельным декодом в потоках."""
    import torch
    
    predictions = {}
    decode_errors = {}
    
    bs = max(1, C.INFERENCE_BATCH_SIZE)
    workers = max(1, C.INFERENCE_WORKERS)
    use_amp = (device.type == "cuda" and C.USE_AMP)

    with ThreadPoolExecutor(max_workers=workers) as ex, torch.inference_mode():
        for start in range(0, len(files), bs):
            batch_idx = list(range(start, min(start + bs, len(files))))
            decoded = list(ex.map(_decode_one, [(i, files[i], tfm) for i in batch_idx]))
            
            tensors, ids = [], []
            for idx, t, err in decoded:
                if t is None:
                    decode_errors[idx] = err or "decode failed"
                else:
                    tensors.append(t)
                    ids.append(idx)
            
            if not tensors:
                continue
            
            xb = torch.stack(tensors).to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(xb)
            
            prob = torch.softmax(out.float(), dim=1)
            conf, pred = prob.max(dim=1)
            
            for j, idx in enumerate(ids):
                predictions[idx] = (int(pred[j].item()), float(conf[j].item()))
    
    return predictions, decode_errors


# ---------------- Фаза 2: CPU-постобработка (пул процессов) ----------------

def _save_replace(dst: Path, image):
    """Сохраняет изображение в dst, удаляя существующий файл."""
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    image.save(dst)


def _postprocess_worker(task):
    """Процесс: загрузка full-res, поворот ориентации, deskew/crop, сохранение."""
    path_str, orient, conf, corrected_dir, not_corrected_dir, review_dir, threshold, abstain = task
    
    fpath = Path(path_str)
    
    try:
        with Image.open(fpath) as im:
            img = im.convert("RGB")

        low_conf = bool(threshold and threshold > 0 and conf < threshold)
        abstained = bool(low_conf and abstain)
        applied = 0 if abstained else orient

        if applied == 0:
            out_img = img
            corrected = False
        else:
            out_img = img.rotate(applied % 360, expand=True)
            corrected = True
        
        try:
            out_img, pp_info = PP.postprocess(out_img)
        except Exception as e:  # noqa: BLE001
            pp_info = {"deskew_angle": 0.0, "cropped": False, "pp_error": str(e)}

        # маршрутизация
        if low_conf and not abstain:
            target_dir = Path(review_dir)
        else:
            target_dir = Path(corrected_dir) if corrected else Path(not_corrected_dir)
        
        _save_replace(target_dir / fpath.name, out_img)
        
        # при abstain — копия в review для отладки/сбора в retrain
        if abstained and review_dir:
            _save_replace(Path(review_dir) / fpath.name, out_img)

        return {
            "file": fpath.name,
            "predicted_orientation": orient,
            "applied_orientation": applied,
            "confidence": round(conf, 4),
            "correction_angle": applied % 360,
            "corrected": corrected,
            "low_confidence": low_conf,
            "abstained": abstained,
            "deskew_angle": round(pp_info.get("deskew_angle", 0.0), 3),
            "borders_cropped": bool(pp_info.get("cropped", False)),
        }
    except Exception as e:  # noqa: BLE001
        return {"file": fpath.name, "error": str(e)}


def main():
    import torch
    
    device = get_device()
    print(f"Устройство: {device}")
    
    model, idx_to_angle = load_retrain_model(device)
    tfm = build_transform()

    threshold = float(getattr(C, "INFERENCE_CONFIDENCE_THRESHOLD", 0.0) or 0.0)
    review_dir = getattr(C, "INFERENCE_REVIEW_DIR", C.NOT_CORRECTED_DIR)
    abstain = bool(getattr(C, "INFERENCE_ABSTAIN", True))

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    C.NOT_CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    
    if threshold > 0:
        Path(review_dir).mkdir(parents=True, exist_ok=True)
    
    if not C.INFERENCE_INPUT_DIR.exists():
        print(f"[ERROR] Входная директория не существует: {C.INFERENCE_INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    files = [f for f in C.INFERENCE_INPUT_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
    
    if not files:
        print(f"[WARN] Во входной директории нет изображений: {C.INFERENCE_INPUT_DIR}")
        return
    
    _inp = getattr(C, "INPUT_MODE", "full")
    if _inp == "crop":
        _win = f"adaptive(g={C.CROP_TARGET_GLYPH})" if getattr(C, "CROP_ADAPTIVE", True) else f"win={C.CROP_WINDOW}"
        _inp_desc = f"crop[{_win}]->{C.IMAGE_SIZE}, fallback_full={getattr(C, 'CROP_FALLBACK_FULL', True)}"
    else:
        _inp_desc = f"full/{getattr(C, 'RESIZE_MODE', 'letterbox')}"
    
    print(f"Файлов: {len(files)}  |  вход={_inp_desc}  batch={C.INFERENCE_BATCH_SIZE}  "
          f"workers={C.INFERENCE_WORKERS}  conf_threshold={threshold or 'off'}")

    # --- Фаза 1: GPU-классификация ---
    t0 = time.time()
    predictions, decode_errors = run_inference_phase(files, model, device, tfm)
    t_infer = time.time() - t0
    
    print(f"Фаза 1 (классификация): {len(predictions)} файлов за {t_infer:.1f}s")

    # освобождаем GPU перед запуском CPU-пула
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Фаза 2: CPU-постобработка ---
    tasks = []
    for idx, (cls_idx, conf) in predictions.items():
        orient = int(idx_to_angle.get(cls_idx,
                     C.CLASS_NAMES[cls_idx] if cls_idx < len(C.CLASS_NAMES) else 0))
        tasks.append((str(files[idx]), orient, conf,
                      str(C.CORRECTED_DIR), str(C.NOT_CORRECTED_DIR),
                      str(review_dir), threshold, abstain))

    results = []
    n_corr = n_not = n_err = n_review = n_abstain = 0
    
    t1 = time.time()
    workers = max(1, C.INFERENCE_WORKERS)
    
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_postprocess_worker, tasks, chunksize=4), 1):
            results.append(res)
            
            if res.get("error"):
                n_err += 1
            elif res.get("abstained"):
                n_abstain += 1
            elif res.get("low_confidence"):
                n_review += 1
            elif res.get("corrected"):
                n_corr += 1
            else:
                n_not += 1
            
            if i % 50 == 0 or i == len(tasks):
                print(f"  постобработка {i}/{len(tasks)}  (corrected={n_corr}, "
                      f"not_corrected={n_not}, abstain={n_abstain}, review={n_review}, err={n_err})")
    
    t_pp = time.time() - t1

    # ошибки декода (фаза 1) — в общий отчёт
    for idx, err in decode_errors.items():
        results.append({"file": files[idx].name, "error": err})
        n_err += 1

    # Сохраняем результаты с суффиксом _retrain для отличия от базовой модели
    inference_results_file = C.MODELS_DIR / "inference_results_retrain.json"
    
    with open(inference_results_file, "w", encoding="utf-8") as fh:
        json.dump({
            "input_dir": str(C.INFERENCE_INPUT_DIR),
            "corrected_dir": str(C.CORRECTED_DIR),
            "not_corrected_dir": str(C.NOT_CORRECTED_DIR),
            "model": f"{C.RETRAIN_MODEL_PREFIX}_latest.pth",  # Последняя retrain модель
            "results": results,
        }, fh, ensure_ascii=False, indent=2)

    total = len(files)
    dt = time.time() - t0
    
    print("\n=== Тестовый прогон дообученной модели завершён ===")
    print(f"Всего файлов: {total}")
    print(f"  скорректировано: {n_corr}  -> {C.CORRECTED_DIR}")
    print(f"  без коррекции:  {n_not}  -> {C.NOT_CORRECTED_DIR}")
    
    if threshold > 0:
        if abstain:
            print(f"  abstain (не повёрнуто, conf<{threshold}): {n_abstain}  -> {C.NOT_CORRECTED_DIR} (+копия {review_dir})")
        else:
            print(f"  на проверку (conf<{threshold}): {n_review}  -> {review_dir}")
    
    print(f"  ошибок:         {n_err}")
    print(f"Время: классификация {t_infer:.1f}s + постобработка {t_pp:.1f}s = {dt:.1f}s "
          f"({total / dt:.1f} файл/с)")
    print(f"Результаты: {inference_results_file}")


if __name__ == "__main__":
    main()
