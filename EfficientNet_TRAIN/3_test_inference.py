"""Тестовый прогон обученной модели на массиве изображений (ускоренный, 2 фазы).

- Загружает дообученную EfficientNet-B1 (лучшую либо финальную — по конфигу).
- Обрабатывает все изображения из INFERENCE_INPUT_DIR.
- Для каждого: определяет ориентацию (0/90/180/270) и уверенность (softmax).
  Если ориентация != 0 — поворачивает для исправления и сохраняет в CORRECTED_DIR.
  Если ориентация == 0 — копирует/сохраняет в NOT_CORRECTED_DIR без поворота.
  Угол поворота для исправления (значение для PIL.rotate, + = против часовой):
     90  -> поворот на +90  (90° против часовой = «-90» по соглашению пользователя)
     180 -> поворот на 180
     270 -> поворот на +270 (90° по часовой = «+90» по соглашению пользователя)
- После исправления ориентации (и для исправленных, и для нетронутых) применяются
  (опционально, через config.py): deskew (выравнивание перекоса) и crop borders.
- Сохраняет сводку результатов в INFERENCE_RESULTS_FILE (json) для review_app.py.

Режим --compare-only: сравнение предсказаний с ground truth из distrib (без постобработки).

ПРОИЗВОДИТЕЛЬНОСТЬ (две фазы):
  Фаза 1 — GPU: декод изображений в потоках (ThreadPool) + батчевая классификация
           (autocast/inference_mode). Сетка — дешёвая часть.
  Фаза 2 — CPU: поворот + постобработка (deskew/crop) + сохранение в пуле процессов
           (multiprocessing.Pool по INFERENCE_WORKERS). Это главная по времени часть.
  torch импортируется ЛЕНИВО (внутри функций), чтобы процессы-воркеры фазы 2
  стартовали быстро и не инициализировали CUDA.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import config as C
import image_postprocess as PP
import text_crop

Image.MAX_IMAGE_PIXELS = None  # отключить DecompressionBombError для больших сканов

ANGLE_ORDER = [0, 90, 180, 270]


def get_device():
    import torch
    pref = C.INFERENCE_DEVICE.strip().lower()
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    import torch
    import torch.nn as nn
    from torchvision import models
    fname = C.BEST_MODEL_FILENAME if C.USE_BEST_MODEL_FOR_INFERENCE else C.MODEL_FILENAME
    model_path = C.MODELS_DIR / fname
    if not model_path.exists():
        print(f"[ERROR] Модель не найдена: {model_path}. Сначала запустите train.py.", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
    model = models.efficientnet_b1(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, ckpt["num_classes"])
    # strict=False для совместимости с DataParallel (ключи с "module.")
    model.load_state_dict(ckpt["model_state"], strict=False)
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
    """Вписать в квадрат IMAGE_SIZE с сохранением пропорций + белые поля.
    ДОЛЖНО совпадать с letterbox в 2_train.py (режим val/инференс)."""
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
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if getattr(C, "INPUT_MODE", "full") == "crop":
        resize = transforms.Lambda(lambda im: text_crop.crop_text_region(im, jitter=0))
    elif getattr(C, "RESIZE_MODE", "letterbox") == "letterbox":
        resize = transforms.Lambda(_letterbox_white)
    else:
        resize = transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE))
    return transforms.Compose([resize, transforms.ToTensor(), norm])


def correct_angle(orientation: int) -> int:
    """Угол для PIL.Image.rotate (положительное = против часовой).
    Соглашение: метка X = изображение повёрнуто на X° CW, исправление = rotate(X)."""
    return orientation % 360


def build_ground_truth() -> dict[str, int]:
    """Сканирует distrib/{0,90,180,270}, строит dict filename -> angle."""
    gt = {}
    for angle in ANGLE_ORDER:
        d = C.DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS:
                    gt[f.name] = angle
    return gt


def compare_with_ground_truth(files: list[Path], predictions: dict, idx_to_angle: dict, ground_truth: dict):
    """Сравнивает предсказания с ground truth. Выводит отчёт в порядке 0,90,180,270."""
    correct = 0
    total = 0
    errors_by_class = defaultdict(list)
    confmatrix = defaultdict(lambda: defaultdict(int))
    
    for idx, (cls_idx, conf) in predictions.items():
        fname = files[idx].name
        if fname not in ground_truth:
            continue
        
        pred_angle = idx_to_angle.get(cls_idx, 0)
        true_angle = ground_truth[fname]
        total += 1
        
        confmatrix[true_angle][pred_angle] += 1
        
        if pred_angle == true_angle:
            correct += 1
        else:
            errors_by_class[true_angle].append({
                "file": fname,
                "predicted": pred_angle,
                "confidence": round(conf, 4)
            })
    
    if total == 0:
        print("[WARN] Нет файлов для сравнения с ground truth")
        return None
    
    accuracy = correct / total
    print(f"\n=== Сравнение с ground truth ===")
    print(f"Всего файлов: {len(files)}  |  Совпадений с distrib: {total}")
    print(f"\nОбщая accuracy: {accuracy:.2%}")
    
    print(f"\nPer-class metrics (порядок 0, 90, 180, 270):")
    per_class_stats = {}
    for angle in ANGLE_ORDER:
        tp = confmatrix[angle][angle]
        fp = sum(confmatrix[true_a][angle] for true_a in ANGLE_ORDER if true_a != angle)
        fn = sum(confmatrix[angle][pred_a] for pred_a in ANGLE_ORDER if pred_a != angle)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        n_true = sum(confmatrix[angle][pred_a] for pred_a in ANGLE_ORDER)
        n_errors = len(errors_by_class[angle])
        
        per_class_stats[angle] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_true": n_true,
            "n_correct": tp,
            "n_errors": n_errors
        }
        
        print(f"  {angle:3d}°: precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  (N={n_true}, ошибок={n_errors})")
    
    print(f"\nConfusion matrix (rows=truth, cols=predicted, порядок 0,90,180,270):")
    header = "         pred: " + "  ".join(f"{a:3d}°" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}°  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    total_errors = total - correct
    print(f"\nОшибок: {total_errors} ({total_errors/total:.2%})")
    
    if errors_by_class:
        print(f"\nПримеры ошибок (до 5 на класс):")
        for angle in ANGLE_ORDER:
            errs = errors_by_class[angle][:5]
            if errs:
                print(f"  {angle}° (истинный класс):")
                for e in errs:
                    print(f"    {e['file'][:60]:60s} -> предсказано {e['predicted']:3d}° (conf={e['confidence']:.3f})")
    
    report = {
        "total_files": len(files),
        "matched_with_gt": total,
        "accuracy": round(accuracy, 4),
        "per_class": per_class_stats,
        "confusion_matrix": {str(true_a): {str(pred_a): confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER} for true_a in ANGLE_ORDER},
        "total_errors": total_errors,
        "error_rate": round(total_errors / total, 4)
    }
    
    return report


# ---------------- Фаза 1: GPU-классификация ----------------
def _decode_one(args):
    """Поток: декод файла + трансформация в тензор. Возвращает (idx, tensor|None, err)."""
    idx, path, tfm = args
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
        return idx, tfm(img), None
    except Exception as e:  # noqa: BLE001
        return idx, None, str(e)


def run_inference_phase(files, model, device, tfm):
    """Батчевая GPU-классификация с параллельным декодом в потоках.
    Возвращает: predictions {idx: (class_idx, conf)}, decode_errors {idx: err}."""
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
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    image.save(dst)


def _postprocess_worker(task):
    """Процесс: загрузка full-res, поворот ориентации, deskew/crop, сохранение.
    task = (path, orient, conf, corrected_dir, not_corrected_dir, review_dir, threshold, abstain).

    Ненадёжные (conf < threshold):
      abstain=True  -> НЕ поворачивать (оставить как есть) -> not_corrected,
                       + копия в review_dir (отладка), флаги low_confidence/abstained.
      abstain=False -> применить предсказанный поворот -> review_dir."""
    path_str, orient, conf, corrected_dir, not_corrected_dir, review_dir, threshold, abstain = task
    fpath = Path(path_str)
    try:
        with Image.open(fpath) as im:
            img = im.convert("RGB")

        low_conf = bool(threshold and threshold > 0 and conf < threshold)
        abstained = bool(low_conf and abstain)
        applied = 0 if abstained else orient  # при abstain поворот НЕ применяется

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

        if low_conf and not abstain:
            target_dir = Path(review_dir)
        else:
            target_dir = Path(corrected_dir) if corrected else Path(not_corrected_dir)
        _save_replace(target_dir / fpath.name, out_img)
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
    
    parser = argparse.ArgumentParser(description="Тестовый прогон EfficientNet модели для ориентации сканов.")
    parser.add_argument("--compare-only", action="store_true",
                        help="Только сравнение с ground truth из distrib (без постобработки).")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Устройство: {device}")
    model, idx_to_angle = load_model(device)
    tfm = build_transform()

    threshold = float(getattr(C, "INFERENCE_CONFIDENCE_THRESHOLD", 0.0) or 0.0)
    review_dir = getattr(C, "INFERENCE_REVIEW_DIR", C.NOT_CORRECTED_DIR)
    abstain = bool(getattr(C, "INFERENCE_ABSTAIN", True))

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
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

    if args.compare_only:
        ground_truth = build_ground_truth()
        print(f"Ground truth из distrib: {len(ground_truth)} файлов")
        
        compare_report = compare_with_ground_truth(files, predictions, idx_to_angle, ground_truth)
        
        if compare_report:
            compare_report["model"] = C.BEST_MODEL_FILENAME if C.USE_BEST_MODEL_FOR_INFERENCE else C.MODEL_FILENAME
            compare_report["input_dir"] = str(C.INFERENCE_INPUT_DIR)
            
            report_path = C.MODELS_DIR / "compare_report.json"
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(compare_report, fh, ensure_ascii=False, indent=2)
            print(f"\nОтчёт сравнения сохранён: {report_path}")
        
        if decode_errors:
            print(f"\nОшибок декода: {len(decode_errors)}")
        
        return

    # освобождаем GPU перед запуском CPU-пула
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Фаза 2: CPU-постобработка ---
    C.CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    C.NOT_CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    if threshold > 0:
        Path(review_dir).mkdir(parents=True, exist_ok=True)
    
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

    for idx, err in decode_errors.items():
        results.append({"file": files[idx].name, "error": err})
        n_err += 1

    with open(C.INFERENCE_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump({
            "input_dir": str(C.INFERENCE_INPUT_DIR),
            "corrected_dir": str(C.CORRECTED_DIR),
            "not_corrected_dir": str(C.NOT_CORRECTED_DIR),
            "model": C.BEST_MODEL_FILENAME if C.USE_BEST_MODEL_FOR_INFERENCE else C.MODEL_FILENAME,
            "results": results,
        }, fh, ensure_ascii=False, indent=2)

    total = len(files)
    dt = time.time() - t0
    print("\n=== Тестовый прогон завершён ===")
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
    print(f"Результаты: {C.INFERENCE_RESULTS_FILE}")


if __name__ == "__main__":
    main()
