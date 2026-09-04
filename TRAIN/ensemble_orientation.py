"""Ансамблирование методов определения ориентации: ViT + EfficientNet + Tesseract OSD.

Для каждого файла из PDF_Images:
1. Получает предсказание от ViT-Large 384
2. Получает предсказание от EfficientNet-B1
3. Получает предсказание от Tesseract OSD (orientation and script detection)
4. Голосует (majority vote из 3 методов)
5. Сравнивает с ground truth из distrib

Выводит отчёт с per-class метриками и confusion matrix.
"""
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # отключить DecompressionBombWarning для больших сканов

# Добавляем пути к скриптам
VIT_DIR = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN")
EFFNET_DIR = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\EfficientNet_TRAIN")

sys.path.insert(0, str(VIT_DIR))
import config as vit_config

sys.path.insert(0, str(EFFNET_DIR))
# Переименовываем чтобы не конфликтовать с vit_config
import importlib.util
spec = importlib.util.spec_from_file_location("effnet_config", EFFNET_DIR / "config.py")
effnet_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(effnet_config)

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

ANGLE_ORDER = [0, 90, 180, 270]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif")


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Вписать в квадрат size×size с сохранением пропорций + белые поля."""
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def load_vit_model():
    """Загружает ViT-Large 384."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = vit_config.MODELS_DIR / vit_config.BEST_MODEL_FILENAME
    
    print(f"Загрузка ViT-Large: {model_path.name}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Lambda(lambda im: letterbox(im, 384)),
        transforms.ToTensor(),
        norm
    ])
    
    # ImageFolder order: ['0', '180', '270', '90']
    idx_to_angle = {0: 0, 1: 180, 2: 270, 3: 90}
    
    return model, transform, device, idx_to_angle


def load_efficientnet_model():
    """Загружает EfficientNet-B1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = effnet_config.MODELS_DIR / effnet_config.BEST_MODEL_FILENAME
    
    print(f"Загрузка EfficientNet-B1: {model_path.name}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
    model = models.efficientnet_b1(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Lambda(lambda im: letterbox(im, 240)),
        transforms.ToTensor(),
        norm
    ])
    
    idx_to_class = ckpt.get("idx_to_class")
    if idx_to_class is not None:
        idx_to_angle = {int(k): int(v) for k, v in idx_to_class.items()}
    else:
        idx_to_angle = {i: effnet_config.CLASS_NAMES[i] for i in range(len(effnet_config.CLASS_NAMES))}
    
    return model, transform, device, idx_to_angle


def predict_vit(model, transform, device, idx_to_angle, img: Image.Image) -> tuple[int, float]:
    """Предсказание ViT-Large."""
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        out = model(img_tensor)
        prob = torch.softmax(out.float(), dim=1)
        conf, pred = prob.max(dim=1)
    
    pred_angle = idx_to_angle[int(pred[0].item())]
    confidence = float(conf[0].item())
    return pred_angle, confidence


def predict_efficientnet(model, transform, device, idx_to_angle, img: Image.Image) -> tuple[int, float]:
    """Предсказание EfficientNet-B1."""
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        out = model(img_tensor)
        prob = torch.softmax(out.float(), dim=1)
        conf, pred = prob.max(dim=1)
    
    pred_angle = idx_to_angle[int(pred[0].item())]
    confidence = float(conf[0].item())
    return pred_angle, confidence


def predict_tesseract(img_path: Path) -> tuple[int, float]:
    """Предсказание Tesseract OSD."""
    try:
        result = subprocess.run(
            [TESSERACT_PATH, str(img_path), "stdout", "--psm", "0", "-l", "rus+eng"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout + result.stderr
        
        # Парсим Orientation in degrees
        orientation = 0
        confidence = 0.0
        
        for line in output.split("\n"):
            if "Orientation in degrees:" in line:
                orientation = int(line.split(":")[1].strip())
            elif "Orientation confidence:" in line:
                confidence = float(line.split(":")[1].strip())
        
        # Нормализуем confidence (Tesseract даёт 0-30, нормализуем к 0-1)
        confidence = min(confidence / 30.0, 1.0)
        
        return orientation, confidence
    except Exception as e:
        print(f"  [WARN] Tesseract error for {img_path.name}: {e}")
        return 0, 0.0


def majority_vote(predictions: list[int]) -> int:
    """Голосование большинством из 3 предсказаний."""
    counts = defaultdict(int)
    for pred in predictions:
        counts[pred] += 1
    
    # Возвращаем наиболее частое предсказание
    return max(counts, key=counts.get)


def build_ground_truth() -> dict[str, int]:
    """Сканирует distrib/{0,90,180,270}, строит dict filename -> angle."""
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    gt[f.name] = angle
    return gt


def process_file(fpath: Path, vit_model, vit_transform, vit_device, vit_idx_to_angle,
                 effnet_model, effnet_transform, effnet_device, effnet_idx_to_angle,
                 ground_truth: dict) -> dict:
    """Обработка одного файла: все 3 метода + голосование."""
    fname = fpath.name
    
    if fname not in ground_truth:
        return None
    
    true_angle = ground_truth[fname]
    
    # Загрузка изображения
    try:
        img = Image.open(fpath).convert("RGB")
    except Exception as e:
        return {"file": fname, "error": str(e)}
    
    # Предсказания
    vit_pred, vit_conf = predict_vit(vit_model, vit_transform, vit_device, vit_idx_to_angle, img)
    effnet_pred, effnet_conf = predict_efficientnet(effnet_model, effnet_transform, effnet_device, effnet_idx_to_angle, img)
    tess_pred, tess_conf = predict_tesseract(fpath)
    
    # Голосование
    ensemble_pred = majority_vote([vit_pred, effnet_pred, tess_pred])
    
    return {
        "file": fname,
        "true_angle": true_angle,
        "vit_pred": vit_pred,
        "vit_conf": round(vit_conf, 3),
        "effnet_pred": effnet_pred,
        "effnet_conf": round(effnet_conf, 3),
        "tess_pred": tess_pred,
        "tess_conf": round(tess_conf, 3),
        "ensemble_pred": ensemble_pred,
        "vit_correct": vit_pred == true_angle,
        "effnet_correct": effnet_pred == true_angle,
        "tess_correct": tess_pred == true_angle,
        "ensemble_correct": ensemble_pred == true_angle
    }


def main():
    print("=== Ансамблирование методов определения ориентации ===\n")
    
    # Загрузка моделей
    vit_model, vit_transform, vit_device, vit_idx_to_angle = load_vit_model()
    effnet_model, effnet_transform, effnet_device, effnet_idx_to_angle = load_efficientnet_model()
    
    # Ground truth
    ground_truth = build_ground_truth()
    print(f"Ground truth из distrib: {len(ground_truth)} файлов\n")
    
    # Файлы для обработки
    files = [f for f in PDF_IMAGES_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    
    print(f"Файлов для обработки: {len(files)}\n")
    
    # Обработка
    results = []
    
    t0 = time.time()
    
    # Параллельная обработка через ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for fpath in files:
            future = executor.submit(
                process_file, fpath,
                vit_model, vit_transform, vit_device, vit_idx_to_angle,
                effnet_model, effnet_transform, effnet_device, effnet_idx_to_angle,
                ground_truth
            )
            futures.append(future)
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result and "error" not in result:
                results.append(result)
            
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"Обработано {i}/{len(files)} ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    
    # Подсчёт метрик
    total = len(results)
    correct_vit = sum(1 for r in results if r["vit_correct"])
    correct_effnet = sum(1 for r in results if r["effnet_correct"])
    correct_tess = sum(1 for r in results if r["tess_correct"])
    correct_ensemble = sum(1 for r in results if r["ensemble_correct"])
    
    # Отчёт
    print(f"\n=== Результаты (всего файлов: {total}) ===\n")
    
    acc_vit = correct_vit / total
    acc_effnet = correct_effnet / total
    acc_tess = correct_tess / total
    acc_ensemble = correct_ensemble / total
    
    print(f"ViT-Large:        {acc_vit:.2%} ({correct_vit}/{total})")
    print(f"EfficientNet-B1:  {acc_effnet:.2%} ({correct_effnet}/{total})")
    print(f"Tesseract OSD:    {acc_tess:.2%} ({correct_tess}/{total})")
    print(f"Ensemble (vote):  {acc_ensemble:.2%} ({correct_ensemble}/{total})")
    
    print(f"\nВремя: {elapsed:.1f}s ({total/elapsed:.1f} файл/с)")
    
    # Confusion matrix для ensemble
    print("\n=== Confusion matrix (Ensemble) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["ensemble_pred"]] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}°" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}°  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    # Сохранение результатов
    report_path = vit_config.MODELS_DIR / "ensemble_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "vit_accuracy": round(acc_vit, 4),
            "effnet_accuracy": round(acc_effnet, 4),
            "tesseract_accuracy": round(acc_tess, 4),
            "ensemble_accuracy": round(acc_ensemble, 4),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
