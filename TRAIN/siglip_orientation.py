"""Определение ориентации документов с помощью SigLIP2 (zero-shot классификация).

Использует модель google/siglip2-so400m-patch14-384 для zero-shot классификации
ориентации документов на основе текстовых промптов.

Сравнивает результаты с ViT-Large 384.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoProcessor

Image.MAX_IMAGE_PIXELS = None

VIT_DIR = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN")
sys.path.insert(0, str(VIT_DIR))
import config as C

SIGLIP_MODEL_PATH = Path(r"D:\MODELS\Transformers\siglip2-so400m-patch14-384")
PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

ANGLE_ORDER = [0, 90, 180, 270]

# Текстовые промпты для zero-shot классификации ориентации
ORIENTATION_PROMPTS = [
    "a document page oriented at 0 degrees, upright, readable from top to bottom",
    "a document page oriented at 90 degrees, rotated clockwise, readable from right to left",
    "a document page oriented at 180 degrees, upside down, readable from bottom to top",
    "a document page oriented at 270 degrees, rotated counterclockwise, readable from left to right"
]


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
    model_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    
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


def load_siglip_model():
    """Загружает SigLIP2 модель."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Загрузка SigLIP2: {SIGLIP_MODEL_PATH}")
    model = AutoModel.from_pretrained(str(SIGLIP_MODEL_PATH))
    processor = AutoProcessor.from_pretrained(str(SIGLIP_MODEL_PATH))
    
    model.to(device).eval()
    
    return model, processor, device


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


def predict_siglip(model, processor, device, img: Image.Image) -> tuple[int, float]:
    """Предсказание SigLIP2 (zero-shot)."""
    try:
        # Подготовка изображения
        inputs = processor(text=ORIENTATION_PROMPTS, images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        # SigLIP возвращает logits (similarity scores)
        logits_per_image = outputs.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)
        
        # Находим ориентацию с максимальной вероятностью
        pred_idx = int(probs[0].argmax().item())
        pred_angle = ANGLE_ORDER[pred_idx]
        confidence = float(probs[0][pred_idx].item())
        
        return pred_angle, confidence
    except Exception as e:
        print(f"  [WARN] SigLIP error: {e}")
        return 0, 0.0


def build_ground_truth() -> dict[str, int]:
    """Сканирует distrib/{0,90,180,270}, строит dict filename -> angle."""
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS:
                    gt[f.name] = angle
    return gt


def main():
    print("=== Сравнение SigLIP2 с ViT-Large 384 ===\n")
    
    # Загрузка моделей
    vit_model, vit_transform, vit_device, vit_idx_to_angle = load_vit_model()
    siglip_model, siglip_processor, siglip_device = load_siglip_model()
    
    # Ground truth
    ground_truth = build_ground_truth()
    print(f"Ground truth из distrib: {len(ground_truth)} файлов\n")
    
    # Файлы для обработки
    files = [f for f in PDF_IMAGES_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
    
    print(f"Файлов для обработки: {len(files)}\n")
    
    # Обработка
    results = []
    
    t0 = time.time()
    
    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        
        if fname not in ground_truth:
            continue
        
        true_angle = ground_truth[fname]
        
        # Загрузка изображения
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            continue
        
        # Предсказания
        vit_pred, vit_conf = predict_vit(vit_model, vit_transform, vit_device, vit_idx_to_angle, img)
        siglip_pred, siglip_conf = predict_siglip(siglip_model, siglip_processor, siglip_device, img)
        
        results.append({
            "file": fname,
            "true_angle": true_angle,
            "vit_pred": vit_pred,
            "vit_conf": round(vit_conf, 3),
            "siglip_pred": siglip_pred,
            "siglip_conf": round(siglip_conf, 3),
            "vit_correct": vit_pred == true_angle,
            "siglip_correct": siglip_pred == true_angle
        })
        
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"Обработано {i}/{len(files)} ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    total = len(results)
    
    # Подсчёт метрик
    correct_vit = sum(1 for r in results if r["vit_correct"])
    correct_siglip = sum(1 for r in results if r["siglip_correct"])
    
    # Отчёт
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    
    acc_vit = correct_vit / total
    acc_siglip = correct_siglip / total
    
    print(f"ViT-Large 384:     {acc_vit:.2%} ({correct_vit}/{total})")
    print(f"SigLIP2:           {acc_siglip:.2%} ({correct_siglip}/{total})")
    
    # Per-class метрики
    print(f"\n=== Per-class метрики ===\n")
    
    print("ViT-Large 384:")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in results if r["true_angle"] == angle]
        if angle_results:
            correct = sum(1 for r in angle_results if r["vit_correct"])
            total_angle = len(angle_results)
            print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")
    
    print("\nSigLIP2:")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in results if r["true_angle"] == angle]
        if angle_results:
            correct = sum(1 for r in angle_results if r["siglip_correct"])
            total_angle = len(angle_results)
            print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")
    
    # Confusion matrix для SigLIP
    print(f"\n=== Confusion matrix (SigLIP2) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["siglip_pred"]] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    # Ошибки SigLIP
    siglip_errors = [r for r in results if not r["siglip_correct"]]
    print(f"\n=== Ошибки SigLIP2 ({len(siglip_errors)} файлов) ===")
    for r in siglip_errors[:20]:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  siglip={r['siglip_pred']:3d}({r['siglip_conf']:.2f})  vit={r['vit_pred']:3d}({r['vit_conf']:.2f})")
    
    # Сравнение: где SigLIP лучше ViT
    siglip_better = [r for r in results if r["siglip_correct"] and not r["vit_correct"]]
    vit_better = [r for r in results if r["vit_correct"] and not r["siglip_correct"]]
    
    print(f"\n=== Сравнение ===")
    print(f"SigLIP лучше ViT: {len(siglip_better)} файлов")
    print(f"ViT лучше SigLIP: {len(vit_better)} файлов")
    
    if siglip_better:
        print(f"\nФайлы, где SigLIP прав, а ViT ошибается:")
        for r in siglip_better[:10]:
            print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  siglip={r['siglip_pred']:3d}  vit={r['vit_pred']:3d}")
    
    if vit_better:
        print(f"\nФайлы, где ViT прав, а SigLIP ошибается:")
        for r in vit_better[:10]:
            print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  siglip={r['siglip_pred']:3d}  vit={r['vit_pred']:3d}")
    
    # Сохранение результатов
    report_path = C.MODELS_DIR / "siglip_vs_vit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "vit_accuracy": round(acc_vit, 4),
            "siglip_accuracy": round(acc_siglip, 4),
            "siglip_better_count": len(siglip_better),
            "vit_better_count": len(vit_better),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
