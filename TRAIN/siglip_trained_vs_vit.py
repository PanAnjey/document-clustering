"""Сравнение обученной SigLIP2 с ViT-Large 384.

Загружает обученные модели и сравнивает их точность на тестовом наборе.
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

PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

ANGLE_ORDER = [0, 90, 180, 270]


def letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


class SigLIPClassifier(nn.Module):
    def __init__(self, model_path: str, num_classes: int):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(model_path)
        hidden_size = self.siglip.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.siglip.vision_model(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def load_vit_model():
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
    
    idx_to_angle = {0: 0, 1: 180, 2: 270, 3: 90}
    
    return model, transform, device, idx_to_angle


def load_siglip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = C.MODELS_DIR / "orientation_siglip2_best.pth"
    
    if not model_path.exists():
        print(f"[ERROR] Модель SigLIP2 не найдена: {model_path}")
        print("Сначала запустите 2_train_siglip.py")
        sys.exit(1)
    
    print(f"Загрузка SigLIP2: {model_path.name}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    siglip_base_path = r"D:\MODELS\Transformers\siglip2-so400m-patch14-384"
    model = SigLIPClassifier(siglip_base_path, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    
    processor = AutoProcessor.from_pretrained(siglip_base_path)
    
    idx_to_class = ckpt.get("idx_to_class")
    if idx_to_class is not None:
        idx_to_angle = {int(k): int(v) for k, v in idx_to_class.items()}
    else:
        idx_to_angle = {i: ANGLE_ORDER[i] for i in range(len(ANGLE_ORDER))}
    
    return model, processor, device, idx_to_angle


def predict_vit(model, transform, device, idx_to_angle, img: Image.Image) -> tuple[int, float]:
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        out = model(img_tensor)
        prob = torch.softmax(out.float(), dim=1)
        conf, pred = prob.max(dim=1)
    
    pred_angle = idx_to_angle[int(pred[0].item())]
    confidence = float(conf[0].item())
    return pred_angle, confidence


def predict_siglip(model, processor, device, idx_to_angle, img: Image.Image) -> tuple[int, float]:
    try:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        with torch.inference_mode():
            out = model(pixel_values)
            prob = torch.softmax(out.float(), dim=1)
            conf, pred = prob.max(dim=1)
        
        pred_angle = idx_to_angle[int(pred[0].item())]
        confidence = float(conf[0].item())
        return pred_angle, confidence
    except Exception as e:
        print(f"  [WARN] SigLIP error: {e}")
        return 0, 0.0


def build_ground_truth() -> dict[str, int]:
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS:
                    gt[f.name] = angle
    return gt


def main():
    print("=== Сравнение SigLIP2 (обученная) с ViT-Large 384 ===\n")
    
    vit_model, vit_transform, vit_device, vit_idx_to_angle = load_vit_model()
    siglip_model, siglip_processor, siglip_device, siglip_idx_to_angle = load_siglip_model()
    
    ground_truth = build_ground_truth()
    print(f"Ground truth из distrib: {len(ground_truth)} файлов\n")
    
    files = [f for f in PDF_IMAGES_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
    
    print(f"Файлов для обработки: {len(files)}\n")
    
    results = []
    t0 = time.time()
    
    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        
        if fname not in ground_truth:
            continue
        
        true_angle = ground_truth[fname]
        
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue
        
        vit_pred, vit_conf = predict_vit(vit_model, vit_transform, vit_device, vit_idx_to_angle, img)
        siglip_pred, siglip_conf = predict_siglip(siglip_model, siglip_processor, siglip_device, siglip_idx_to_angle, img)
        
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
    
    correct_vit = sum(1 for r in results if r["vit_correct"])
    correct_siglip = sum(1 for r in results if r["siglip_correct"])
    
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    
    acc_vit = correct_vit / total
    acc_siglip = correct_siglip / total
    
    print(f"ViT-Large 384:     {acc_vit:.2%} ({correct_vit}/{total})")
    print(f"SigLIP2:           {acc_siglip:.2%} ({correct_siglip}/{total})")
    
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
    
    print(f"\n=== Confusion matrix (SigLIP2) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["siglip_pred"]] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    siglip_errors = [r for r in results if not r["siglip_correct"]]
    print(f"\n=== Ошибки SigLIP2 ({len(siglip_errors)} файлов) ===")
    for r in siglip_errors[:20]:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  siglip={r['siglip_pred']:3d}({r['siglip_conf']:.2f})  vit={r['vit_pred']:3d}({r['vit_conf']:.2f})")
    
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
    
    report_path = C.MODELS_DIR / "siglip_trained_vs_vit_report.json"
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
