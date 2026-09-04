"""Ансамбль ViT-Large + PaddleOCR с голосованием.

Определяет ориентацию документов через:
1. ViT-Large 384 - обученная модель
2. PaddleOCR - модель PP-LCNet_x1_0_doc_ori

Голосование: если обе модели согласны - используем их предсказание.
Если не согласны - используем предсказание с большей уверенностью.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VIT_DIR = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN")
sys.path.insert(0, str(VIT_DIR))
import config as C

PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

ANGLE_ORDER = [0, 90, 180, 270]


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


def load_paddle_model():
    """Загружает PaddleOCR модель определения ориентации."""
    from paddlex import create_model
    
    print("Загрузка PaddleOCR PP-LCNet_x1_0_doc_ori...")
    model = create_model(model_name='PP-LCNet_x1_0_doc_ori')
    return model


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


def predict_paddle(model, img_path: Path) -> tuple[int, float]:
    """Предсказание PaddleOCR."""
    try:
        result = model.predict(str(img_path))
        
        for r in result:
            label = r['label_names'][0]
            score = float(r['scores'][0])
            
            # PaddleOCR возвращает '0', '90', '180', '270'
            angle = int(label)
            return angle, score
        
        return 0, 0.0
    except Exception as e:
        print(f"  [WARN] PaddleOCR error for {img_path.name}: {e}")
        return 0, 0.0


def ensemble_vote(vit_pred, vit_conf, paddle_pred, paddle_conf) -> tuple[int, float, str]:
    """Голосование ансамбля.
    
    Возвращает: (предсказание, уверенность, метод)
    """
    if vit_pred == paddle_pred:
        # Согласие - используем среднюю уверенность
        avg_conf = (vit_conf + paddle_conf) / 2
        return vit_pred, avg_conf, "agree"
    else:
        # Несогласие - используем предсказание с большей уверенностью
        if vit_conf >= paddle_conf:
            return vit_pred, vit_conf, "vit"
        else:
            return paddle_pred, paddle_conf, "paddle"


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
    print("=== Ансамбль ViT-Large + PaddleOCR ===\n")
    
    # Загрузка моделей
    vit_model, vit_transform, vit_device, vit_idx_to_angle = load_vit_model()
    paddle_model = load_paddle_model()
    
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
        paddle_pred, paddle_conf = predict_paddle(paddle_model, fpath)
        
        # Голосование ансамбля
        ensemble_pred, ensemble_conf, vote_method = ensemble_vote(
            vit_pred, vit_conf, paddle_pred, paddle_conf
        )
        
        results.append({
            "file": fname,
            "true_angle": true_angle,
            "vit_pred": vit_pred,
            "vit_conf": round(vit_conf, 3),
            "paddle_pred": paddle_pred,
            "paddle_conf": round(paddle_conf, 3),
            "ensemble_pred": ensemble_pred,
            "ensemble_conf": round(ensemble_conf, 3),
            "vote_method": vote_method,
            "vit_correct": vit_pred == true_angle,
            "paddle_correct": paddle_pred == true_angle,
            "ensemble_correct": ensemble_pred == true_angle
        })
        
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"Обработано {i}/{len(files)} ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    total = len(results)
    
    # Подсчёт метрик
    correct_vit = sum(1 for r in results if r["vit_correct"])
    correct_paddle = sum(1 for r in results if r["paddle_correct"])
    correct_ensemble = sum(1 for r in results if r["ensemble_correct"])
    
    # Статистика голосования
    n_agree = sum(1 for r in results if r["vote_method"] == "agree")
    n_vit_wins = sum(1 for r in results if r["vote_method"] == "vit")
    n_paddle_wins = sum(1 for r in results if r["vote_method"] == "paddle")
    
    # Отчёт
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    
    acc_vit = correct_vit / total
    acc_paddle = correct_paddle / total
    acc_ensemble = correct_ensemble / total
    
    print(f"ViT-Large 384:       {acc_vit:.2%} ({correct_vit}/{total})")
    print(f"PaddleOCR:           {acc_paddle:.2%} ({correct_paddle}/{total})")
    print(f"Ensemble (vote):     {acc_ensemble:.2%} ({correct_ensemble}/{total})")
    
    print(f"\n=== Статистика голосования ===")
    print(f"Согласие (agree):    {n_agree}/{total} ({n_agree/total:.2%})")
    print(f"ViT побеждает:       {n_vit_wins}/{total} ({n_vit_wins/total:.2%})")
    print(f"PaddleOCR побеждает: {n_paddle_wins}/{total} ({n_paddle_wins/total:.2%})")
    
    # Per-class метрики для ансамбля
    print(f"\n=== Per-class метрики (Ensemble) ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in results if r["true_angle"] == angle]
        if angle_results:
            correct = sum(1 for r in angle_results if r["ensemble_correct"])
            total_angle = len(angle_results)
            print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")
    
    # Confusion matrix для ансамбля
    print(f"\n=== Confusion matrix (Ensemble) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["ensemble_pred"]] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    # Ошибки ансамбля
    ensemble_errors = [r for r in results if not r["ensemble_correct"]]
    print(f"\n=== Ошибки ансамбля ({len(ensemble_errors)} файлов) ===")
    for r in ensemble_errors[:20]:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  vit={r['vit_pred']:3d}({r['vit_conf']:.2f})  paddle={r['paddle_pred']:3d}({r['paddle_conf']:.2f})  ens={r['ensemble_pred']:3d}  [{r['vote_method']}]")
    
    # Сравнение: где ансамбль лучше отдельных моделей
    ensemble_better_vit = [r for r in results if r["ensemble_correct"] and not r["vit_correct"]]
    ensemble_better_paddle = [r for r in results if r["ensemble_correct"] and not r["paddle_correct"]]
    
    print(f"\n=== Улучшения ансамбля ===")
    print(f"Ensemble лучше ViT:      {len(ensemble_better_vit)} файлов")
    print(f"Ensemble лучше PaddleOCR: {len(ensemble_better_paddle)} файлов")
    
    # Сохранение результатов
    report_path = C.MODELS_DIR / "ensemble_vit_paddle_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "vit_accuracy": round(acc_vit, 4),
            "paddle_accuracy": round(acc_paddle, 4),
            "ensemble_accuracy": round(acc_ensemble, 4),
            "n_agree": n_agree,
            "n_vit_wins": n_vit_wins,
            "n_paddle_wins": n_paddle_wins,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
