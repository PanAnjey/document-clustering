"""Гибридный метод: ViT-Large с self-consistency check.

Pass 1: оригинальное изображение -> pred1, conf1
Pass 2: повёрнутое на 90° -> pred2, conf2 -> истинный угол = (pred2 - 90) % 360

Если pred1 == (pred2 - 90) % 360 -> согласие -> высокая уверенность
Если не совпадают -> флаг на ручную проверку или выбор по максимальной confidence
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


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    print(f"Загрузка модели: {model_path.name}")
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


def predict(model, transform, device, idx_to_angle, img: Image.Image):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(img_tensor)
        prob = torch.softmax(out.float(), dim=1)
        conf, pred = prob.max(dim=1)
    return idx_to_angle[int(pred[0].item())], float(conf[0].item())


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
    print("=== ViT-Large с self-consistency check ===\n")

    model, transform, device, idx_to_angle = load_model()
    ground_truth = build_ground_truth()
    print(f"Ground truth: {len(ground_truth)} файлов\n")

    files = [f for f in PDF_IMAGES_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
    print(f"Файлов: {len(files)}\n")

    results = []
    t0 = time.time()

    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        if fname not in ground_truth:
            continue

        true_angle = ground_truth[fname]

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            continue

        # Pass 1: оригинал
        pred1, conf1 = predict(model, transform, device, idx_to_angle, img)

        # Pass 2: повёрнутое на 90°
        img_rot = img.rotate(90, expand=True)
        pred2, conf2 = predict(model, transform, device, idx_to_angle, img_rot)
        pred2_corrected = (pred2 - 90) % 360  # истинный угол из pass 2

        # Анализ согласованности
        consistent = (pred1 == pred2_corrected)

        # Финальное решение
        if consistent:
            # Согласие — выбираем по максимальной confidence
            if conf1 >= conf2:
                final_pred = pred1
                final_conf = conf1
            else:
                final_pred = pred2_corrected
                final_conf = conf2
        else:
            # Несогласие — выбираем по максимальной confidence
            if conf1 >= conf2:
                final_pred = pred1
                final_conf = conf1
            else:
                final_pred = pred2_corrected
                final_conf = conf2

        results.append({
            "file": fname,
            "true_angle": true_angle,
            "pred1": pred1,
            "conf1": round(conf1, 3),
            "pred2_rot": pred2,
            "pred2_corrected": pred2_corrected,
            "conf2": round(conf2, 3),
            "consistent": consistent,
            "final_pred": final_pred,
            "final_conf": round(final_conf, 3),
            "correct": final_pred == true_angle
        })

        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"Обработано {i}/{len(files)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    total = len(results)

    # Базовая accuracy (только pass 1)
    correct_pass1 = sum(1 for r in results if r["pred1"] == r["true_angle"])
    # Гибридная accuracy
    correct_hybrid = sum(1 for r in results if r["correct"])
    # Согласие
    n_consistent = sum(1 for r in results if r["consistent"])
    correct_consistent = sum(1 for r in results if r["consistent"] and r["correct"])
    correct_inconsistent = sum(1 for r in results if not r["consistent"] and r["correct"])

    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"Pass 1 (оригинал):       {correct_pass1}/{total} = {correct_pass1/total:.2%}")
    print(f"Гибрид (pass1+pass2):    {correct_hybrid}/{total} = {correct_hybrid/total:.2%}")
    print(f"\nСогласие pass1==pass2:   {n_consistent}/{total} ({n_consistent/total:.2%})")
    print(f"  правильных:            {correct_consistent}/{n_consistent} = {correct_consistent/max(n_consistent,1):.2%}")
    print(f"Несогласие:              {total-n_consistent}/{total} ({(total-n_consistent)/total:.2%})")
    print(f"  правильных:            {correct_inconsistent}/{total-n_consistent} = {correct_inconsistent/max(total-n_consistent,1):.2%}")

    # Confusion matrix для гибрида
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["final_pred"]] += 1

    print(f"\nConfusion matrix (гибрид):")
    header = "         pred: " + "  ".join(f"{a:3d}" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")

    # Ошибки
    errors = [r for r in results if not r["correct"]]
    print(f"\nОшибок: {len(errors)}")
    for r in errors:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  pred1={r['pred1']:3d}({r['conf1']:.2f})  pred2={r['pred2_corrected']:3d}({r['conf2']:.2f})  final={r['final_pred']:3d}  {'CONS' if r['consistent'] else 'DIFF'}")

    # Сохранение
    report_path = C.MODELS_DIR / "hybrid_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    main()
