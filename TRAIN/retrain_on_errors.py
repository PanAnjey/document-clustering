"""Дообучение модели на ошибках + дополнительных файлах из distrib.

Собирает:
1. Ошибочные файлы из PDF_Images (те, что модель предсказывает неправильно)
2. Дополнительные файлы из distrib/0 и distrib/180 (классы с путаницей)

Дообучает текущую best модель с маленьким LR.
"""
import json
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

import config as C
import text_crop

Image.MAX_IMAGE_PIXELS = None

RETRAIN_DIR = Path(r"D:\FileOrganizer\TRAIN\retrain_data")
RETRAIN_EPOCHS = 20
RETRAIN_LR = 1e-5
RETRAIN_BATCH_SIZE = 8


def _make_writable(p: Path):
    """Снять ReadOnly атрибут на Windows."""
    try:
        import os
        import stat
        cur = os.stat(p).st_mode
        os.chmod(p, cur | stat.S_IWRITE)
    except OSError:
        pass


def _on_rmtree_error(func, fpath, exc_info):
    """callback для shutil.rmtree: снять ReadOnly и повторить."""
    _make_writable(Path(fpath))
    try:
        func(fpath)
    except OSError as e:
        print(f"  [warn] не удалось удалить {fpath}: {e}")


def clear_dir(path: Path):
    """Полностью удалить директорию и пересоздать пустую."""
    if path.exists():
        shutil.rmtree(path, onerror=_on_rmtree_error)
    path.mkdir(parents=True, exist_ok=True)


def get_input_size() -> int:
    if "384" in C.MODEL_NAME:
        return 384
    elif "224" in C.MODEL_NAME:
        return 224
    return 224


def _letterbox_white(img: Image.Image, size: int) -> Image.Image:
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
    input_size = get_input_size()
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        norm
    ])


def collect_error_files():
    """Собирает ошибочные файлы из PDF_Images."""
    results_file = C.MODELS_DIR / "inference_results.json"
    if not results_file.exists():
        print("[WARN] inference_results.json не найден, запускаю инференс...")
        import subprocess
        subprocess.run([sys.executable, "3_test_inference.py", "--compare-only"], check=True)
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    gt = {}
    for angle in [0, 90, 180, 270]:
        d = C.DISTRIB_DIR / str(angle)
        if d.exists():
            for file in d.iterdir():
                if file.is_file():
                    gt[file.name] = angle
    
    errors = []
    for res in data.get("results", []):
        fname = res.get("file")
        pred = res.get("predicted_orientation")
        if fname and fname in gt:
            true_angle = gt[fname]
            if pred != true_angle:
                errors.append((C.INFERENCE_INPUT_DIR / fname, true_angle))
    
    return errors


def collect_additional_files(n_per_class=200):
    """Собирает дополнительные файлы из distrib/0 и distrib/180."""
    files = []
    for angle in [0, 180]:
        d = C.DISTRIB_DIR / str(angle)
        if d.exists():
            all_files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
            random.shuffle(all_files)
            selected = all_files[:n_per_class]
            for f in selected:
                files.append((f, angle))
    return files


def prepare_retrain_dir():
    """Подготавливает директорию для retrain."""
    clear_dir(RETRAIN_DIR)
    
    for angle in [0, 90, 180, 270]:
        (RETRAIN_DIR / str(angle)).mkdir(parents=True, exist_ok=True)
    
    errors = collect_error_files()
    print(f"Ошибочных файлов: {len(errors)}")
    
    additional = collect_additional_files(n_per_class=200)
    print(f"Дополнительных файлов: {len(additional)}")
    
    all_files = errors + additional
    
    for src_path, angle in all_files:
        dst = RETRAIN_DIR / str(angle) / src_path.name
        if not dst.exists():
            shutil.copy2(src_path, dst)
    
    for angle in [0, 90, 180, 270]:
        d = RETRAIN_DIR / str(angle)
        n = len(list(d.iterdir()))
        print(f"  {angle}°: {n} файлов")
    
    return len(all_files)


class SimpleDataset(Dataset):
    def __init__(self, files_with_labels, transform):
        self.samples = files_with_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, angle = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        angle_to_idx = {0: 0, 180: 1, 270: 2, 90: 3}
        target = angle_to_idx[angle]
        return img, target


def retrain():
    """Дообучает модель на собранных файлах."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    model_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    if not model_path.exists():
        print(f"[ERROR] Модель не найдена: {model_path}")
        return
    
    print(f"Загрузка модели: {model_path.name}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
    
    tfm = build_transform()
    
    files_with_labels = []
    for angle in [0, 90, 180, 270]:
        d = RETRAIN_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS:
                    files_with_labels.append((f, angle))
    
    random.shuffle(files_with_labels)
    n_train = int(len(files_with_labels) * 0.8)
    train_files = files_with_labels[:n_train]
    val_files = files_with_labels[n_train:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_ds = SimpleDataset(train_files, tfm)
    val_ds = SimpleDataset(val_files, tfm)
    
    train_loader = DataLoader(train_ds, batch_size=RETRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=RETRAIN_BATCH_SIZE, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch:2d}/{RETRAIN_EPOCHS}  train_loss={train_loss/total:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, C.MODELS_DIR / "orientation_vit_large_384_retrained.pth")
            print(f"  -> лучшая модель сохранена (val_acc={val_acc:.4f})")
    
    print(f"\nДообучение завершено. Лучшая val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    random.seed(42)
    print("=== Подготовка файлов для дообучения ===")
    n = prepare_retrain_dir()
    print(f"\nВсего файлов: {n}")
    
    print("\n=== Дообучение модели ===")
    retrain()
