"""Обучение SigLIP2 для классификации ориентации документов (4 класса: 0°, 90°, 180°, 270°).

Использует vision encoder из google/siglip2-so400m-patch14-384.
Загружает предобученные веса и дообучает на датасете ориентации.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoModel, AutoProcessor

import config as C
import text_crop


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


PAD_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}


def letterbox(img: Image.Image, size: int, fill) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def add_black_borders(img: Image.Image, max_frac: float) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    for e in edges[:random.randint(1, 4)]:
        if e in ("top", "bottom"):
            bw = random.randint(1, max(1, int(h * max_frac)))
            box = [0, 0, w, bw] if e == "top" else [0, h - bw, w, h]
        else:
            bw = random.randint(1, max(1, int(w * max_frac)))
            box = [0, 0, bw, h] if e == "left" else [w - bw, 0, w, h]
        draw.rectangle(box, fill=(0, 0, 0))
    return img


class OrientationDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, mode: str, processor):
        self.samples = base.samples
        self.classes = base.classes
        self.targets = [c for _, c in base.samples]
        
        try:
            self.idx_to_angle = {i: int(name) for i, name in enumerate(base.classes)}
            self.angle_to_idx = {int(name): i for i, name in enumerate(base.classes)}
        except ValueError:
            self.idx_to_angle = {}
            self.angle_to_idx = {}
            for i, name in enumerate(base.classes):
                try:
                    angle = int(name)
                    self.idx_to_angle[i] = angle
                    self.angle_to_idx[angle] = i
                except ValueError:
                    self.idx_to_angle[i] = -1
                    self.angle_to_idx[-1] = i
        
        self.mode = mode
        self.processor = processor
        self.val_full = (mode == "val" and C.ROTATION_AUG and C.ROTATION_AUG_VAL_FULL)
        self.n = len(self.samples)

    def __len__(self):
        return self.n * 4 if self.val_full else self.n

    def __getitem__(self, idx):
        if self.val_full:
            sample_idx, k = idx // 4, idx % 4
        else:
            sample_idx = idx
            k = random.randint(0, 3) if (self.mode == "train" and C.ROTATION_AUG) else 0

        path, class_idx = self.samples[sample_idx]
        orig_angle = self.idx_to_angle[class_idx]
        img = Image.open(path).convert("RGB")

        a = 90 * k
        if a:
            img = img.rotate((-a) % 360, expand=True)
        
        new_angle = (orig_angle + a) % 360
        target = self.angle_to_idx.get(new_angle, 0)

        if self.mode == "train" and C.VERTICAL_FLIP_AUG and random.random() < C.VERTICAL_FLIP_PROB:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            angle_0_idx = self.angle_to_idx.get(0, 0)
            angle_180_idx = self.angle_to_idx.get(180, 1)
            angle_90_idx = self.angle_to_idx.get(90, 3)
            angle_270_idx = self.angle_to_idx.get(270, 2)
            
            if target == angle_0_idx:
                target = angle_180_idx
            elif target == angle_180_idx:
                target = angle_0_idx
            elif target == angle_90_idx:
                target = angle_270_idx
            elif target == angle_270_idx:
                target = angle_90_idx

        if self.mode == "train":
            if C.SKEW_AUG and random.random() < C.SKEW_PROB:
                ang = random.uniform(-C.SKEW_MAX_DEG, C.SKEW_MAX_DEG)
                fill = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
                img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=fill)
            if C.BORDER_AUG and random.random() < C.BORDER_PROB:
                img = add_black_borders(img, C.BORDER_MAX_FRAC)
            if C.HFLIP_AUG and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # SigLIP2 processor обрабатывает изображение
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        num_classes = len(self.classes)
        if not (0 <= target < num_classes):
            target = max(0, min(target, num_classes - 1))

        return pixel_values, target


class SigLIPClassifier(nn.Module):
    def __init__(self, model_path: str, num_classes: int):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(model_path)
        # Замораживаем vision encoder
        for param in self.siglip.vision_model.parameters():
            param.requires_grad = False
        
        # Получаем размерность признаков из vision encoder
        hidden_size = self.siglip.vision_model.config.hidden_size  # 1152 для so400m
        
        # Классификатор поверх [CLS] токена
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        # Получаем визуальные признаки
        outputs = self.siglip.vision_model(pixel_values=pixel_values)
        # Используем [CLS] токен (последний hidden state)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        logits = self.classifier(cls_output)
        return logits


def build_loaders(device):
    base_train = datasets.ImageFolder(str(C.TRAIN_DIR))
    base_val = datasets.ImageFolder(str(C.VAL_DIR))

    if len(base_train) == 0 or len(base_val) == 0:
        print("[ERROR] Пустой обучающий или валидационный набор.", file=sys.stderr)
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(r"D:\MODELS\Transformers\siglip2-so400m-patch14-384")

    class_names = base_train.classes
    print(f"Классы ImageFolder: {class_names} (samples: {len(base_train)} train / {len(base_val)} val)")

    train_ds = OrientationDataset(base_train, "train", processor)
    val_ds = OrientationDataset(base_val, "val", processor)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=pin)
    return train_loader, val_loader, base_train.class_to_idx


def evaluate(model, loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            for t, p in zip(y.view(-1), pred.view(-1)):
                confmat[t.long(), p.long()] += 1

    val_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    per_class_total = confmat.sum(dim=1)
    per_class_correct = confmat.diag()
    per_class_recall = []
    for c in range(num_classes):
        n = int(per_class_total[c].item())
        per_class_recall.append(per_class_correct[c].item() / n if n > 0 else float("nan"))
    
    valid = [r for r in per_class_recall if r == r]
    balanced_acc = sum(valid) / len(valid) if valid else 0.0
    
    return val_loss, accuracy, balanced_acc, per_class_recall, confmat.tolist()


def main():
    device = get_device()
    print(f"Устройство: {device}")
    
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_loaders(device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model_path = r"D:\MODELS\Transformers\siglip2-so400m-patch14-384"
    model = SigLIPClassifier(model_path, C.OUTPUT_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6)

    class_label_names = [idx_to_class[i] for i in range(C.OUTPUT_CLASSES)]
    best_score = -1.0
    best_path = C.MODELS_DIR / "orientation_siglip2_best.pth"
    final_path = C.MODELS_DIR / "orientation_siglip2_final.pth"
    
    history = {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "epochs": []}

    no_improve = 0
    for epoch in range(1, 51):
        model.train()
        t0 = time.time()
        running = 0.0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val_loss, val_acc, bal_acc, per_class_recall, confmat = evaluate(
            model, val_loader, device, criterion, C.OUTPUT_CLASSES)
        
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        print(f"Epoch {epoch:2d}/50  {dt:5.1f}s  lr={cur_lr:.2e}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  bal_acc={bal_acc:.4f}")
        
        recall_desc = "  ".join(
            f"{class_label_names[i]}:{per_class_recall[i]:.3f}" for i in range(C.OUTPUT_CLASSES))
        print(f"           per-class recall: {recall_desc}")
        
        history["epochs"].append({
            "epoch": epoch, "lr": cur_lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "balanced_acc": bal_acc,
            "per_class_recall": per_class_recall, "confusion_matrix": confmat,
        })

        if bal_acc > best_score:
            best_score = bal_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": C.OUTPUT_CLASSES,
                "model_name": "siglip2-so400m-patch14-384"
            }, best_path)
            print(f"  -> лучший результат сохранён: {best_path.name} "
                  f"(balanced={best_score:.4f}, val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                print(f"Early stopping: нет улучшения {no_improve} эпох.")
                break

    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": C.OUTPUT_CLASSES,
        "model_name": "siglip2-so400m-patch14-384"
    }, final_path)
    
    history["best_metric"] = "balanced"
    history["best_score"] = best_score
    
    with open(C.MODELS_DIR / "siglip2_train_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(f"\nОбучение завершено. Лучшая метрика (balanced)={best_score:.4f}")
    print(f"  лучшая модель: {best_path}")
    print(f"  финальная:     {final_path}")


if __name__ == "__main__":
    main()
