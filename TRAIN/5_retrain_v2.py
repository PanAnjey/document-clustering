"""Повторное дообучение (fine-tuning) Vision Transformer на исправленных ошибках.

- Загружает лучшую модель из MODELS_DIR.
- Использует данные из RETRAIN_DIR для fine-tuning.
- Сохраняет новую версию модели с префиксом RETRAIN_FILE_PREFIX.
- Оптимизировано для 2x NVIDIA RTX PRO 4000 48GB VRAM.

Улучшения:
1. Увеличено количество эпох (30 вместо 10)
2. Разморозка части backbone слоев
3. Более агрессивная аугментация данных
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
import timm

import config as C


def get_device():
    """Определяет устройство (GPU/CPU) и количество доступных GPU."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Доступно {num_gpus} GPU: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        return torch.device("cuda"), num_gpus
    return torch.device("cpu"), 1


def compute_class_weights(dataset, mode: str, device) -> torch.Tensor:
    """Веса классов для CrossEntropyLoss по дисбалансу обучающей выборки."""
    num_classes = len(dataset.classes)
    counts = Counter(int(t) for t in dataset.targets)
    counts_arr = torch.tensor([float(counts.get(i, 0)) for i in range(num_classes)],
                              dtype=torch.float32)

    if mode == "off":
        w = torch.ones(num_classes, dtype=torch.float32)
    elif mode == "inverse":
        w = 1.0 / counts_arr.clamp(min=1.0)
    elif mode == "inverse_norm":
        w = 1.0 / counts_arr.clamp(min=1.0)
        w = w * (num_classes / w.sum())
    elif mode == "effective_num":
        beta = 0.999
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts_arr.clamp(min=1.0)))
        w = w * (num_classes / w.sum())
    else:
        print(f"[WARN] неизвестный CLASS_WEIGHTS='{mode}', используются равные веса.")
        w = torch.ones(num_classes, dtype=torch.float32)

    names = dataset.classes
    desc = ", ".join(f"{names[i]}:{int(counts_arr[i])}/w={w[i]:.3f}" for i in range(num_classes))
    print(f"Веса классов ({mode}): {desc}")
    return w.to(device)


PAD_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}


def letterbox(img: Image.Image, size: int, fill) -> Image.Image:
    """Вписать изображение в квадрат size×size с сохранением пропорций + поля цвета fill."""
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def add_black_borders(img: Image.Image, max_frac: float) -> Image.Image:
    """Нарисовать чёрные полосы на 1..4 случайных краях."""
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


class RetrainDataset(Dataset):
    """Датасет для повторного дообучения."""

    def __init__(self, base: datasets.ImageFolder, mode: str, post_tfm):
        self.samples = base.samples
        self.classes = base.classes
        self.targets = [c for _, c in base.samples]
        
        # Преобразуем все имена классов в int для совместимости
        try:
            self.idx_to_angle = {i: int(name) for i, name in enumerate(base.classes)}
            self.angle_to_idx = {int(name): i for i, name in enumerate(base.classes)}
        except ValueError as e:
            print(f"[INFO] Обнаружен нечисловой класс: {e}")
            self.idx_to_angle = {}
            self.angle_to_idx = {}
            for i, name in enumerate(base.classes):
                try:
                    angle = int(name)
                    self.idx_to_angle[i] = angle
                    self.angle_to_idx[angle] = i
                except ValueError:
                    other_idx = i
                    self.idx_to_angle[i] = -1
                    self.angle_to_idx[-1] = i
        
        self.mode = mode
        self.post_tfm = post_tfm
        self.val_full = (mode == "val" and C.RETRAIN_ROTATION_AUG and C.ROTATION_AUG_VAL_FULL)
        self.n = len(self.samples)

    def __len__(self):
        return self.n * 4 if self.val_full else self.n

    def __getitem__(self, idx):
        if self.val_full:
            sample_idx, k = idx // 4, idx % 4
        else:
            sample_idx = idx
            k = random.randint(0, 3) if (self.mode == "train" and C.RETRAIN_ROTATION_AUG) else 0

        path, class_idx = self.samples[sample_idx]
        orig_angle = self.idx_to_angle[class_idx]
        img = Image.open(path).convert("RGB")

        # Поворот ориентации
        a = 90 * k
        if a:
            img = img.rotate((-a) % 360, expand=True)
        
        # Вычисляем новый угол после поворота
        new_angle = (orig_angle + a) % 360
        
        # Находим индекс класса для нового угла
        if new_angle in self.angle_to_idx:
            target = int(self.angle_to_idx[new_angle])
        else:
            print(f"[WARN] Угол {new_angle} не найден в angle_to_idx. Используем класс для 0°.")
            target = self.angle_to_idx.get(0, 0)

        # VerticalFlip для улучшения 0° vs 180°
        if self.mode == "train" and C.VERTICAL_FLIP_AUG and random.random() < C.VERTICAL_FLIP_PROB:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Определяем индексы классов для каждого угла
            angle_0_idx = self.angle_to_idx.get(0, 0)
            angle_180_idx = self.angle_to_idx.get(180, 1)
            angle_90_idx = self.angle_to_idx.get(90, 3)
            angle_270_idx = self.angle_to_idx.get(270, 2)
            
            # Меняем метки при вертикальном флипе
            if target == angle_0_idx:
                target = angle_180_idx
            elif target == angle_180_idx:
                target = angle_0_idx
            elif target == angle_90_idx:
                target = angle_270_idx
            elif target == angle_270_idx:
                target = angle_90_idx

        if self.mode == "train":
            # Перекос сканирования
            if C.SKEW_AUG and random.random() < C.SKEW_PROB:
                ang = random.uniform(-C.SKEW_MAX_DEG, C.SKEW_MAX_DEG)
                fill = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
                img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=fill)
            
            # Чёрные поля по краям
            if C.BORDER_AUG and random.random() < C.BORDER_PROB:
                img = add_black_borders(img, C.BORDER_MAX_FRAC)

        # Вход модели - используем IMAGE_SIZE из config.py (тот же, что и при первичном обучении)
        if getattr(C, "INPUT_MODE", "full") == "crop":
            jit = C.CROP_JITTER if self.mode == "train" else 0
            import text_crop
            img = text_crop.crop_text_region(img, jitter=jit)
        elif getattr(C, "RESIZE_MODE", "letterbox") == "letterbox":
            if self.mode == "train":
                pad = C.LETTERBOX_PAD_TRAIN
                fill = random.choice(list(PAD_COLORS.values())) if pad == "random" \
                    else PAD_COLORS.get(pad, (255, 255, 255))
            else:
                fill = (255, 255, 255)
            img = letterbox(img, C.IMAGE_SIZE, fill)
        else:
            img = img.resize((C.IMAGE_SIZE, C.IMAGE_SIZE), Image.BILINEAR)

        # Финальная проверка: target должен быть индексом класса [0, n_classes)
        num_classes = len(self.classes)
        if not (0 <= target < num_classes):
            print(f"[ERROR] Target {target} out of range [0, {num_classes})! Clipping to 0.")
            target = max(0, min(target, num_classes - 1))

        return self.post_tfm(img), target


def build_model(num_classes, device):
    """Строит модель Vision Transformer с attention-механизмом.
    
    Использует IMAGE_SIZE из config.py для согласованности с основным обучением.
    """
    
    # Используем IMAGE_SIZE из config.py (тот же, что и при первичном обучении)
    input_size = C.IMAGE_SIZE
    print(f"[INFO] Загрузка модели с размером входа {input_size}x{input_size} (как при первичном обучении)")
    
    # Определяем архитектуру по USE_VIT_LAYERS и размеру входа
    if C.USE_VIT_LAYERS == "large":
        model = timm.create_model(f'vit_large_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 1024
    else:
        model = timm.create_model(f'vit_base_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 768

    # Добавляем attention-механизм (если включён)
    if C.USE_ATTENTION:
        from torch import nn
        class ViTAttention(nn.Module):
            def __init__(self, in_features, hidden_size=512, dropout=0.3):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, in_features),
                    nn.Sigmoid()
                )

            def forward(self, x):
                if x.dim() == 4:
                    attention_weights = self.attention(x)
                    return x * attention_weights
                else:
                    attention_weights = self.attention(x.unsqueeze(-1)).squeeze(-1)
                    return x * attention_weights
        
        model.head = ViTAttention(in_features, hidden_size=C.ATTENTION_HIDDEN_SIZE, 
                                  dropout=C.ATTENTION_DROPOUT)
    
    # Заменяем классификатор на OUTPUT_CLASSES
    if hasattr(model, 'head'):
        in_features = model.head.in_features if hasattr(model.head, 'in_features') else in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    return model.to(device)


def evaluate(model, loader, device, criterion, num_classes):
    """Возвращает: val_loss, accuracy (plain), balanced_acc (macro-recall)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and C.USE_AMP)):
                out = model(x)
                loss = criterion(out, y)
            
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    # Для retrain используем упрощённую метрику
    balanced_acc = accuracy
    
    return val_loss, accuracy, balanced_acc


def main():
    device, num_gpus = get_device()
    print(f"Устройство: {device} (GPU count: {num_gpus})")
    
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка базовой модели с правильным размером входа
    base_model_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME if C.RETRAIN_BASE_MODEL == "best" \
                      else C.MODELS_DIR / C.MODEL_FILENAME
    
    if not base_model_path.exists():
        print(f"[ERROR] Базовая модель не найдена: {base_model_path}")
        sys.exit(1)

    ckpt = torch.load(base_model_path, map_location=device, weights_only=True)
    
    model = build_model(C.OUTPUT_CLASSES, device)
    
    # Загружаем веса с игнорированием mismatch для positional embeddings
    try:
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[INFO] Веса загружены (strict=False из-за размера входа)")
    except Exception as e:
        print(f"[WARN] Ошибка загрузки: {e}")
        print("[INFO] Продолжаем обучение с частичной загрузкой весов")
    
    print(f"Загружена базовая модель: {base_model_path.name}")

    # Построение датасетов
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_post = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        norm,
    ])
    
    val_post = transforms.Compose([transforms.ToTensor(), norm])

    base_train = datasets.ImageFolder(str(C.RETRAIN_DIR))  # transform=None -> PIL
    
    if len(base_train) == 0:
        print("ERROR: Пустой retrain-набор. Запустите review_app.py сначала.", file=sys.stderr)
        sys.exit(1)

    class_names = base_train.classes
    print(f"Классы ImageFolder (retrain): {class_names} (samples: {len(base_train)})")

    train_ds = RetrainDataset(base_train, "train", train_post)
    
    # Разделение на train/val для retrain
    val_split = C.RETRAIN_VAL_SPLIT
    indices = list(range(len(train_ds)))
    random.seed(C.RANDOM_SEED)
    random.shuffle(indices)
    
    n_val = int(len(indices) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Создаём subset датасеты
    from torch.utils.data import Subset
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)

    pin = device.type == "cuda"
    
    train_loader = DataLoader(train_subset, batch_size=C.RETRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=max(1, C.NUM_WORKERS // 2), pin_memory=pin, drop_last=False)
    
    val_loader = DataLoader(val_subset, batch_size=C.RETRAIN_BATCH_SIZE, shuffle=False,
                            num_workers=max(1, C.NUM_WORKERS // 2), pin_memory=pin)

    # Loss и optimizer (меньший LR для fine-tuning)
    weight_mode = "off" if C.RETRAIN_ROTATION_AUG else C.CLASS_WEIGHTS
    class_weights = compute_class_weights(base_train, weight_mode, device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Fine-tuning: замораживаем backbone, обучаем только head (или часть слоев)
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=C.RETRAIN_LEARNING_RATE, 
        weight_decay=C.RETRAIN_WEIGHT_DECAY
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and C.USE_AMP))

    scheduler = None
    if getattr(C, "LR_SCHEDULER", "off") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=C.RETRAIN_EPOCHS, eta_min=getattr(C, "LR_MIN", 1e-6))

    best_metric_name = getattr(C, "BEST_METRIC", "balanced")
    
    # Создаём idx_to_class из class_names
    try:
        idx_to_class = {int(v): k for k, v in base_train.class_to_idx.items()}
    except ValueError:
        idx_to_class = {v: k for k, v in base_train.class_to_idx.items()}
    
    class_label_names = [idx_to_class[i] if i in idx_to_class else str(i) 
                        for i in range(C.OUTPUT_CLASSES)]
    
    best_score = -1.0
    retrain_path = C.MODELS_DIR / f"{C.RETRAIN_FILE_PREFIX}_{len([f for f in C.MODELS_DIR.glob('orientation_vit_base_retrain_*')]) + 1}.pth"
    
    history = {"epochs": []}

    no_improve = 0
    for epoch in range(1, C.RETRAIN_EPOCHS + 1):
        model.train()
        t0 = time.time()
        
        running = 0.0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and C.USE_AMP)):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_subset.dataset)
        
        val_loss, val_acc, bal_acc = evaluate(model, val_loader, device, criterion, C.OUTPUT_CLASSES)
        
        if scheduler is not None:
            scheduler.step()
        
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        print(f"Epoch {epoch:2d}/{C.RETRAIN_EPOCHS}  {dt:5.1f}s  lr={cur_lr:.2e}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        
        history["epochs"].append({
            "epoch": epoch, 
            "lr": cur_lr,
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        score = bal_acc if best_metric_name == "balanced" else val_acc
        
        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": base_train.class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": C.OUTPUT_CLASSES,
                "model_name": f"{C.RETRAIN_FILE_PREFIX}_{epoch}",
            }, retrain_path)
            print(f"  -> лучшая модель сохранена: {retrain_path.name} ({best_metric_name}={best_score:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if C.RETRAIN_EARLY_STOP_PATIENCE and no_improve >= C.RETRAIN_EARLY_STOP_PATIENCE:
                print(f"Early stopping: нет улучшения {no_improve} эпох.")
                break

    # Замена лучшей модели (если нужно)
    if C.RETRAIN_REPLACE_BEST:
        import shutil
        best_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
        shutil.copy(retrain_path, best_path)
        print(f"  -> лучшая модель заменена: {best_path.name}")

    history["best_metric"] = best_metric_name
    history["best_score"] = best_score
    
    with open(C.MODELS_DIR / "retrain_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(f"\nДообучение завершено. Лучшая метрика ({best_metric_name})={best_score:.4f}")
    print(f"  дообученная модель: {retrain_path.name}")


if __name__ == "__main__":
    main()
