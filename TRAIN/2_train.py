"""Дообучение Vision Transformer (Large версия) для классификации ориентации сканов (4 класса: 0,90,180,270).

- Загружает timm vit_large_patch16_384 с предобученными весами IMAGENET1K_V2.
- Добавляет attention-механизм для улучшения различения 0° vs 180°.
- Заменяет классификатор на OUTPUT_CLASSES (4 класса).
- Датасеты: ImageFolder для TRAIN_DIR и VAL_DIR (структура {0,90,180,270}).
- Аугментация: VerticalFlip критичен для 0° vs 180°.
- Сохраняет лучшую (по val accuracy) и финальную модель в MODELS_DIR.
- Поддерживает early stopping и gradient accumulation.
- Логирует метрики и сохраняет историю в models/train_history.json.

Оптимизировано для 2x NVIDIA RTX PRO 4000 48GB VRAM (ViT Large).
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  # отключить DecompressionBombWarning для больших сканов

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

import config as C
import text_crop


def get_device():
    """Определяет устройство (GPU/CPU) и количество доступных GPU."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Доступно {num_gpus} GPU: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        return torch.device("cuda"), num_gpus
    device = torch.device("cpu")
    print("[INFO] CPU режим (GPU не обнаружен)")
    return device, 1


def compute_class_weights(dataset, mode: str, device) -> torch.Tensor:
    """Веса классов для CrossEntropyLoss по дисбалансу обучающей выборки.

    mode:
        off           -> нет весов (равные единице)
        inverse       -> 1 / class_count
        inverse_norm  -> 1/class_count, нормировано к среднему=1
        effective_num -> "effective number of samples" (Cui et al., 2019)
                         w[c] = (1 - beta) / (1 - beta^n_c), beta=0.999
    """
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


class SymmetryAwareFocalLoss(nn.Module):
    """Focal Loss с усилением штрафа за путаницу 0°↔180°.
    
    - Focal Loss (gamma) фокусируется на сложных примерах (низкая уверенность).
    - SYMMETRY_WEIGHT умножает loss для samples с true class ∈ {0, 180}.
    """
    
    def __init__(self, gamma: float = 2.0, symmetry_weight: float = 1.0, 
                 class_weights: torch.Tensor = None, device=None):
        super().__init__()
        self.gamma = gamma
        self.symmetry_weight = symmetry_weight
        self.class_weights = class_weights
        self.device = device
        
    def forward(self, logits, targets):
        """
        logits: [batch, num_classes]
        targets: [batch] — индексы классов
        """
        ce_loss = nn.functional.cross_entropy(
            logits, targets, 
            weight=self.class_weights, 
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.symmetry_weight > 1.0:
            symmetry_mask = (targets == 0) | (targets == 1)
            focal_loss = focal_loss * (1.0 + (self.symmetry_weight - 1.0) * symmetry_mask.float())
        
        return focal_loss.mean()


PAD_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}


def get_input_size() -> int:
    """Определяет размер входа из MODEL_NAME."""
    if "384" in C.MODEL_NAME:
        return 384
    elif "224" in C.MODEL_NAME:
        return 224
    else:
        return 224  # default


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
    """Нарисовать чёрные полосы на 1..4 случайных краях (имитация тёмных полей сканера)."""
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


def add_top_border(img: Image.Image, thickness_frac: float, color: tuple) -> Image.Image:
    """Добавить тонкую рамку только сверху (помогает модели выучить "верх" документа)."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    thickness = max(1, int(h * thickness_frac))
    draw.rectangle([0, 0, w, thickness], fill=color)
    return img


class OrientationDataset(Dataset):
    """On-the-fly датасет ориентации поверх ImageFolder для ViT Large.

    Ключевые особенности:
    - VerticalFlip критичен для различения 0° vs 180°.
    - ROTATION_AUG сбалансированно распределяет все 4 ориентации.
    """

    def __init__(self, base: datasets.ImageFolder, mode: str, post_tfm):
        self.samples = base.samples                    # [(path, class_idx)]
        self.classes = base.classes                    # ['0','180','270','90'] или с 'other'
        self.targets = [c for _, c in base.samples]    # исходное распределение (для весов)
        
        # Преобразуем все имена классов в int для совместимости
        try:
            self.idx_to_angle = {i: int(name) for i, name in enumerate(base.classes)}
            self.angle_to_idx = {int(name): i for i, name in enumerate(base.classes)}
            
            # Проверяем, что все 4 угла присутствуют
            expected_angles = {0, 90, 180, 270}
            actual_angles = set(self.idx_to_angle.values())
            if not expected_angles.issubset(actual_angles):
                missing = expected_angles - actual_angles
                print(f"[WARN] Отсутствуют углы в датасете: {missing}")
                print(f"       Доступные углы: {actual_angles}")
        except ValueError as e:
            # Если есть класс "other", преобразуем его в индекс
            print(f"[INFO] Обнаружен нечисловой класс: {e}")
            self.idx_to_angle = {}
            self.angle_to_idx = {}
            for i, name in enumerate(base.classes):
                try:
                    angle = int(name)
                    self.idx_to_angle[i] = angle
                    self.angle_to_idx[angle] = i
                except ValueError:
                    # Это класс "other" или другой нечисловой - присваиваем индекс
                    other_idx = i
                    self.idx_to_angle[i] = -1  # special marker for 'other'
                    self.angle_to_idx[-1] = i
        
        self.mode = mode
        self.post_tfm = post_tfm
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

        # 0) Top border: тонкая рамка только сверху (ДО поворота, чтобы при повороте рамка оказалась на разных сторонах)
        #    Это создаёт асимметрию: модель учит "где рамка — там верх документа"
        if self.mode == "train" and getattr(C, "TOP_BORDER_AUG", False) and random.random() < getattr(C, "TOP_BORDER_PROB", 0.5):
            img = add_top_border(img, getattr(C, "TOP_BORDER_THICKNESS", 0.008), getattr(C, "TOP_BORDER_COLOR", (100, 100, 100)))

        # 1) Поворот ориентации (кратно 90 — без потерь). label-space CW=a => PIL rotate(-a).
        a = 90 * k
        if a:
            img = img.rotate((-a) % 360, expand=True)
        
        # 2) Вычисляем новый угол после поворота
        new_angle = (orig_angle + a) % 360
        
        # 3) Находим индекс класса для нового угла
        if new_angle in self.angle_to_idx:
            target = int(self.angle_to_idx[new_angle])
        else:
            print(f"[WARN] Угол {new_angle} не найден в angle_to_idx. Используем класс для 0°.")
            target = self.angle_to_idx.get(0, 0)

        # 4) VerticalFlip для улучшения 0° vs 180° (только train)
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
            # 5) Перекос сканирования (метка не меняется), углы — белые/чёрные как фон сканера
            if C.SKEW_AUG and random.random() < C.SKEW_PROB:
                ang = random.uniform(-C.SKEW_MAX_DEG, C.SKEW_MAX_DEG)
                fill = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
                img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=fill)
            # 6) Чёрные поля по краям листа
            if C.BORDER_AUG and random.random() < C.BORDER_PROB:
                img = add_black_borders(img, C.BORDER_MAX_FRAC)
            # 7) Горизонтальный флип (опционально; для документов обычно выкл.)
            if C.HFLIP_AUG and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # 7b) Bottom marker: тёмная полоса внизу (имитация подписи/печати)
            #     Помогает модели различать 0° vs 180° по асимметрии.
            if getattr(C, "BOTTOM_MARKER_AUG", False) and random.random() < getattr(C, "BOTTOM_MARKER_PROB", 0.3):
                w, h = img.size
                draw = ImageDraw.Draw(img)
                marker_h = random.randint(max(1, int(h * 0.02)), max(2, int(h * 0.05)))
                gray_val = random.randint(30, 80)
                draw.rectangle([0, h - marker_h, w, h], fill=(gray_val, gray_val, gray_val))

        # 8) Вход модели: кроп текст-плотной области (crop) или вся страница (full)
        input_size = get_input_size()
        if getattr(C, "INPUT_MODE", "full") == "crop":
            jit = C.CROP_JITTER if self.mode == "train" else 0
            img = text_crop.crop_text_region(img, jitter=jit, out_size=input_size)
        elif getattr(C, "RESIZE_MODE", "letterbox") == "letterbox":
            if self.mode == "train":
                pad = C.LETTERBOX_PAD_TRAIN
                fill = random.choice(list(PAD_COLORS.values())) if pad == "random" \
                    else PAD_COLORS.get(pad, (255, 255, 255))
            else:
                fill = (255, 255, 255)
            img = letterbox(img, input_size, fill)
        else:
            img = img.resize((input_size, input_size), Image.BILINEAR)

        # Финальная проверка: target должен быть индексом класса [0, n_classes)
        num_classes = len(self.classes)
        if not (0 <= target < num_classes):
            print(f"[ERROR] Target {target} out of range [0, {num_classes})! Clipping to 0.")
            target = max(0, min(target, num_classes - 1))

        return self.post_tfm(img), target


class ViTAttention(nn.Module):
    """Дополнительный attention-механизм для улучшения различения 0° vs 180°.

    Добавляет механизм внимания после основного блока признаков, чтобы модель
    могла фокусироваться на важных областях (заголовки, подписи, логотипы).
    """

    def __init__(self, in_features: int, hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, in_features),
            nn.Sigmoid()  # Attention weights (0-1)
        )

    def forward(self, x):
        """x: [batch, features, height, width] или [batch, features]."""
        if x.dim() == 4:
            attention_weights = self.attention(x)
            return x * attention_weights
        else:
            # Если уже flattened (после pooling)
            attention_weights = self.attention(x.unsqueeze(-1)).squeeze(-1)
            return x * attention_weights


def build_model(num_classes, device, num_gpus):
    """Строит модель Vision Transformer Large с attention-механизмом."""
    
    input_size = get_input_size()
    print(f"[INFO] Загрузка модели с размером входа {input_size}x{input_size}")
    
    # Загружаем предобученную ViT модель через timm (LARGE версия)
    if C.USE_VIT_LAYERS == "large":
        model = timm.create_model(f'vit_large_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 1024  # vit_large имеет 1024 features
    else:
        model = timm.create_model(f'vit_base_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 768  # vit_base имеет 768 features

    # Добавляем attention-механизм (если включён)
    if C.USE_ATTENTION:
        model.head = ViTAttention(in_features, hidden_size=C.ATTENTION_HIDDEN_SIZE, 
                                  dropout=C.ATTENTION_DROPOUT)
    
    # Заменяем классификатор на OUTPUT_CLASSES
    if hasattr(model, 'head'):
        in_features = model.head.in_features if hasattr(model.head, 'in_features') else in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    # Распределяем модель по GPU (DataParallel для 2xGPU)
    if num_gpus > 1 and C.DISTRIBUTED_TRAINING:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    return model


def evaluate(model, loader, device, criterion, num_classes):
    """Возвращает: val_loss, accuracy (plain), balanced_acc (macro-recall),
    per_class_recall (list), confusion matrix (num_classes x num_classes).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
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
            
            for t, p in zip(y.view(-1), pred.view(-1)):
                confmat[t.long(), p.long()] += 1

    val_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    # per-class recall = диагональ / сумма по строке (истинные примеры класса)
    per_class_total = confmat.sum(dim=1)
    per_class_correct = confmat.diag()
    per_class_recall = []
    for c in range(num_classes):
        n = int(per_class_total[c].item())
        per_class_recall.append(per_class_correct[c].item() / n if n > 0 else float("nan"))
    
    valid = [r for r in per_class_recall if r == r]  # отфильтровать NaN
    balanced_acc = sum(valid) / len(valid) if valid else 0.0
    
    return val_loss, accuracy, balanced_acc, per_class_recall, confmat.tolist()


def main():
    device, num_gpus = get_device()
    print(f"Устройство: {device} (GPU count: {num_gpus})")
    
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_loaders(device, num_gpus)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(C.OUTPUT_CLASSES, device, num_gpus)
    
    weight_mode = C.CLASS_WEIGHTS
    if C.ROTATION_AUG:
        print("[INFO] ROTATION_AUG включён — классы балансируются ротацией, "
              "веса классов отключены (CLASS_WEIGHTS игнорируется).")
        weight_mode = "off"
    
    class_weights = compute_class_weights(train_loader.dataset, weight_mode, device)
    
    focal_gamma = getattr(C, "FOCAL_GAMMA", 2.0)
    symmetry_weight = getattr(C, "SYMMETRY_WEIGHT", 1.0)
    if focal_gamma > 0:
        criterion = SymmetryAwareFocalLoss(
            gamma=focal_gamma, 
            symmetry_weight=symmetry_weight,
            class_weights=class_weights,
            device=device
        )
        print(f"Loss: SymmetryAwareFocalLoss (gamma={focal_gamma}, symmetry_weight={symmetry_weight})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Loss: CrossEntropyLoss")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=C.LEARNING_RATE, 
        weight_decay=C.WEIGHT_DECAY
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and C.USE_AMP))

    scheduler = None
    if getattr(C, "LR_SCHEDULER", "off") == "cosine":
        # Cosine decay с учетом gradient accumulation
        effective_epochs = C.EPOCHS // max(1, C.GRADIENT_ACCUMULATION_STEPS)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS, eta_min=getattr(C, "LR_MIN", 1e-6))
        print(f"LR scheduler: cosine (eta_min={getattr(C, 'LR_MIN', 1e-6)})")

    best_metric_name = getattr(C, "BEST_METRIC", "balanced")
    class_label_names = [idx_to_class[i] for i in range(C.OUTPUT_CLASSES)]
    
    best_score = -1.0
    best_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    final_path = C.MODELS_DIR / C.MODEL_FILENAME
    
    history = {
        "class_to_idx": class_to_idx, 
        "idx_to_class": idx_to_class, 
        "epochs": [],
        "num_gpus": num_gpus,
        "model_name": C.MODEL_NAME
    }

    no_improve = 0
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        t0 = time.time()
        
        running = 0.0
        
        # Gradient accumulation для увеличения effective batch size
        grad_accum_steps = max(1, C.GRADIENT_ACCUMULATION_STEPS)
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and C.USE_AMP)):
                out = model(x)
                loss = criterion(out, y)
            
            # Scale loss для gradient accumulation
            loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Step optimizer после накопления градиентов
            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        
        val_loss, val_acc, bal_acc, per_class_recall, confmat = evaluate(
            model, val_loader, device, criterion, C.OUTPUT_CLASSES)
        
        if scheduler is not None:
            scheduler.step()
        
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        print(f"Epoch {epoch:2d}/{C.EPOCHS}  {dt:5.1f}s  lr={cur_lr:.2e}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  bal_acc={bal_acc:.4f}")
        
        recall_desc = "  ".join(
            f"{class_label_names[i]}:{per_class_recall[i]:.3f}" for i in range(C.OUTPUT_CLASSES))
        print(f"           per-class recall: {recall_desc}")
        
        history["epochs"].append({
            "epoch": epoch, 
            "lr": cur_lr,
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "val_acc": val_acc, 
            "balanced_acc": bal_acc,
            "per_class_recall": per_class_recall, 
            "confusion_matrix": confmat,
        })

        score = bal_acc if best_metric_name == "balanced" else val_acc
        
        if score > best_score:
            best_score = score
            torch.save(model, best_path)  # сохраняем всю модель целиком
            print(f"  -> лучший результат сохранён: {best_path.name} "
                  f"({best_metric_name}={best_score:.4f}, val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if C.EARLY_STOP_PATIENCE and no_improve >= C.EARLY_STOP_PATIENCE:
                print(f"Early stopping: нет улучшения {no_improve} эпох (метрика={best_metric_name}).")
                break

    # финальная модель
    torch.save(model, final_path)  # сохраняем всю модель целиком
    
    history["best_metric"] = best_metric_name
    history["best_score"] = best_score
    
    with open(C.MODELS_DIR / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(f"\nОбучение завершено. Лучшая метрика ({best_metric_name})={best_score:.4f}")
    print(f"  лучшая модель: {best_path}")
    print(f"  финальная:     {final_path}")


def build_loaders(device, num_gpus):
    """Строит DataLoader для train и val наборов."""
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Увеличенный batch size для 2xGPU (но меньше чем у Base из-за Large модели)
    effective_batch_size = C.BATCH_SIZE * max(1, num_gpus) if device.type == "cuda" else C.BATCH_SIZE
    
    print(f"[INFO] Effective batch size: {effective_batch_size} (batch={C.BATCH_SIZE}, GPUs={num_gpus})")
    
    # Аугментация для train (более агрессивная для ViT Large)
    train_post = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        norm,
    ])
    
    # Валидация без аугментации (кроме ротации)
    val_post = transforms.Compose([transforms.ToTensor(), norm])

    base_train = datasets.ImageFolder(str(C.TRAIN_DIR))  # transform=None -> PIL
    base_val = datasets.ImageFolder(str(C.VAL_DIR))

    if len(base_train) == 0 or len(base_val) == 0:
        print("ERROR: Пустой обучающий или валидационный набор. Запустите prepare_dataset.py.", file=sys.stderr)
        sys.exit(1)

    class_names = base_train.classes
    print(f"Классы ImageFolder: {class_names} (samples: {len(base_train)} train / {len(base_val)} val)")
    
    if list(map(int, class_names)) != list(map(int, map(str, C.CLASS_NAMES))):
        print(f"[WARN] Порядок классов ImageFolder ({class_names}) отличается от CLASS_NAMES ({C.CLASS_NAMES}).")

    _inp = getattr(C, "INPUT_MODE", "full")
    if _inp == "crop":
        _win = (f"adaptive(target_glyph={C.CROP_TARGET_GLYPH})" if getattr(C, "CROP_ADAPTIVE", True)
                else f"win={C.CROP_WINDOW}")
        _inp_desc = f"crop[{_win}]->{C.IMAGE_SIZE}, jitter={C.CROP_JITTER}, fallback_full={getattr(C, 'CROP_FALLBACK_FULL', True)}"
    else:
        _inp_desc = f"full/{getattr(C, 'RESIZE_MODE', 'letterbox')}"
    
    print(f"Вход: {_inp_desc}  |  ROTATION_AUG={C.ROTATION_AUG} "
          f"(val_full={C.ROTATION_AUG and C.ROTATION_AUG_VAL_FULL})  |  "
          f"SKEW_AUG={C.SKEW_AUG}  BORDER_AUG={C.BORDER_AUG}  HFLIP_AUG={C.HFLIP_AUG}  "
          f"BOTTOM_MARKER_AUG={getattr(C, 'BOTTOM_MARKER_AUG', False)}")

    train_ds = OrientationDataset(base_train, "train", train_post)
    val_ds = OrientationDataset(base_val, "val", val_post)

    pin = device.type == "cuda"
    
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=pin, drop_last=False)
    
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=pin)
    
    return train_loader, val_loader, base_train.class_to_idx


if __name__ == "__main__":
    main()
