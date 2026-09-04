"""Дообучение MobileNetV3-Large для классификации ориентации сканов (4 класса: 0,90,180,270).

- Загружает torchvision mobilenet_v3_large с предобученными весами IMAGENET1K_V2.
- Заменяет классификатор на OUTPUT_CLASSES.
- Датасеты: ImageFolder для TRAIN_DIR и VAL_DIR (структура {0,90,180,270}).
- Аугментация: Resize/RandomHorizontalFlip (горизонтальный флип сохраняет угол
  ориентации) + ColorJitter (brightness/contrast ±15%, saturation ±10%, hue=0 —
  без изменения оттенка, т.к. цвет печатей/подписей может быть полезным сигналом).
  Вертикальный флип/повороты НЕ используются (меняют метку ориентации).
- Сохраняет лучшую (по val accuracy) и финальную модель в MODELS_DIR.
- Поддерживает early stopping.
- Логирует метрики и сохраняет историю в models/train_history.json.
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
from torchvision import datasets, models, transforms

import config as C
import text_crop


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(dataset, mode: str, device) -> torch.Tensor:
    """Веса классов для CrossEntropyLoss по дисбалансу обучающей выборки.

    dataset.targets — список меток ImageFolder (порядок = dataset.class_to_idx).
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


class OrientationDataset(Dataset):
    """On-the-fly датасет ориентации поверх ImageFolder.

    ВАЖНО: ориентация = свойство СОДЕРЖИМОГО (направление чтения), а не формы листа.
    В каждом классе есть и портретные, и альбомные документы (это корректно). Поворот
    на кратный 90° меняет метку чисто геометрически: (orig + поворот) % 360 — независимо
    от того, портретный документ или альбомный. Поэтому ротация по ВСЕМ классам даёт
    каждому классу сбалансированную смесь пропорций, и модель не может использовать
    соотношение сторон как решающий признак (вынуждена опираться на содержимое).

    - Поворот на 0/90/180/270 c пересчётом метки = (orig + поворот) % 360.
      Направление согласовано с 3_test_inference.correct_angle: метка X = документ
      повёрнут на X° CW, значит синтез класса делается через img.rotate(-a) (CW на a).
    - train: + перекос (skew), + чёрные поля, + letterbox со случайным цветом полей.
    - val (ROTATION_AUG_VAL_FULL): каждое изображение во всех 4 ориентациях
      детерминированно (сбалансированная честная метрика), letterbox с белыми полями.
    """

    def __init__(self, base: datasets.ImageFolder, mode: str, post_tfm):
        self.samples = base.samples                    # [(path, class_idx)]
        self.classes = base.classes                    # ['0','180','270','90']
        self.targets = [c for _, c in base.samples]    # исходное распределение (для весов)
        self.idx_to_angle = {i: int(name) for i, name in enumerate(base.classes)}
        self.angle_to_idx = {int(name): i for i, name in enumerate(base.classes)}
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

        # 1) Поворот ориентации (кратно 90 — без потерь). label-space CW=a => PIL rotate(-a).
        a = 90 * k
        if a:
            img = img.rotate((-a) % 360, expand=True)
        target = self.angle_to_idx[(orig_angle + a) % 360]

        if self.mode == "train":
            # 2) Перекос сканирования (метка не меняется), углы — белые/чёрные как фон сканера
            if C.SKEW_AUG and random.random() < C.SKEW_PROB:
                ang = random.uniform(-C.SKEW_MAX_DEG, C.SKEW_MAX_DEG)
                fill = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
                img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=fill)
            # 3) Чёрные поля по краям листа
            if C.BORDER_AUG and random.random() < C.BORDER_PROB:
                img = add_black_borders(img, C.BORDER_MAX_FRAC)
            # 4) Горизонтальный флип (опционально; для документов обычно выкл.)
            if C.HFLIP_AUG and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 5) Вход модели: кроп текст-плотной области (crop) или вся страница (full)
        if getattr(C, "INPUT_MODE", "full") == "crop":
            jit = C.CROP_JITTER if self.mode == "train" else 0
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

        return self.post_tfm(img), target


def build_loaders(device):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_post = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        norm,
    ])
    val_post = transforms.Compose([transforms.ToTensor(), norm])

    base_train = datasets.ImageFolder(str(C.TRAIN_DIR))  # transform=None -> PIL
    base_val = datasets.ImageFolder(str(C.VAL_DIR))

    if len(base_train) == 0 or len(base_val) == 0:
        print("[ERROR] Пустой обучающий или валидационный набор. Запустите prepare_dataset.py.", file=sys.stderr)
        sys.exit(1)

    class_names = base_train.classes
    print(f"Классы ImageFolder: {class_names} (samples: {len(base_train)} train / {len(base_val)} val)")
    if list(map(int, class_names)) != list(map(int, map(str, C.CLASS_NAMES))):
        print(f"[WARN] Порядок классов ImageFolder ({class_names}) отличается от CLASS_NAMES ({C.CLASS_NAMES}).")
        print("       Порядок CLASS_NAMES НЕ влияет на train.py (используется порядок ImageFolder).")
        print("       ВАЖНО: test_inference.py сопоставляет индексы с CLASS_NAMES — наоборот, проверяйте!")

    _inp = getattr(C, "INPUT_MODE", "full")
    if _inp == "crop":
        _win = (f"adaptive(target_glyph={C.CROP_TARGET_GLYPH})" if getattr(C, "CROP_ADAPTIVE", True)
                else f"win={C.CROP_WINDOW}")
        _inp_desc = f"crop[{_win}]->{C.IMAGE_SIZE}, jitter={C.CROP_JITTER}, fallback_full={getattr(C, 'CROP_FALLBACK_FULL', True)}"
    else:
        _inp_desc = f"full/{getattr(C, 'RESIZE_MODE', 'letterbox')}"
    print(f"Вход: {_inp_desc}  |  ROTATION_AUG={C.ROTATION_AUG} "
          f"(val_full={C.ROTATION_AUG and C.ROTATION_AUG_VAL_FULL})  |  "
          f"SKEW_AUG={C.SKEW_AUG}  BORDER_AUG={C.BORDER_AUG}  HFLIP_AUG={C.HFLIP_AUG}")

    train_ds = OrientationDataset(base_train, "train", train_post)
    val_ds = OrientationDataset(base_val, "val", val_post)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=pin)
    return train_loader, val_loader, base_train.class_to_idx


def build_model(num_classes):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device, criterion, num_classes):
    """Возвращает: val_loss, accuracy (plain), balanced_acc (macro-recall),
    per_class_recall (list), confusion matrix (num_classes x num_classes, строки=истина)."""
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
    device = get_device()
    print(f"Устройство: {device}")
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_loaders(device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(C.OUTPUT_CLASSES).to(device)
    weight_mode = C.CLASS_WEIGHTS
    if C.ROTATION_AUG:
        print("[INFO] ROTATION_AUG включён — классы балансируются ротацией, "
              "веса классов отключены (CLASS_WEIGHTS игнорируется).")
        weight_mode = "off"
    class_weights = compute_class_weights(train_loader.dataset, weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and C.USE_AMP))

    scheduler = None
    if getattr(C, "LR_SCHEDULER", "off") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS, eta_min=getattr(C, "LR_MIN", 1e-6))
        print(f"LR scheduler: cosine (eta_min={getattr(C, 'LR_MIN', 1e-6)})")

    best_metric_name = getattr(C, "BEST_METRIC", "balanced")
    class_label_names = [idx_to_class[i] for i in range(C.OUTPUT_CLASSES)]
    best_score = -1.0
    best_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    final_path = C.MODELS_DIR / C.MODEL_FILENAME
    history = {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "epochs": []}

    no_improve = 0
    for epoch in range(1, C.EPOCHS + 1):
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
            "epoch": epoch, "lr": cur_lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "balanced_acc": bal_acc,
            "per_class_recall": per_class_recall, "confusion_matrix": confmat,
        })

        score = bal_acc if best_metric_name == "balanced" else val_acc
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "idx_to_class": idx_to_class,
                        "num_classes": C.OUTPUT_CLASSES,
                        "model_name": C.MODEL_NAME}, best_path)
            print(f"  -> лучший результат сохранён: {best_path.name} "
                  f"({best_metric_name}={best_score:.4f}, val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if C.EARLY_STOP_PATIENCE and no_improve >= C.EARLY_STOP_PATIENCE:
                print(f"Early stopping: нет улучшения {no_improve} эпох (метрика={best_metric_name}).")
                break

    # финальная модель
    torch.save({"model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": C.OUTPUT_CLASSES,
                "model_name": C.MODEL_NAME}, final_path)
    history["best_metric"] = best_metric_name
    history["best_score"] = best_score
    with open(C.MODELS_DIR / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(f"Обучение завершено. Лучшая метрика ({best_metric_name})={best_score:.4f}")
    print(f"  лучшая модель: {best_path}")
    print(f"  финальная:     {final_path}")


if __name__ == "__main__":
    main()