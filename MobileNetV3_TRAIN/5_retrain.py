"""Повторное дообучение обученной ранее модели на тестовом наборе из RETRAIN_DIR.

Идея — fine-tuning уже обученной модели исправленными/новыми данными, собранными
через review_app.py в RETRAIN_DIR/{0,90,180,270}.

Отличия от первичного обучения (2_train.py):
- Старт НЕ с ImageNet-весов, а с уже дообученной модели из MODELS_DIR (best/final).
- Малый LR (RETRAIN_LEARNING_RATE), меньше эпох (RETRAIN_EPOCHS).
- Используется СОХРАНЁННЫЙ порядок классов из чекпойнта базовой модели (idx→угол),
  чтобы не было рассинхрона меток между старым и новым обучением.
- Пустые классы в retrain допустимы: они не участвуют в тренировке, но модель
  сохраняет все OUTPUT_CLASSES выходов (только класс с 0 сэмплов не получает градиента).
  Для `compute_class_weights` класс-пустышка получает вес 0 (чтобы не доминировать).
- Валидация: если хотя бы 2 класса с валидационными сэмплами — считается, иначе
  можно отключить (RETRAIN_VAL_SPLIT=0) — тогда model.save по последней эпохе.
- Веса и `idx_to_class` берутся из базовой модели, чтобы_MAPPING сохранился.
- Новая модель сохраняется как RETRAIN_FILE_PREFIX + "_v{N}.pth" (N = авто-инкремент,
  чтобы не затирать предыдущие версии). Если RETRAIN_REPLACE_BEST — дополнительно
  копируется поверх BEST_MODEL_FILENAME (чтобы 3_test_inference.py сразу использовал).
"""
import bisect
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

import config as C
import text_crop


# ---------- геометрия / аугментации (синхронно с 2_train.py) ----------
PAD_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}


def letterbox(img: Image.Image, size: int, fill) -> Image.Image:
    """Вписать в квадрат size×size с сохранением пропорций + поля цвета fill."""
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def add_black_borders(img: Image.Image, max_frac: float) -> Image.Image:
    """Чёрные полосы на 1..4 случайных краях (имитация тёмных полей сканера)."""
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


# ---------- свои типы ----------
class OrientationFlatDataset(Dataset):
    """On-the-fly датасет ориентации поверх плоского списка (path, label_idx).

    Синхронно с 2_train.OrientationDataset: поворот 0/90/180/270 c пересчётом метки,
    letterbox, skew, чёрные поля. Направление поворота согласовано с
    3_test_inference.correct_angle (синтез класса = img.rotate(-a)).

    ВАЖНО: ориентация = свойство СОДЕРЖИМОГО, а не формы листа. В каждом классе есть
    и портретные, и альбомные документы (корректно). Поворот меняет метку чисто
    геометрически ((orig + поворот) % 360), независимо от пропорций, поэтому соотношение
    сторон НЕ является решающим признаком.
    """
    def __init__(self, samples, mode, post_tfm, idx_to_angle, rotation_aug):
        self.samples = samples                         # [(path, label_idx)]
        self.mode = mode
        self.post_tfm = post_tfm
        self.idx_to_angle = {int(k): int(v) for k, v in idx_to_angle.items()}
        self.angle_to_idx = {v: k for k, v in self.idx_to_angle.items()}
        self.rotation_aug = rotation_aug
        self.val_full = (mode == "val" and rotation_aug and C.ROTATION_AUG_VAL_FULL)
        self.n = len(self.samples)

    def __len__(self):
        return self.n * 4 if self.val_full else self.n

    def __getitem__(self, idx):
        if self.val_full:
            sample_idx, k = idx // 4, idx % 4
        else:
            sample_idx = idx
            k = random.randint(0, 3) if (self.mode == "train" and self.rotation_aug) else 0

        path, lbl = self.samples[sample_idx]
        orig_angle = self.idx_to_angle[int(lbl)]
        img = Image.open(path).convert("RGB")

        a = 90 * k
        if a:
            img = img.rotate((-a) % 360, expand=True)   # label-space CW=a => PIL rotate(-a)
        target = self.angle_to_idx[(orig_angle + a) % 360]

        if self.mode == "train":
            if C.SKEW_AUG and random.random() < C.SKEW_PROB:
                ang = random.uniform(-C.SKEW_MAX_DEG, C.SKEW_MAX_DEG)
                fill = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
                img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=fill)
            if C.BORDER_AUG and random.random() < C.BORDER_PROB:
                img = add_black_borders(img, C.BORDER_MAX_FRAC)
            if C.HFLIP_AUG and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

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


# ---------- helpers ----------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_base_checkpoint(device) -> dict:
    fname = C.BEST_MODEL_FILENAME if C.RETRAIN_BASE_MODEL == "best" else C.MODEL_FILENAME
    p = C.MODELS_DIR / fname
    if not p.exists():
        print(f"[ERROR] Базовая модель не найдена: {p}", file=sys.stderr)
        print("        Сначала выполните обучение через 2_train.py.", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(p, map_location=device, weights_only=True)
    print(f"Загружена базовая модель: {p.name}")
    return ckpt


def _build_model_with_base(ckpt, device):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_f = model.classifier[-1].in_features
    num_classes = ckpt.get("num_classes", C.OUTPUT_CLASSES)
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        print(f"[ERROR] state_dict не подошёл: {e}", file=sys.stderr)
        sys.exit(1)
    return model.to(device), num_classes


def _idx_to_class_map(ckpt):
    idx_to_class = ckpt.get("idx_to_class")
    if idx_to_class is not None:
        return {int(k): int(v) for k, v in idx_to_class.items()}
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is not None:
        return {int(v): int(k) for k, v in class_to_idx.items()}
    # fallback
    return {i: int(a) for i, a in enumerate(C.CLASS_NAMES)}


def _collect_retrain_samples(idx_to_angle: dict):
    """Собрать (path, lbl) по всем классам из RETRAIN_DIR/{angle_name}.

    Возвращает:
      train_samples, val_samples (могут быть пустые),
      per_class_counts (по всем собранным — для логирования).
    """
    angles_to_lbl = {int(a): lbl for lbl, a in idx_to_angle.items()}
    all_samples_by_lbl = {}
    for angle in C.RETRAIN_CLASSES:
        d = C.RETRAIN_DIR / str(angle)
        lbl = angles_to_lbl.get(int(angle))
        if lbl is None:
            print(f"[WARN] угол {angle} отсутствует в mapping базовой модели — пропускаем.")
            continue
        if not d.exists():
            print(f"[WARN] папка отсутствует: {d} (класс {angle}).")
            continue
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
        if not files:
            print(f"[WARN] папка {d} пуста (класс {angle}).")
        all_samples_by_lbl.setdefault(lbl, []).extend(files)
        print(f"  класс {angle}° (idx={lbl}): {len(files)} файлов")

    # разделяем по классам — стратифицированно
    import random as _r
    _r.seed(C.RANDOM_SEED)
    train, val = [], []
    for lbl, files in all_samples_by_lbl.items():
        _r.shuffle(files)
        if C.RETRAIN_VAL_SPLIT > 0 and len(files) >= 3:
            n_val = max(1, int(round(len(files) * C.RETRAIN_VAL_SPLIT)))
            val.extend((str(p), lbl) for p in files[:n_val])
            train.extend((str(p), lbl) for p in files[n_val:])
        else:
            train.extend((str(p), lbl) for p in files)
    return train, val


def _class_weights(train_samples, num_classes, mode: str, device):
    counts = Counter(int(l) for _, l in train_samples)
    counts_arr = torch.tensor([float(counts.get(i, 0)) for i in range(num_classes)],
                              dtype=torch.float32)
    if mode == "off":
        w = torch.ones(num_classes)
    elif mode == "inverse":
        w = 1.0 / counts_arr.clamp(min=1.0)
    elif mode == "inverse_norm":
        w = 1.0 / counts_arr.clamp(min=1.0)
        s = w.sum()
        if s > 0:
            w = w * (num_classes / s)
    elif mode == "effective_num":
        beta = 0.999
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts_arr.clamp(min=1.0)))
        s = w.sum()
        if s > 0:
            w = w * (num_classes / s)
    else:
        w = torch.ones(num_classes)
    # классы с нулём сэмплов → вес 0 (чтобы не доминировали в loss)
    w = w * (counts_arr > 0).float()
    # нормируем к среднему не-нулевому = 1
    nz = (w > 0).sum()
    if nz > 0 and w.sum() > 0:
        w = w * (nz / w.sum())
    print(f"Веса классов ({mode}): " + " ".join(
        f"idx{i}={w[i]:.3f}({'n='+str(int(counts_arr[i])) if counts_arr[i] > 0 else 'нет'})" for i in range(num_classes)))
    return w.to(device)


def _evaluate(model, loader, device, criterion, num_classes):
    """Возвращает: val_loss, accuracy, balanced_acc, per_class_recall, confusion_matrix."""
    nan_recall = [float("nan")] * num_classes
    if not loader or len(loader.dataset) == 0:
        return float("nan"), float("nan"), float("nan"), nan_recall, []
    model.eval()
    loss_s, correct, total = 0.0, 0, 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and C.USE_AMP)):
                out = model(x)
                loss = criterion(out, y)
            loss_s += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.view(-1), pred.view(-1)):
                confmat[t.long(), p.long()] += 1

    accuracy = correct / max(total, 1)
    per_class_total = confmat.sum(dim=1)
    per_class_correct = confmat.diag()
    per_class_recall = []
    for c in range(num_classes):
        nc = int(per_class_total[c].item())
        per_class_recall.append(per_class_correct[c].item() / nc if nc > 0 else float("nan"))
    valid = [r for r in per_class_recall if r == r]
    balanced_acc = sum(valid) / len(valid) if valid else 0.0
    return loss_s / max(total, 1), accuracy, balanced_acc, per_class_recall, confmat.tolist()


def _next_version_path(prefix: str) -> Path:
    """Найти следующий свободный _v{N}.pth."""
    existing = [int(f.stem.split("_v")[1])
                for f in C.MODELS_DIR.glob(f"{prefix}_v*.pth")
                if f.stem.startswith(f"{prefix}_v") and f.stem.split("_v")[1].isdigit()]
    n = (max(existing) + 1) if existing else 1
    return C.MODELS_DIR / f"{prefix}_v{n}.pth"


# ---------- main ----------
def main():
    device = get_device()
    print(f"Устройство: {device}")
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not C.RETRAIN_DIR.exists():
        print(f"[ERROR] RETRAIN_DIR не существует: {C.RETRAIN_DIR}", file=sys.stderr)
        sys.exit(1)

    ckpt = _load_base_checkpoint(device)
    idx_to_angle = _idx_to_class_map(ckpt)
    print(f"_mapping базовой модели (idx→angle): {idx_to_angle}")

    model, num_classes = _build_model_with_base(ckpt, device)

    print("Сбор samples из retrain:")
    train_samples, val_samples = _collect_retrain_samples(idx_to_angle)

    if not train_samples:
        print("[ERROR] retrain-набор пуст. Сначала соберите пары в review_app.py.", file=sys.stderr)
        sys.exit(1)

    print(f"Итого: train={len(train_samples)}, val={len(val_samples)}")

    # Resize/аугментации синхронны с 2_train.py (letterbox + on-the-fly ротация + skew + поля).
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_post = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        norm,
    ])
    val_post = transforms.Compose([transforms.ToTensor(), norm])

    rotation_aug = getattr(C, "RETRAIN_ROTATION_AUG", C.ROTATION_AUG)
    print(f"Resize: {getattr(C, 'RESIZE_MODE', 'letterbox')}  |  RETRAIN_ROTATION_AUG={rotation_aug} "
          f"(val_full={rotation_aug and C.ROTATION_AUG_VAL_FULL})  |  "
          f"SKEW_AUG={C.SKEW_AUG}  BORDER_AUG={C.BORDER_AUG}  HFLIP_AUG={C.HFLIP_AUG}")

    train_ds = OrientationFlatDataset(train_samples, "train", train_post, idx_to_angle, rotation_aug)
    val_ds = OrientationFlatDataset(val_samples, "val", val_post, idx_to_angle, rotation_aug) if val_samples else None

    pin = device.type == "cuda"
    bs = max(1, min(C.RETRAIN_BATCH_SIZE, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=pin) if val_ds else None

    weight_mode = C.CLASS_WEIGHTS
    if rotation_aug:
        print("[INFO] RETRAIN_ROTATION_AUG включён — классы балансируются ротацией, "
              "веса классов отключены (CLASS_WEIGHTS игнорируется).")
        weight_mode = "off"
    weights = _class_weights(train_samples, num_classes, weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.RETRAIN_LEARNING_RATE,
                                  weight_decay=C.RETRAIN_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and C.USE_AMP))

    scheduler = None
    if getattr(C, "LR_SCHEDULER", "off") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=C.RETRAIN_EPOCHS, eta_min=getattr(C, "LR_MIN", 1e-6))
        print(f"LR scheduler: cosine (eta_min={getattr(C, 'LR_MIN', 1e-6)})")

    best_metric_name = getattr(C, "BEST_METRIC", "balanced")
    class_label_names = [idx_to_angle[i] for i in range(num_classes)]
    best_acc = -1.0
    best_path = _next_version_path(C.RETRAIN_FILE_PREFIX)
    has_val = val_loader is not None and len(val_loader.dataset) > 0
    history = {"class_to_idx": ckpt.get("class_to_idx"),
               "idx_to_class": ckpt.get("idx_to_class"),
               "base_model": (C.BEST_MODEL_FILENAME if C.RETRAIN_BASE_MODEL == "best"
                              else C.MODEL_FILENAME),
               "retrain_dir": str(C.RETRAIN_DIR),
               "num_classes": num_classes, "epochs": []}
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
        train_loss = running / len(train_loader.dataset)
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        epoch_rec = {"epoch": epoch, "lr": cur_lr, "train_loss": train_loss}
        if has_val:
            val_loss, val_acc, bal_acc, per_class_recall, confmat = _evaluate(
                model, val_loader, device, criterion, num_classes)
            # метрика отслеживания: balanced_acc (рекомендуется) или plain acc
            track_acc = bal_acc if best_metric_name == "balanced" else val_acc
            track_msg = (f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  bal_acc={bal_acc:.4f}")
            epoch_rec.update({"val_loss": val_loss, "val_acc": val_acc,
                              "balanced_acc": bal_acc, "per_class_recall": per_class_recall,
                              "confusion_matrix": confmat})
        else:
            track_acc = -train_loss  # нет валидации — сохраняем по train_loss
            track_msg = f"train_loss={train_loss:.4f} (нет val)"
            per_class_recall = None
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{C.RETRAIN_EPOCHS}  {dt:5.1f}s  lr={cur_lr:.2e}  "
              f"train_loss={train_loss:.4f}  {track_msg}")
        if has_val:
            recall_desc = "  ".join(
                f"{class_label_names[i]}:{per_class_recall[i]:.3f}" for i in range(num_classes))
            print(f"           per-class recall: {recall_desc}")
        history["epochs"].append(epoch_rec)

        is_better = track_acc > best_acc
        if is_better:
            best_acc = track_acc
            torch.save({"model_state": model.state_dict(),
                        "class_to_idx": ckpt.get("class_to_idx"),
                        "idx_to_class": ckpt.get("idx_to_class"),
                        "num_classes": num_classes,
                        "model_name": C.MODEL_NAME,
                        "retrain_from": C.RETRAIN_BASE_MODEL}, best_path)
            metric_label = (best_metric_name if has_val else "neg_train_loss")
            print(f"  -> лучшая модель сохранена: {best_path.name} ({metric_label}={best_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if C.RETRAIN_EARLY_STOP_PATIENCE and no_improve >= C.RETRAIN_EARLY_STOP_PATIENCE:
                print(f"Early stopping: нет улучшения {no_improve} эпох.")
                break

    # при необходимости замещаем «best»
    if C.RETRAIN_REPLACE_BEST:
        target = C.MODELS_DIR / C.BEST_MODEL_FILENAME
        backup = None
        if target.exists():
            backup = _next_version_path("orientation_mobilenet_v3_large_best_backup")
            shutil.copy2(target, backup)
            print(f"  резервная копия прежнего best: {backup.name}")
        shutil.copy2(best_path, target)
        print(f"  новый best модель: {target}")
    # сохраняем историю рядом с файлом новой версии
    hist_path = best_path.with_suffix(".history.json")
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    print("\n=== Повторное дообучение завершено ===")
    print(f"  новая модель: {best_path}")
    if C.RETRAIN_REPLACE_BEST:
        print(f"  также установлена как best: {C.MODELS_DIR / C.BEST_MODEL_FILENAME}")
    print(f"  история: {hist_path}")


if __name__ == "__main__":
    main()