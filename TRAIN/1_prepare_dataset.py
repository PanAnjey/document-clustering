"""Подготовка тренировочного и валидационного наборов из distrib для Vision Transformer.

Берёт файлы из DISTRIB_DIR/{0,90,180,270} и распределяет их по TRAIN_DIR и VAL_DIR
в аналогичную структуру поддиректорий, со стратифицированным разделением по VAL_SPLIT.

Файлы КОПИРУЮТСЯ (не перемещаются), чтобы distrib оставался нетронутым.
Перед разделением TRAIN_DIR/VAL_DIR очищаются, чтобы избежать дублирования.

RETRAIN_DIR также очищается (новый датасет + обучение с нуля => retrain-набор
должен собираться заново из ошибок НОВОЙ модели). Отключить: флаг --keep-retrain.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import config as C


def _make_writable(p: Path):
    """Снять ReadOnly атрибут на Windows (chmod не помогает для NTFS ACL)."""
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
    """Полностью удалить директорию (со всем содержимым) и пересоздать пустую.
    Снимает ReadOnly-атрибут на Windows для упорных файлов через onerror-callback.
    """
    if path.exists():
        shutil.rmtree(path, onerror=_on_rmtree_error)
    path.mkdir(parents=True, exist_ok=True)


def clear_retrain_dir():
    """Очистить RETRAIN_DIR (вместе с review_report.csv) и пересоздать пустые
    поддиректории классов RETRAIN_CLASSES.
    """
    clear_dir(C.RETRAIN_DIR)
    for angle in C.RETRAIN_CLASSES:
        (C.RETRAIN_DIR / str(angle)).mkdir(parents=True, exist_ok=True)
    print(f"  RETRAIN_DIR очищен: {C.RETRAIN_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка train/val из distrib для ViT.")
    parser.add_argument("--keep-retrain", action="store_true",
                        help="НЕ очищать RETRAIN_DIR (по умолчанию очищается).")
    args = parser.parse_args()

    random.seed(C.RANDOM_SEED)

    if not C.DISTRIB_DIR.exists():
        print(f"[ERROR] Директория датасета не существует: {C.DISTRIB_DIR}", file=sys.stderr)
        sys.exit(1)

    classes = sorted([d.name for d in C.DISTRIB_DIR.iterdir() if d.is_dir()])
    if not classes:
        print(f"[ERROR] В {C.DISTRIB_DIR} нет поддиректорий классов.", file=sys.stderr)
        sys.exit(1)

    print(f"Классы: {classes}")
    clear_dir(C.TRAIN_DIR)
    clear_dir(C.VAL_DIR)
    if args.keep_retrain:
        print(f"  RETRAIN_DIR сохранён (--keep-retrain): {C.RETRAIN_DIR}")
    else:
        clear_retrain_dir()

    total_train = 0
    total_val = 0
    for cls in classes:
        src_dir = C.DISTRIB_DIR / cls
        files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
        random.shuffle(files)
        n_val = max(1, int(round(len(files) * C.VAL_SPLIT))) if files else 0
        val_files = files[:n_val]
        train_files = files[n_val:]

        train_cls_dir = C.TRAIN_DIR / cls
        val_cls_dir = C.VAL_DIR / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copyfile(f, train_cls_dir / f.name)
        for f in val_files:
            shutil.copyfile(f, val_cls_dir / f.name)

        print(f"  {cls}: всего {len(files):4d}  -> train {len(train_files):4d}  val {len(val_files):3d}")
        total_train += len(train_files)
        total_val += len(val_files)

    print(f"\nГотово. train={total_train} val={total_val}")
    print(f"  TRAIN_DIR: {C.TRAIN_DIR}")
    print(f"  VAL_DIR:   {C.VAL_DIR}")


if __name__ == "__main__":
    main()
