"""Оркестратор полного обучения «с нуля» с текущими настройками config.py.

Запускает по порядку:
    1) 1_prepare_dataset.py  — пересборка train/val из distrib (+ очистка retrain)
    2) 2_train.py            — полное обучение MobileNetV3 (вход: INPUT_MODE)
    3) 3_test_inference.py   — тестовый прогон (опционально, флаг --with-inference)

Каждый шаг запускается тем же интерпретатором, что и этот скрипт (sys.executable),
поэтому запускайте раннер нужным Python:
    D:\\VENV\\LLM\\Scripts\\python.exe run_full_training.py

Флаги:
    --skip-prepare     не запускать 1_prepare_dataset (использовать готовые train/val)
    --keep-retrain     не очищать RETRAIN_DIR в 1_prepare_dataset
    --with-inference   после обучения запустить 3_test_inference.py
    --backup-models    скопировать существующие модели в models/backup_<timestamp>/
    --yes              не спрашивать подтверждение

ВНИМАНИЕ: новое обучение ПЕРЕЗАПИШЕТ orientation_*_best.pth / *.pth в MODELS_DIR.
Используйте --backup-models, если старую модель нужно сохранить.
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent


def _print_config_summary():
    inp = getattr(C, "INPUT_MODE", "full")
    inp_desc = (f"crop(win={C.CROP_WINDOW}->{C.IMAGE_SIZE}, jitter={C.CROP_JITTER})"
                if inp == "crop" else f"full/{getattr(C, 'RESIZE_MODE', 'letterbox')}")
    print("=" * 70)
    print("ПОЛНОЕ ОБУЧЕНИЕ — текущие настройки:")
    print(f"  distrib:        {C.DISTRIB_DIR}")
    print(f"  train/val:      {C.TRAIN_DIR}  |  {C.VAL_DIR}  (val_split={C.VAL_SPLIT})")
    print(f"  models:         {C.MODELS_DIR}")
    print(f"  вход модели:    {inp_desc}")
    print(f"  IMAGE_SIZE={C.IMAGE_SIZE}  BATCH_SIZE={C.BATCH_SIZE}  EPOCHS={C.EPOCHS}  "
          f"LR={C.LEARNING_RATE}  scheduler={getattr(C, 'LR_SCHEDULER', 'off')}")
    print(f"  выбор best:     {getattr(C, 'BEST_METRIC', 'balanced')}  "
          f"early_stop_patience={C.EARLY_STOP_PATIENCE}")
    print(f"  аугментации:    ROTATION_AUG={C.ROTATION_AUG} (val_full={C.ROTATION_AUG and C.ROTATION_AUG_VAL_FULL})  "
          f"SKEW={C.SKEW_AUG}  BORDER={C.BORDER_AUG}  HFLIP={C.HFLIP_AUG}")
    print(f"  инференс:       {C.INFERENCE_INPUT_DIR}  conf_threshold={C.INFERENCE_CONFIDENCE_THRESHOLD}")
    print("=" * 70)


def _run(step_name, args):
    print(f"\n{'#' * 70}\n# {step_name}\n# {' '.join(args)}\n{'#' * 70}", flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, *args], cwd=str(HERE))
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ОШИБКА] '{step_name}' завершился с кодом {res.returncode} (за {dt:.1f}s). Остановка.",
              file=sys.stderr)
        sys.exit(res.returncode)
    print(f"\n[OK] '{step_name}' завершён за {dt:.1f}s")


def _backup_models():
    md = C.MODELS_DIR
    files = list(md.glob("orientation_*.pth")) if md.exists() else []
    if not files:
        print("  (бэкап моделей: существующих *.pth не найдено — пропуск)")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = md / f"backup_{stamp}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)
    for extra in ("train_history.json", "inference_results.json"):
        p = md / extra
        if p.exists():
            shutil.copy2(p, dst / extra)
    print(f"  бэкап моделей -> {dst} ({len(files)} файлов)")


def main():
    ap = argparse.ArgumentParser(description="Полное обучение с текущими настройками.")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--keep-retrain", action="store_true")
    ap.add_argument("--with-inference", action="store_true")
    ap.add_argument("--backup-models", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    _print_config_summary()
    print(f"Интерпретатор: {sys.executable}")
    if not args.yes:
        ans = input("\nНачать полное обучение? Это ПЕРЕЗАПИШЕТ текущую модель. [y/N]: ").strip().lower()
        if ans not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return

    t_all = time.time()
    if args.backup_models:
        print("\n--- Бэкап существующих моделей ---")
        _backup_models()

    if not args.skip_prepare:
        prep_args = ["1_prepare_dataset.py"]
        if args.keep_retrain:
            prep_args.append("--keep-retrain")
        _run("Этап 1: подготовка датасета", prep_args)
    else:
        print("\n[skip] Этап 1 (подготовка) пропущен (--skip-prepare).")

    _run("Этап 2: обучение", ["2_train.py"])

    if args.with_inference:
        _run("Этап 3: тестовый инференс", ["3_test_inference.py"])
    else:
        print("\n[info] Этап 3 (инференс) не запускался. Запустите вручную при необходимости:")
        print(f"       {sys.executable} 3_test_inference.py")

    print(f"\n{'=' * 70}\nГОТОВО. Полный цикл за {time.time() - t_all:.1f}s")
    print(f"  модель:  {C.MODELS_DIR / C.BEST_MODEL_FILENAME}")
    print(f"  история: {C.MODELS_DIR / 'train_history.json'}")


if __name__ == "__main__":
    main()
