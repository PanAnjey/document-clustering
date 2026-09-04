"""Полный пайплайн обучения Vision Transformer для определения ориентации сканов.

Запускает все этапы последовательно:
1. Подготовка датасета (prepare_dataset.py)
2. Проверка дисбаланса (check_label_conflicts.py)
3. Обучение модели (train.py)
4. Тестовый прогон (test_inference.py)
5. Ручная проверка (review_app.py) - опционально

Оптимизировано для 2x NVIDIA RTX PRO 4000 48GB VRAM.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list = None):
    """Запускает скрипт с аргументами."""
    
    script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'=' * 60}")
    print(f"Запуск: {script_name} {' '.join(args or [])}")
    print('=' * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Скрипт {script_name} завершён с ошибкой (код {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Полный пайплайн обучения ViT для ориентации сканов")
    
    # Этапы выполнения
    parser.add_argument("--prepare", action="store_true", default=True,
                        help="Подготовить датасет (1_prepare_dataset.py)")
    parser.add_argument("--check-conflicts", action="store_true", default=True,
                        help="Проверить дисбаланс классов (check_label_conflicts.py)")
    parser.add_argument("--train", action="store_true", default=True,
                        help="Обучить модель (2_train.py)")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Протестировать на новых данных (3_test_inference.py)")
    parser.add_argument("--review", action="store_true", default=False,
                        help="Открыть GUI для ручной проверки (4_review_app.py)")
    parser.add_argument("--retrain", action="store_true", default=True,
                        help="Fine-tune на ошибках (5_retrain.py)")
    
    # Параметры обучения
    parser.add_argument("--epochs", type=int, default=None,
                        help="Количество эпох для обучения (переопределяет config.EPOCHS)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (переопределяет config.BATCH_SIZE)")
    
    # Опции
    parser.add_argument("--keep-retrain", action="store_true",
                        help="Не очищать RETRAIN_DIR при подготовке датасета")
    parser.add_argument("--no-check-conflicts", action="store_true",
                        help="Пропустить проверку дисбаланса")
    parser.add_argument("--no-test", action="store_true",
                        help="Пропустить тестовый прогон после обучения")
    
    args = parser.parse_args()

    print("=" * 60)
    print("Vision Transformer Training Pipeline")
    print("Оптимизировано для 2x NVIDIA RTX PRO 4000 (48GB VRAM)")
    print("=" * 60 + "\n")

    # Этап 1: Подготовка датасета
    if args.prepare:
        prepare_args = []
        if args.keep_retrain:
            prepare_args.append("--keep-retrain")
        
        run_script("1_prepare_dataset.py", prepare_args)
    
    # Этап 2: Проверка дисбаланса (опционально)
    if not args.no_check_conflicts and args.check_conflicts:
        run_script("check_label_conflicts.py")
    
    # Этап 3: Обучение модели
    if args.train:
        train_args = []
        
        if args.epochs is not None:
            # Временное изменение config.EPOCHS через env var или переопределение
            import os
            os.environ["TRAIN_EPOCHS"] = str(args.epochs)
        
        run_script("2_train.py", train_args)
    
    # Этап 4: Тестовый прогон (опционально)
    if args.test and not args.no_test:
        run_script("3_test_inference.py")
    
    # Этап 5: Ручная проверка (GUI)
    if args.review:
        run_script("4_review_app.py", [])
    
    # Этап 6: Fine-tuning на ошибках
    if args.retrain and not args.no_test:
        run_script("5_retrain.py")
    
    print("\n" + "=" * 60)
    print("Пайплайн завершён!")
    print("=" * 60)
    print("\nРезультаты:")
    print(f"  Лучшая модель: {Path('D:\\FileOrganizer\\TRAIN\\models').joinpath(C.BEST_MODEL_FILENAME)}")
    print(f"  Результаты инференса: {Path('D:\\FileOrganizer\\TRAIN\\models').joinpath('inference_results.json')}")


if __name__ == "__main__":
    main()
