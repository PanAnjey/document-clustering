"""Проверка конфликтов меток в датасете для Vision Transformer.

- Проверяет распределение классов в TRAIN_DIR и VAL_DIR.
- Выявляет дисбаланс между классами (особенно 0° vs 180°).
- Генерирует отчёт о потенциальных проблемах.
"""
import json
from collections import Counter
from pathlib import Path

import config as C


def count_files_in_dir(directory: Path) -> dict:
    """Подсчитывает количество файлов в каждой поддиректории."""
    counts = {}
    
    if not directory.exists():
        return counts
    
    for cls in C.CLASS_NAMES[:4]:  # только 0, 90, 180, 270
        cls_dir = directory / str(cls)
        if cls_dir.exists():
            files = [f for f in cls_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS]
            counts[str(cls)] = len(files)
    
    return counts


def analyze_distribution(train_counts: dict, val_counts: dict):
    """Анализирует распределение классов."""
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    print("\n=== Анализ распределения классов ===\n")
    
    # Общая статистика
    print(f"Всего в train: {total_train} изображений")
    print(f"Всего в val:   {total_val} изображений\n")
    
    # Распределение по классам (train)
    print("Распределение в TRAIN:")
    for cls, count in sorted(train_counts.items(), key=lambda x: int(x[0])):
        pct = 100 * count / total_train if total_train > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:>4}°: {count:5d} ({pct:6.2f}%) {bar}")
    
    # Распределение по классам (val)
    print("\nРаспределение в VAL:")
    for cls, count in sorted(val_counts.items(), key=lambda x: int(x[0])):
        pct = 100 * count / total_val if total_val > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:>4}°: {count:5d} ({pct:6.2f}%) {bar}")
    
    # Проверка дисбаланса
    print("\n=== Проверка дисбаланса ===\n")
    
    if total_train > 0:
        max_count = max(train_counts.values())
        min_count = min(train_counts.values())
        
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Максимальный класс: {max_count} изображений")
        print(f"Минимальный класс:  {min_count} изображений")
        print(f"Соотношение макс/мин: {ratio:.2f}x")
        
        if ratio > 3:
            print("\n⚠️  ВНИМАНИЕ: Сильный дисбаланс классов!")
            print("   Рекомендуется использовать CLASS_WEIGHTS='effective_num'")
    
    # Проверка симметрии 0° vs 180° (критично для проблемы путаницы)
    print("\n=== Проверка симметрии 0° vs 180° ===\n")
    
    count_0 = train_counts.get("0", 0)
    count_180 = train_counts.get("180", 0)
    
    if count_0 > 0 and count_180 > 0:
        ratio_0_180 = max(count_0, count_180) / min(count_0, count_180)
        
        print(f"Класс 0°: {count_0} изображений")
        print(f"Класс 180°: {count_180} изображений")
        print(f"Соотношение: {ratio_0_180:.2f}x")
        
        if ratio_0_180 > 2:
            print("\n⚠️  ВНИМАНИЕ: Дисбаланс между 0° и 180°!")
            print("   Это может усилить путаницу при обучении.")
            print("   Рекомендации:")
            print("   1. Добавить VerticalFlip аугментацию (VERTICAL_FLIP_AUG=True)")
            print("   2. Использовать WEIGHTS='effective_num'")
            print("   3. Увеличить количество данных для меньшего класса")


def main():
    """Основная функция."""
    
    print("=" * 60)
    print("Проверка конфликтов меток - Vision Transformer")
    print("=" * 60 + "\n")
    
    # Подсчёт файлов
    train_counts = count_files_in_dir(C.TRAIN_DIR)
    val_counts = count_files_in_dir(C.VAL_DIR)
    
    if not train_counts:
        print(f"[ERROR] Директория TRAIN не найдена или пуста: {C.TRAIN_DIR}")
        return
    
    # Анализ распределения
    analyze_distribution(train_counts, val_counts)
    
    # Сохранение отчёта в JSON
    report = {
        "train": train_counts,
        "val": val_counts,
        "total_train": sum(train_counts.values()),
        "total_val": sum(val_counts.values()),
        "config": {
            "CLASS_WEIGHTS": C.CLASS_WEIGHTS,
            "VERTICAL_FLIP_AUG": C.VERTICAL_FLIP_AUG,
            "ROTATION_AUG": C.ROTATION_AUG,
        }
    }
    
    report_path = C.MODELS_DIR / "label_conflicts_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
