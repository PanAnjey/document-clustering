"""Анализ паттернов ошибок в датасете retrain."""
import json
from pathlib import Path
from collections import Counter


def analyze_retrain_dataset():
    """Анализирует распределение классов и потенциальные проблемы."""
    
    print("=" * 60)
    print("Анализ датасета для дообучения (retrain)")
    print("=" * 60 + "\n")
    
    retrain_dir = Path(r"D:\FileOrganizer\TRAIN\retrain")
    
    if not retrain_dir.exists():
        print(f"[ERROR] Директория не найдена: {retrain_dir}")
        return
    
    # Подсчитаем количество файлов в каждой директории
    class_counts = {}
    for cls in [0, 90, 180, 270]:
        cls_dir = retrain_dir / str(cls)
        if cls_dir.exists():
            files = list(cls_dir.glob("*"))
            class_counts[cls] = len(files)
        else:
            class_counts[cls] = 0
    
    # Вывод статистики
    print("Распределение классов в retrain:")
    total = sum(class_counts.values())
    for cls, count in sorted(class_counts.items()):
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:>4}°: {count:5d} ({pct:6.2f}%) {bar}")
    
    # Проверка дисбаланса
    if total > 0:
        max_count = max(class_counts.values())
        min_count = min(c for c in class_counts.values() if c > 0)
        
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nДисбаланс классов:")
        print(f"  Максимальный класс: {max_count} изображений")
        print(f"  Минимальный класс:  {min_count} изображений (если >0)")
        print(f"  Соотношение макс/мин: {ratio:.2f}x")
        
        if ratio > 3:
            print("\n⚠️  ВНИМАНИЕ: Сильный дисбаланс классов!")
            print("   Рекомендации:")
            print("   1. Добавить аугментацию для меньших классов")
            print("   2. Использовать CLASS_WEIGHTS='effective_num'")
    
    # Проверка на наличие ошибок 0° vs 180°
    count_0 = class_counts.get(0, 0)
    count_180 = class_counts.get(180, 0)
    
    if count_0 > 0 and count_180 > 0:
        ratio_0_180 = max(count_0, count_180) / min(count_0, count_180)
        
        print(f"\nПроверка симметрии 0° vs 180°:")
        print(f"  Класс 0°: {count_0} изображений")
        print(f"  Класс 180°: {count_180} изображений")
        print(f"  Соотношение: {ratio_0_180:.2f}x")
        
        if ratio_0_180 > 2:
            print("\n⚠️  ВНИМАНИЕ: Дисбаланс между 0° и 180°!")
            print("   Это может усилить путаницу при обучении.")
    
    # Проверка на наличие не-сканов (если есть)
    other_dir = retrain_dir / "other"
    if other_dir.exists():
        files = list(other_dir.glob("*"))
        print(f"\nНе-сканы (other): {len(files)} изображений")


def analyze_inference_results():
    """Анализирует результаты инференса для выявления паттернов ошибок."""
    
    results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results.json")
    
    if not results_file.exists():
        print(f"[WARN] Файл результатов не найден: {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    
    print("\n" + "=" * 60)
    print("Анализ результатов инференса")
    print("=" * 60 + "\n")
    
    # Подсчитаем ошибки по классам
    errors_by_class = Counter()
    low_confidence_count = 0
    
    for result in results.get("results", []):
        if result.get("low_confidence"):
            low_confidence_count += 1
        
        # Можно добавить анализ конкретных ошибок здесь
    
    print(f"Всего изображений: {len(results.get('results', []))}")
    print(f"С низкой уверенностью (conf<{0.85}): {low_confidence_count}")


if __name__ == "__main__":
    analyze_retrain_dataset()
    analyze_inference_results()
