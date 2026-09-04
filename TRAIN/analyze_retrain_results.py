"""Детальный анализ результатов после retrain."""
import json
from pathlib import Path


def analyze_inference_results():
    """Анализирует результаты инференса по классам."""
    
    print("=" * 60)
    print("Детальный анализ ошибок по классам")
    print("=" * 60 + "\n")
    
    results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results.json")
    
    if not results_file.exists():
        print(f"[WARN] Результаты инференса не найдены: {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    
    # Подсчёт по классам
    class_stats = {}
    total_corrected = 0
    total_not_corrected = 0
    low_confidence_count = 0
    
    for result in results.get("results", []):
        predicted_orient = result.get("predicted_orientation", "N/A")
        
        if predicted_orient not in class_stats:
            class_stats[predicted_orient] = {
                "total": 0,
                "corrected": 0,
                "not_corrected": 0,
                "low_confidence": 0,
                "avg_confidence": []
            }
        
        stats = class_stats[predicted_orient]
        stats["total"] += 1
        
        if result.get("corrected"):
            stats["corrected"] += 1
            total_corrected += 1
        else:
            stats["not_corrected"] += 1
            total_not_corrected += 1
        
        if result.get("low_confidence"):
            stats["low_confidence"] += 1
            low_confidence_count += 1
        
        conf = result.get("confidence", 0)
        stats["avg_confidence"].append(conf)
    
    # Вывод статистики
    print(f"Всего обработано: {len(results.get('results', []))}")
    print(f"\nОбщая статистика:")
    print(f"  Скорректировано: {total_corrected} ({100*total_corrected/len(results['results']):.2f}%)")
    print(f"  Без коррекции: {total_not_corrected} ({100*total_not_corrected/len(results['results']):.2f}%)")
    print(f"  С низкой уверенностью (conf<85%): {low_confidence_count} ({100*low_confidence_count/len(results['results']):.2f}%)")
    
    # Статистика по классам
    print("\nСтатистика по классам:")
    for cls in sorted(class_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        stats = class_stats[cls]
        corrected_pct = 100 * stats["corrected"] / stats["total"] if stats["total"] > 0 else 0
        avg_conf = sum(stats["avg_confidence"]) / len(stats["avg_confidence"]) if stats["avg_confidence"] else 0
        
        print(f"\nКласс {cls}°:")
        print(f"  Всего: {stats['total']}")
        print(f"  Скорректировано: {stats['corrected']} ({corrected_pct:.2f}%)")
        print(f"  Без коррекции: {stats['not_corrected']}")
        print(f"  Низкая уверенность: {stats['low_confidence']}")
        print(f"  Средняя уверенность: {avg_conf:.2%}")


def check_0_vs_180_accuracy():
    """Проверяет точность для 0° vs 180°."""
    
    results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results.json")
    
    if not results_file.exists():
        print(f"[WARN] Результаты инференса не найдены: {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    
    # Подсчёт ошибок для 0° и 180°
    errors_0_vs_180 = {"predicted": {}, "corrected": {}}
    
    for result in results.get("results", []):
        predicted_orient = str(result.get("predicted_orientation", ""))
        
        if predicted_orient not in ["0", "180"]:
            continue
        
        if predicted_orient not in errors_0_vs_180["predicted"]:
            errors_0_vs_180["predicted"][predicted_orient] = {"total": 0, "corrected": 0}
        
        errors_0_vs_180["predicted"][predicted_orient]["total"] += 1
        
        if result.get("corrected"):
            errors_0_vs_180["predicted"][predicted_orient]["corrected"] += 1
    
    print("\n" + "=" * 60)
    print("Точность для 0° vs 180°")
    print("=" * 60)
    
    for cls in ["0", "180"]:
        stats = errors_0_vs_180["predicted"].get(cls, {"total": 0, "corrected": 0})
        accuracy = 100 * stats["corrected"] / stats["total"] if stats["total"] > 0 else 0
        
        print(f"\nКласс {cls}°:")
        print(f"  Всего: {stats['total']}")
        print(f"  Скорректировано: {stats['corrected']} ({accuracy:.2f}%)")


if __name__ == "__main__":
    analyze_inference_results()
    check_0_vs_180_accuracy()
