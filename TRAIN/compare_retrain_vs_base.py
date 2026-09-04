"""Сравнение результатов базовой и дообученной (retrain) модели."""
import json
from pathlib import Path


def compare_models():
    """Сравнивает результаты базовой и retrain моделей."""
    
    print("=" * 60)
    print("Сравнение: Базовая vs Дообученная модель")
    print("=" * 60 + "\n")
    
    # Путь к результатам
    base_results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results.json")
    retrain_results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results_retrain.json")
    
    if not base_results_file.exists():
        print(f"[WARN] Результаты базовой модели не найдены: {base_results_file}")
        return
    
    with open(base_results_file, "r", encoding="utf-8") as fh:
        base_data = json.load(fh)
    
    if retrain_results_file.exists():
        with open(retrain_results_file, "r", encoding="utf-8") as fh:
            retrain_data = json.load(fh)
        
        print("Базовая модель:")
        analyze_results(base_data, "base")
        
        print("\n" + "-" * 60 + "\n")
        
        print("Дообученная модель (retrain):")
        analyze_results(retrain_data, "retrain")
        
        # Сравнение
        compare_metrics(base_data, retrain_data)
    else:
        print(f"[INFO] Результаты дообученной модели не найдены: {retrain_results_file}")
        print("[INFO] Запустите 3_test_inference_retrain.py для тестирования")


def analyze_results(data, model_name):
    """Анализирует результаты одной модели."""
    
    results = data.get("results", [])
    total = len(results)
    
    if total == 0:
        print(f"  Нет результатов")
        return
    
    corrected = sum(1 for r in results if r.get("corrected"))
    not_corrected = sum(1 for r in results if not r.get("corrected"))
    low_conf = sum(1 for r in results if r.get("low_confidence"))
    
    # Средняя уверенность
    confidences = [r.get("confidence", 0) for r in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"  Всего файлов: {total}")
    print(f"  Скорректировано: {corrected} ({100*corrected/total:.2f}%)")
    print(f"  Без коррекции: {not_corrected} ({100*not_corrected/total:.2f}%)")
    print(f"  Низкая уверенность (conf<85%): {low_conf} ({100*low_conf/total:.2f}%)")
    print(f"  Средняя уверенность: {avg_conf:.2%}")


def compare_metrics(base_data, retrain_data):
    """Сравнивает метрики двух моделей."""
    
    base_results = base_data.get("results", [])
    retrain_results = retrain_data.get("results", [])
    
    if not base_results or not retrain_results:
        print("\n[WARN] Недостаточно данных для сравнения")
        return
    
    # Подсчёт метрик
    def calc_metrics(results):
        total = len(results)
        corrected = sum(1 for r in results if r.get("corrected"))
        low_conf = sum(1 for r in results if r.get("low_confidence"))
        confidences = [r.get("confidence", 0) for r in results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total": total,
            "corrected_pct": 100 * corrected / total if total > 0 else 0,
            "low_conf_pct": 100 * low_conf / total if total > 0 else 0,
            "avg_confidence": avg_conf
        }
    
    base_metrics = calc_metrics(base_results)
    retrain_metrics = calc_metrics(retrain_results)
    
    print("\n" + "=" * 60)
    print("Сравнение метрик")
    print("=" * 60)
    
    print(f"\n{'Метрика':<35} {'Базовая':>12} {'Re-train':>12} {'Изменение':>12}")
    print("-" * 71)
    
    # Скорректировано
    diff_corr = retrain_metrics["corrected_pct"] - base_metrics["corrected_pct"]
    print(f"{'Скорректировано (%)':<35} {base_metrics['corrected_pct']:>11.2f}% {retrain_metrics['corrected_pct']:>11.2f}% {'+' if diff_corr >= 0 else ''}{diff_corr:>10.2f}%")
    
    # Низкая уверенность
    diff_low_conf = retrain_metrics["low_conf_pct"] - base_metrics["low_conf_pct"]
    print(f"{'Низкая уверенность (%)':<35} {base_metrics['low_conf_pct']:>11.2f}% {retrain_metrics['low_conf_pct']:>11.2f}% {'+' if diff_low_conf >= 0 else ''}{diff_low_conf:>10.2f}%")
    
    # Средняя уверенность
    diff_avg_conf = retrain_metrics["avg_confidence"] - base_metrics["avg_confidence"]
    print(f"{'Средняя уверенность (%)':<35} {base_metrics['avg_confidence']*100:>11.2f}% {retrain_metrics['avg_confidence']*100:>11.2f}% {'+' if diff_avg_conf >= 0 else ''}{diff_avg_conf*100:>10.2f}%")
    
    # Итоговая оценка
    print("\n" + "=" * 60)
    print("Итоговая оценка:")
    print("=" * 60)
    
    improvements = []
    if diff_corr > 0:
        improvements.append(f"+{diff_corr:.2f}% скорректировано")
    if diff_low_conf < 0:
        improvements.append(f"-{abs(diff_low_conf):.2f}% низкой уверенности")
    if diff_avg_conf > 0:
        improvements.append(f"+{diff_avg_conf*100:.2f}% средняя уверенность")
    
    if improvements:
        print("✅ Улучшения:")
        for imp in improvements:
            print(f"   • {imp}")
    else:
        print("⚠️  Значимых улучшений не обнаружено")


if __name__ == "__main__":
    compare_models()
