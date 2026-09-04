"""Сравнение метрик до и после retrain."""
import json
from pathlib import Path


def compare_metrics():
    """Сравнивает результаты обучения до и после retrain."""
    
    print("=" * 60)
    print("Сравнение метрик: До vs После retrain")
    print("=" * 60 + "\n")
    
    # Путь к истории обучения
    train_history = Path(r"D:\FileOrganizer\TRAIN\models\train_history.json")
    retrain_history = Path(r"D:\FileOrganizer\TRAIN\models\retrain_history.json")
    
    if not train_history.exists():
        print(f"[WARN] История обучения не найдена: {train_history}")
        return
    
    # Загрузка истории
    with open(train_history, "r", encoding="utf-8") as fh:
        train_data = json.load(fh)
    
    best_train_score = train_data.get("best_score", 0)
    print(f"До retrain (лучшая метрика): {best_train_score:.4f}")
    
    if retrain_history.exists():
        with open(retrain_history, "r", encoding="utf-8") as fh:
            retrain_data = json.load(fh)
        
        best_retrain_score = retrain_data.get("best_score", 0)
        print(f"После retrain (лучшая метрика): {best_retrain_score:.4f}")
        
        improvement = best_retrain_score - best_train_score
        print(f"\nУлучшение: +{improvement:.4f} ({100*improvement:.2f}%)")
    else:
        print("\n[INFO] История retrain ещё не создана (запустите 5_retrain.py)")


def analyze_confusion_matrix():
    """Анализирует матрицу ошибок."""
    
    results_file = Path(r"D:\FileOrganizer\TRAIN\models\inference_results.json")
    
    if not results_file.exists():
        print(f"[WARN] Результаты инференса не найдены: {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    
    print("\n" + "=" * 60)
    print("Анализ ошибок по классам")
    print("=" * 60 + "\n")
    
    errors_by_class = {}
    total_corrected = 0
    total_not_corrected = 0
    
    for result in results.get("results", []):
        if result.get("corrected"):
            total_corrected += 1
        else:
            total_not_corrected += 1
        
        # Можно добавить анализ конкретных ошибок здесь
    
    print(f"Всего обработано: {total_corrected + total_not_corrected}")
    print(f"Скорректировано: {total_corrected} ({100*total_corrected/(total_corrected+total_not_corrected):.2f}%)")
    print(f"Без коррекции: {total_not_corrected} ({100*total_not_corrected/(total_corrected+total_not_corrected):.2f}%)")


if __name__ == "__main__":
    compare_metrics()
    analyze_confusion_matrix()
