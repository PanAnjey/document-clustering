import json
from collections import defaultdict

with open(r'D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Фильтруем только файлы с conf > 0 (успешно обработанные)
successful = [r for r in results if r['qwen_conf'] > 0]
failed = [r for r in results if r['qwen_conf'] == 0]

print(f"Всего файлов: {len(results)}")
print(f"Успешно обработано: {len(successful)}")
print(f"Не обработано (conf=0): {len(failed)}")

if successful:
    correct = sum(1 for r in successful if r['qwen_correct'])
    accuracy = correct / len(successful)
    
    print(f"\n=== Результаты для успешно обработанных ===")
    print(f"Правильных: {correct}/{len(successful)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Per-class метрики
    print("\n=== Per-class метрики ===")
    for angle in [0, 90, 180, 270]:
        angle_results = [r for r in successful if r['true_angle'] == angle]
        if angle_results:
            correct = sum(1 for r in angle_results if r['qwen_correct'])
            total_angle = len(angle_results)
            print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")
    
    # Confusion matrix
    print("\n=== Confusion matrix ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in successful:
        confmatrix[r['true_angle']][r['qwen_pred']] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}" for a in [0, 90, 180, 270])
    print(header)
    for true_a in [0, 90, 180, 270]:
        row = [confmatrix[true_a][pred_a] for pred_a in [0, 90, 180, 270]]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
