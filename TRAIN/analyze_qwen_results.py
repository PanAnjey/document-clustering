import json
from collections import defaultdict

with open(r'D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
total = len(results)

# Подсчёт метрик
correct = sum(1 for r in results if r['qwen_correct'])
accuracy = correct / total if total > 0 else 0

print(f"Всего обработано: {total} файлов")
print(f"Правильных: {correct}")
print(f"Accuracy: {accuracy:.2%}")

# Per-class метрики
print("\n=== Per-class метрики ===")
for angle in [0, 90, 180, 270]:
    angle_results = [r for r in results if r['true_angle'] == angle]
    if angle_results:
        correct = sum(1 for r in angle_results if r['qwen_correct'])
        total_angle = len(angle_results)
        print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")

# Confusion matrix
print("\n=== Confusion matrix ===")
confmatrix = defaultdict(lambda: defaultdict(int))
for r in results:
    confmatrix[r['true_angle']][r['qwen_pred']] += 1

header = "         pred: " + "  ".join(f"{a:3d}" for a in [0, 90, 180, 270])
print(header)
for true_a in [0, 90, 180, 270]:
    row = [confmatrix[true_a][pred_a] for pred_a in [0, 90, 180, 270]]
    print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")

# Статистика предсказаний
print("\n=== Распределение предсказаний ===")
pred_counts = defaultdict(int)
for r in results:
    pred_counts[r['qwen_pred']] += 1

for pred in sorted(pred_counts.keys()):
    print(f"  {pred:3d}°: {pred_counts[pred]} файлов")
