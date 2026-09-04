import json
with open(r'D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json', encoding='utf-8') as f:
    data = json.load(f)

# Проверяем первые 10 результатов
for r in data['results'][:10]:
    print(f"File: {r['file'][:50]}")
    print(f"  True: {r['true_angle']}, Pred: {r['qwen_pred']}, Conf: {r['qwen_conf']}")
