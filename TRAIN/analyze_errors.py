import json

with open(r'D:\FileOrganizer\TRAIN\models\ensemble_vit_paddle_report.json', encoding='utf-8') as f:
    data = json.load(f)

errors = [r for r in data['results'] if not r['ensemble_correct']]

print(f"Всего ошибок ансамбля: {len(errors)}\n")

# Группируем по истинному классу
from collections import defaultdict
by_class = defaultdict(list)
for r in errors:
    by_class[r['true_angle']].append(r)

for angle in sorted(by_class.keys()):
    print(f"=== Истинный класс {angle}°: {len(by_class[angle])} ошибок ===")
    for r in by_class[angle]:
        fname = r['file'][:60]
        print(f"  {fname:60s} pred={r['ensemble_pred']:3d}  vit={r['vit_pred']:3d}({r['vit_conf']:.2f})  paddle={r['paddle_pred']:3d}({r['paddle_conf']:.2f})  [{r['vote_method']}]")
    print()
