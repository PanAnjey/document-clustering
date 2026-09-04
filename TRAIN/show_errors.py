import json

with open(r'D:\FileOrganizer\TRAIN\models\ensemble_report.json', encoding='utf-8') as f:
    data = json.load(f)

errors = [r for r in data['results'] if not r['ensemble_correct']]
print(f'Всего ошибок: {len(errors)}\n')

for r in errors:
    fname = r['file'][:70]
    print(f'{fname:70s} true={r["true_angle"]:3d}  vit={r["vit_pred"]:3d}  eff={r["effnet_pred"]:3d}  tess={r["tess_pred"]:3d}  ens={r["ensemble_pred"]:3d}')
