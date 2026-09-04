import json

with open(r'D:\FileOrganizer\TRAIN\models\ensemble_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = [r for r in data['results'] if not r['ensemble_correct']]
base = r'D:\FileOrganizer\Extracted\PDF_Images'

for r in sorted(errors, key=lambda x: x['true_angle']):
    print(f'{base}\\{r["file"]}')
