import json

with open(r'D:\FileOrganizer\TRAIN\models\qwen_new_prompt_test.json', encoding='utf-8') as f:
    data = json.load(f)

# Фильтруем файлы с истинной ориентацией 90°
files_90 = [r for r in data['results'] if r['true_angle'] == 90]

print(f'Файлы с истинной ориентацией 90° ({len(files_90)} шт):\n')
for i, r in enumerate(files_90, 1):
    print(f'{i}. {r["file"]}')
