import json
import shutil
from pathlib import Path

with open(r'D:\FileOrganizer\TRAIN\models\ensemble_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = [r for r in data['results'] if not r['ensemble_correct']]
src_dir = Path(r'D:\FileOrganizer\Extracted\PDF_Images')
dst_dir = Path(r'D:\FileOrganizer\Extracted\Ensemble_Errors')

for r in errors:
    src = src_dir / r['file']
    dst = dst_dir / r['file']
    shutil.copy2(src, dst)
    print(f'Copied: {r["file"][:60]}...')

print(f'\nTotal: {len(errors)} files copied to {dst_dir}')
