# phase_o1c_finereader.py
# Пакетный прогон ABBYY FineReader 16 (десктоп) через Hot Folder
# на наборе docs_set.json (те же 200 pdf_text + 10 pdf_scan).
#
# Задача Hot Folder 'ocr_experiment' (тип Run once, создана UI-автоматизацией):
#   вход C:\Windows\Temp\opencode\fr_in → распознавание (Russian and English)
#   → выход C:\Windows\Temp\opencode\fr_out, Text (*.txt) UTF-8, [F].txt.
#
# Скрипт: чистит fr_in/fr_out, копирует рендеры (рендерит недостающие),
# нажимает Start Now, ждёт завершения, пишет results/o1_finereader.jsonl
# (слияние в общий o1_results.jsonl — merge_o1c.py).
#
# Запуск (venv_ocr): python phase_o1c_finereader.py

import io
import json
import shutil
import sys
import time
from pathlib import Path

import fitz

from oc_config import DPI, MIN_GT_CHARS, RENDER_DIR, RESULTS_DIR, RESULTS_JSONL

FR_IN = Path(r'C:\Windows\Temp\opencode\fr_in')
FR_OUT = Path(r'C:\Windows\Temp\opencode\fr_out')
OUT_JSONL = RESULTS_DIR / 'o1_finereader.jsonl'
DOCS_SET = RESULTS_DIR / 'docs_set.json'

POLL_SEC = 10          # опрос fr_out
STALL_SEC = 120        # нет новых файлов столько секунд → финиш
MAX_WAIT_SEC = 3600    # абсолютный таймаут прогона


def load_meta():
    """doc_id -> (gt_text, file_name, source) из общего jsonl."""
    meta = {}
    for l in io.open(RESULTS_JSONL, encoding='utf-8'):
        r = json.loads(l)
        meta[r['doc_id']] = (r['gt_text'], r['file_name'], r['source'])
    return meta


def main():
    docs = json.loads(io.open(DOCS_SET, encoding='utf-8').read())
    meta = load_meta()

    # 1. чистка
    for p in FR_IN.glob('*.png'):
        p.unlink()
    for p in FR_OUT.glob('*.txt'):
        p.unlink()
    print('folders cleaned', flush=True)

    # 2. копирование рендеров (рендерим недостающие)
    expected = {}
    for d in docs:
        did = d['doc_id']
        gt, fname, source = meta[did]
        if source == 'pdf_text' and len(gt) < MIN_GT_CHARS:
            continue
        png = RENDER_DIR / f'{source}_{did}.png'
        if not png.exists():
            doc = fitz.open(d['file_path'])
            try:
                pix = doc[0].get_pixmap(dpi=DPI)
                pix.save(str(png))
            finally:
                doc.close()
        shutil.copy2(png, FR_IN / png.name)
        expected[did] = (gt, fname, source, png.name)
    print(f'copied {len(expected)} renders', flush=True)

    # 3. Start Now
    from pywinauto import Application
    from pywinauto.findwindows import find_windows
    from pywinauto.mouse import click as mclick
    app = Application(backend='uia').connect(
        path=r'C:\Program Files\ABBYY FineReader 16\HotFolder.exe', timeout=30)
    main_win = None
    for h in find_windows(process=app.process):
        spec = app.window(handle=h)
        if 'Hot Folder' in spec.window_text():
            main_win = spec
            break
    assert main_win
    main_win.set_focus()
    time.sleep(0.7)
    mclick(button='left', coords=(400, 179))   # строка задачи
    time.sleep(1.0)
    mclick(button='left', coords=(308, 233))   # Start Now
    t0 = time.time()
    print('task started', flush=True)

    # 4. ожидание результатов
    last_n, stall = 0, 0
    while True:
        n = len(list(FR_OUT.glob('*.txt'))) - (
            1 if (FR_OUT / 'Hot Folder Log.txt').exists() else 0)
        if n >= len(expected):
            print(f'all {n} txt ready', flush=True)
            break
        if n == last_n:
            stall += POLL_SEC
        else:
            stall = 0
            last_n = n
            print(f'  {n}/{len(expected)} txt...', flush=True)
        if stall >= STALL_SEC:
            print(f'stall {STALL_SEC}s at {n}/{len(expected)} — finishing',
                  flush=True)
            break
        if time.time() - t0 > MAX_WAIT_SEC:
            print('MAX_WAIT reached', flush=True)
            break
        time.sleep(POLL_SEC)
    total_sec = time.time() - t0
    avg = total_sec / max(len(expected), 1)
    print(f'batch done in {total_sec:.0f}s (~{avg:.1f} s/file)', flush=True)

    # 5. сбор записей
    n_ok = 0
    with io.open(OUT_JSONL, 'w', encoding='utf-8') as out:
        for did, (gt, fname, source, pngname) in expected.items():
            txt_path = FR_OUT / (Path(pngname).stem + '.txt')
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8-sig').strip()
                o = {'text': text, 'sec': round(avg, 2)}
                n_ok += 1
            else:
                o = {'text': '', 'sec': round(avg, 2),
                     'error': 'no output txt'}
            rec = {'doc_id': did, 'source': source, 'file_name': fname,
                   'gt_text': gt, 'outputs': {'finereader': o}}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'written {n_ok}/{len(expected)} → {OUT_JSONL}', flush=True)


if __name__ == '__main__':
    main()
