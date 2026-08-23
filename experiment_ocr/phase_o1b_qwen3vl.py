# phase_o1b_qwen3vl.py
# Фаза O1b: шардированный прогон Qwen3-VL-8B на наборе docs_set.json
# (медленный движок ~45-70 с/стр → 2 GPU).
#
# Шардирование взвешенное (cuda:1 — дисплейная, примерно вдвое медленнее):
#   shard 0 (cuda:0) — 60% документов (i % 10 < 6)
#   shard 1 (cuda:1) — 40% документов (i % 10 >= 6)
# Каждый процесс пишет СВОЙ файл results/o1_qwen3vl_shard{N}.jsonl
# (во избежание конкурентной записи в общий o1_results.jsonl),
# слияние — после завершения обоих.
#
# Запуск (venv_ocr):
#   python phase_o1b_qwen3vl.py --device cuda:0 --shard-id 0
#   python phase_o1b_qwen3vl.py --device cuda:1 --shard-id 1

import argparse
import io
import json
import time
from pathlib import Path

import fitz

from oc_config import (  # noqa: E402
    DPI, MIN_GT_CHARS, RENDER_DIR, RESULTS_DIR, RESULTS_JSONL,
)

DOCS_SET = RESULTS_DIR / 'docs_set.json'


def load_done(*paths):
    done = set()
    for p in paths:
        if not p.exists():
            continue
        with io.open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for eng in rec.get('outputs', {}):
                        done.add((rec['doc_id'], eng))
                except Exception:  # noqa: BLE001
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard-id', type=int, required=True, choices=[0, 1])
    ap.add_argument('--engine', default='qwen3vl',
                    choices=['qwen3vl', 'qwen3vl4b', 'gemma4'])
    args = ap.parse_args()
    ENG = args.engine

    docs = json.loads(io.open(DOCS_SET, encoding='utf-8').read())
    # 60/40: shard0 берёт 6 из каждых 10 подряд идущих позиций
    mine = [d for i, d in enumerate(docs)
            if (i % 10 < 6) == (args.shard_id == 0)]
    shard_file = RESULTS_DIR / f'o1_{ENG}_shard{args.shard_id}.jsonl'
    done = load_done(RESULTS_JSONL, shard_file)
    print(f'shard{args.shard_id} ({args.device}, {ENG}): {len(mine)} документов, '
          f'уже сделано {sum((d["doc_id"], ENG) in done for d in mine)}',
          flush=True)

    if ENG == 'qwen3vl':
        from engines.oc_qwen3vl import Qwen3VLEngine
        eng = Qwen3VLEngine(device=args.device)
    elif ENG == 'qwen3vl4b':
        from engines.oc_qwen3vl4b import get_engine
        eng = get_engine(device=args.device)
    else:
        from engines.oc_gemma4 import get_engine
        eng = get_engine(device=args.device)
    print('engine ready', flush=True)

    n_new = 0
    with io.open(shard_file, 'a', encoding='utf-8') as out:
        for k, d in enumerate(mine):
            if (d['doc_id'], ENG) in done:
                continue
            png = RENDER_DIR / f"{d['source']}_{d['doc_id']}.png"
            try:
                doc = fitz.open(d['file_path'])
                try:
                    page = doc[0]
                    gt = page.get_text().strip()
                    if not png.exists():
                        pix = page.get_pixmap(dpi=DPI)
                        pix.save(str(png))
                finally:
                    doc.close()
            except Exception as e:  # noqa: BLE001
                print(f'  doc {d["doc_id"]}: render failed: {e}', flush=True)
                continue
            if d['source'] == 'pdf_text' and len(gt) < MIN_GT_CHARS:
                continue
            t0 = time.time()
            try:
                text = eng.ocr(str(png))
                o = {'text': text, 'sec': round(time.time() - t0, 2)}
            except Exception as e:  # noqa: BLE001
                o = {'text': '', 'sec': round(time.time() - t0, 2),
                     'error': f'{type(e).__name__}: {e}'}
            rec = {'doc_id': d['doc_id'], 'source': d['source'],
                   'file_name': Path(d['file_path']).name,
                   'gt_text': gt, 'outputs': {ENG: o}}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            n_new += 1
            if n_new % 5 == 0 or n_new == 1:
                print(f'  [{k + 1}/{len(mine)}] new={n_new} '
                      f'last={o["sec"]}s', flush=True)
    print(f'shard{args.shard_id} done. новых: {n_new} → {shard_file}',
          flush=True)


if __name__ == '__main__':
    main()
