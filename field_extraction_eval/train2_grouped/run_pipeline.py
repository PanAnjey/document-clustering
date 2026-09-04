# -*- coding: utf-8 -*-
# run_pipeline.py [seed ...]
# Оркестратор «правильного» дообучения: для каждого сида по очереди запускает
# zeroshot -> train -> eval, каждый этап -- свежий процесс (чистый GPU).
# Идемпотентно: завершённый этап пропускается. Прогресс -- в progress.log.
#
# Долгий (~2-2.5 ч на 3 сида). Запуск в фоне.

import subprocess
import sys
import time
from pathlib import Path

from common import (ADAPTER_DIR, BASE, FT_JSONL, SEEDS, SPLIT_CSV, ZS_JSONL, log)

PY = sys.executable


def _test_n(seed):
    import csv
    path = str(SPLIT_CSV).format(seed=seed)
    return sum(1 for r in csv.DictReader(open(path, encoding='utf-8')) if r['split'] == 'test')


def _lines(path):
    try:
        return sum(1 for _ in open(path, encoding='utf-8'))
    except FileNotFoundError:
        return 0


def done_zeroshot(seed):
    return _lines(str(ZS_JSONL).format(seed=seed)) >= _test_n(seed)


def done_train(seed):
    return (Path(str(ADAPTER_DIR).format(seed=seed)) / 'final' / 'adapter_model.safetensors').exists()


def done_eval(seed):
    return _lines(str(FT_JSONL).format(seed=seed)) >= _test_n(seed)


def run(stage, seed):
    log(f'>>> {stage} seed={seed} :: старт {time.strftime("%H:%M:%S")}')
    t0 = time.time()
    r = subprocess.run([PY, '-X', 'utf8', '-u', str(BASE / f'stage_{stage}.py'), str(seed)],
                       cwd=str(BASE))
    dt = time.time() - t0
    if r.returncode != 0:
        log(f'!!! {stage} seed={seed} :: код {r.returncode} за {dt / 60:.1f} мин -- СТОП')
        sys.exit(r.returncode)
    log(f'<<< {stage} seed={seed} :: готово за {dt / 60:.1f} мин')


def main():
    seeds = [int(x) for x in sys.argv[1:]] or SEEDS
    log(f'=== pipeline старт, сиды {seeds}, python {PY} ===')
    for seed in seeds:
        for stage, is_done in (('zeroshot', done_zeroshot), ('train', done_train), ('eval', done_eval)):
            if is_done(seed):
                log(f'--- {stage} seed={seed} :: уже готово, пропуск')
                continue
            run(stage, seed)
    log('=== pipeline ВЕСЬ ГОТОВ ===')


if __name__ == '__main__':
    main()
