# -*- coding: utf-8 -*-
# common.py -- «правильное» дообучение на TRAIN2 с ГЛУБОКОЙ чисткой меток.
#
# Отличие от ../train2_grouped/ (та версия убирала лишь 52 заведомо ошибочные
# метки, порог 11/13):
#   * канонический набор -- тот же, что у эксперимента с эмбеддингами:
#     D:\FileOrganizer\TRAIN2\exp_embeddings_13\ (texts_cache.csv 2279 док.,
#     preds_all.json -- пофайловые предсказания 13 вариантов, dup_groups.csv).
#     Это делает точность «на независимых» напрямую сравнимой с
#     train2_grouped_loo.json и снимает оговорку «набор не тот же».
#   * чистка: если >=9/13 вариантов LOO единогласно относят документ к чужому
#     типу И это подтверждает второй независимый сигнал (ключевые слова в
#     имени файла ИЛИ zero-shot LLM), документ ПЕРЕМЕЧАЕТСЯ в этот тип;
#     если второго сигнала нет -- документ УДАЛЯЕТСЯ из набора.
#   * group_size пересчитывается после чистки (удаление близнеца может
#     сделать группу синглтоном).
#
# База модели и рецепт LoRA не меняются (Qwen3.5-4B, r=16, 1 эпоха).

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'train2_full'                 # только за LABELS/PROMPT
EXP13 = Path(r'D:\FileOrganizer\TRAIN2\exp_embeddings_13')
TEXTS_CACHE = EXP13 / 'texts_cache.csv'
PREDS_ALL = EXP13 / 'preds_all.json'
DUP_GROUPS = EXP13 / 'dup_groups.csv'

MODEL_PATH = r'D:\MODELS\Transformers\Qwen_Qwen3.5-4B'

SEEDS = [42, 123, 2024]
TEST_FRAC = 0.22
GROUP_CAP = 40
ENSEMBLE_THR = 9                                  # >=9/13 согласны на чужой класс
MAX_TEXT_CHARS_EVAL = 2500
MAX_TEXT_CHARS_TRAIN = 900
MAX_SEQ_LEN = 700

POOL_PATH = BASE / 'pool.jsonl'
DUP_CSV = BASE / 'dup_groups_clean.csv'
DECISIONS_CSV = BASE / 'cleaning_decisions.csv'
SPLIT_CSV = BASE / 'split_seed{seed}.csv'
ZS_JSONL = BASE / 'zeroshot_seed{seed}.jsonl'
FT_JSONL = BASE / 'finetuned_seed{seed}.jsonl'
ADAPTER_DIR = BASE / 'lora_seed{seed}'
RESULTS_JSON = BASE / 'results.json'
PROGRESS_LOG = BASE / 'progress.log'

sys.path.insert(0, str(OLD))
from run_zeroshot import LABELS, PROMPT, extract_json  # noqa: E402,F401


def load_pool():
    return [json.loads(l) for l in open(POOL_PATH, encoding='utf-8')]


def load_split(seed):
    import csv as _csv
    pool = {d['entity_id']: d for d in load_pool()}
    train, test = [], []
    for row in _csv.DictReader(open(str(SPLIT_CSV).format(seed=seed), encoding='utf-8')):
        d = pool[row['entity_id']]
        rec = {'entity_id': row['entity_id'], 'type': d['type'], 'text': d['text'],
               'group_size': int(row['group_size'])}
        (train if row['split'] == 'train' else test).append(rec)
    return train, test


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
        f.write(f'{msg}\n')
