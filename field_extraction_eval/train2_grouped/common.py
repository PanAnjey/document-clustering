# -*- coding: utf-8 -*-
# common.py -- общие константы и утилиты «правильного» дообучения на TRAIN2.
#
# Что чинится относительно field_extraction_eval/train2_full/:
#   1. Групповой train/test-сплит по near-дубликатам шаблона: все близнецы
#      одного шаблона (ежемесячные счета/акты одного контрагента) остаются на
#      ОДНОЙ стороне. Без этого дообучение меряет «узнавание копии».
#   2. Из train и test выкинуты 52 документа с заведомо ошибочной меткой папки
#      (FOR TEST EMBEDINGS/reports/train2_suspected_mislabeled.csv: >=11 из 13
#      вариантов эмбеддингов единогласно спорят с папкой).
#   3. Точность считается в разбивке: общая / на независимых документах
#      (singletons) / на документах с близнецами (in-group) -- как в таблице
#      grouped-LOO для эмбеддингов (FOR TEST EMBEDINGS/reports/train2_grouped_loo.json).
#   4. Три сида разбиения, итог -- среднее и разброс.
#
# База модели и рецепт LoRA не меняются относительно train2_full (Qwen3.5-4B,
# r=16, alpha=32, 1 эпоха) -- чтобы «до/после» и сравнение со старым числом
# были чистыми.

import json
import sys
from pathlib import Path

# PowerShell 5.1: stdout в cp1251 -> кириллица и символы вроде ≈ ломают print.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'train2_full'
PROJECT_ROOT = BASE.parents[1]

MODEL_PATH = r'D:\MODELS\Transformers\Qwen_Qwen3.5-4B'

MISLABELED_CSV = PROJECT_ROOT / 'FOR TEST EMBEDINGS' / 'reports' / 'train2_suspected_mislabeled.csv'

SEEDS = [42, 123, 2024]
TEST_FRAC = 0.22
DUP_THRESHOLD = 0.80          # тот же порог, что в dedup_groups.py для DIADOC
MAX_TEXT_CHARS_EVAL = 2500    # как в train2_full/run_zeroshot_4b.py и run_finetuned_eval.py
MAX_TEXT_CHARS_TRAIN = 900    # как в train2_full/train_lora.py
MAX_SEQ_LEN = 700

POOL_PATH = BASE / 'pool.jsonl'
DUP_CSV = BASE / 'dup_groups.csv'
SPLIT_CSV = BASE / 'split_seed{seed}.csv'
ZS_JSONL = BASE / 'zeroshot_seed{seed}.jsonl'
FT_JSONL = BASE / 'finetuned_seed{seed}.jsonl'
ADAPTER_DIR = BASE / 'lora_seed{seed}'
RESULTS_JSON = BASE / 'results.json'
PROGRESS_LOG = BASE / 'progress.log'

# Промпт и список классов -- ИМПОРТ из старого эксперимента, чтобы zero-shot
# «до» был буквально тем же, что и раньше (тот же текст подсказки).
sys.path.insert(0, str(OLD))
from run_zeroshot import LABELS, PROMPT, extract_json  # noqa: E402,F401


def load_pool():
    return [json.loads(l) for l in open(POOL_PATH, encoding='utf-8')]


def load_split(seed):
    """-> (train_docs, test_docs); каждый док: entity_id, type, text, group_size."""
    import csv as _csv
    pool = {d['entity_id']: d for d in load_pool()}
    train, test = [], []
    path = str(SPLIT_CSV).format(seed=seed)
    for row in _csv.DictReader(open(path, encoding='utf-8')):
        d = pool[row['entity_id']]
        rec = {'entity_id': row['entity_id'], 'type': d['type'], 'text': d['text'],
               'group_size': int(row['group_size'])}
        (train if row['split'] == 'train' else test).append(rec)
    return train, test


def log(msg):
    line = f'{msg}'
    print(line, flush=True)
    with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
