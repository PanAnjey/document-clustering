# -*- coding: utf-8 -*-
# prep_grouped.py -- шаг 1: пул текстов, чистка меток, группы дублей, 3 сплита.
# Быстрый (~1-2 мин), запускать интерактивно перед долгим run_pipeline.py.
#
#   python prep_grouped.py

import csv
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer

from common import (DUP_CSV, DUP_THRESHOLD, LABELS, MISLABELED_CSV, OLD, POOL_PATH,
                    SEEDS, SPLIT_CSV, TEST_FRAC)

# Группы крупнее CAP -- это мегашаблоны (напр. 301 почти одинаковая претензия,
# 148 уведомлений). В групповом сплите они не могут попасть в test осмысленно:
# либо в train нет класса, либо test = 300 копий одного бланка. Держим их в
# train всегда; per-class n в test показывает, где выборка мала.
GROUP_CAP = 40

DIGITS = re.compile(r'\d+')
WS = re.compile(r'\s+')


def norm(t):
    t = WS.sub(' ', str(t).lower().strip())
    return DIGITS.sub('0', t)[:8000]


def build_pool():
    pool = {}
    for name in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
        for line in open(OLD / name, encoding='utf-8'):
            d = json.loads(line)
            pool[d['entity_id']] = d
    pool = {k: v for k, v in pool.items() if v['type'] in LABELS}

    mis_ids = set()
    for row in csv.DictReader(open(MISLABELED_CSV, encoding='utf-8')):
        mis_ids.add(row['file'].split('.', 1)[0])
    dropped = [e for e in pool if e in mis_ids]
    for e in dropped:
        del pool[e]
    print(f'пул: {len(pool)} документов после удаления {len(dropped)} '
          f'заведомо ошибочных меток (из {len(mis_ids)} в списке)')

    docs = sorted(pool.values(), key=lambda d: d['entity_id'])
    with open(POOL_PATH, 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps({
                'entity_id': d['entity_id'], 'type': d['type'],
                'file_name': d['file_name'], 'text': d['text'],
            }, ensure_ascii=False) + '\n')
    return docs


def dup_groups(docs):
    texts = [norm(d['text']) for d in docs]
    n = len(texts)
    vec = HashingVectorizer(analyzer='char_wb', ngram_range=(5, 5),
                            n_features=2 ** 18, binary=True, norm='l2',
                            alternate_sign=False, lowercase=False)
    X = vec.transform(texts).astype(np.float32)
    print(f'{n} документов, бинарные 5-граммы, ненулевых/док {X.getnnz(axis=1).mean():.0f}')

    print('чувствительность к порогу:')
    keep = None
    for t in (0.70, 0.80, 0.90, 0.95):
        rows, cols = [], []
        for s in range(0, n, 500):
            S = (X[s:s + 500] @ X.T).toarray()
            S[np.arange(len(S)), s + np.arange(len(S))] = 0
            r, c = np.nonzero(S >= t)
            rows.extend(r + s)
            cols.extend(c)
        g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        ncomp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        multi_docs = int(sizes[sizes > 1].sum())
        print(f'  порог {t:.2f}: групп {ncomp}, док. в мультигруппах {multi_docs} '
              f'({multi_docs / n:.0%}), крупнейшая {sizes.max()}')
        if abs(t - DUP_THRESHOLD) < 1e-9:
            keep = (lab, sizes)

    lab, sizes = keep
    # группы, накрывающие >1 типа папки -- ещё один сигнал грязной разметки
    span = 0
    types = np.array([d['type'] for d in docs])
    for gid in np.where(sizes > 1)[0]:
        if len(set(types[lab == gid])) > 1:
            span += 1
    print(f'порог {DUP_THRESHOLD}: {int(len(sizes))} групп, '
          f'{int((sizes > 1).sum())} мультигрупп, из них {span} накрывают >1 типа')

    out = pd.DataFrame({
        'entity_id': [d['entity_id'] for d in docs],
        'type': types,
        'group_id': lab,
        'group_size': sizes[lab],
    })
    out.to_csv(DUP_CSV, index=False, encoding='utf-8')
    print(f'-> {DUP_CSV}')
    return out


def make_split(dg, seed):
    """Групповой сплит со стратификацией по типу папки. Ни одна группа дублей
    не разрывается train/test. Три раздельных бюджета на тип: синглтоны,
    малые группы (2..CAP), мегагруппы (>CAP -> всегда train) -- иначе большие
    шаблоны вытесняют синглтоны из test и метрика «на независимых» пустеет."""
    rng = np.random.default_rng(seed)
    split = {}                       # group_id -> 'train'|'test'
    for typ, sub in dg.groupby('type'):
        gsz = sub.groupby('group_id').size().to_dict()
        singles = [g for g, s in gsz.items() if s == 1]
        smalls = [g for g, s in gsz.items() if 2 <= s <= GROUP_CAP]
        larges = [g for g, s in gsz.items() if s > GROUP_CAP]

        for g in larges:
            split[g] = 'train'

        rng.shuffle(singles)
        k = round(len(singles) * TEST_FRAC)
        for g in singles[:k]:
            split[g] = 'test'
        for g in singles[k:]:
            split[g] = 'train'

        rng.shuffle(smalls)
        target = sum(gsz[g] for g in smalls) * TEST_FRAC
        acc = 0
        for g in smalls:
            if acc < target:
                split[g] = 'test'
                acc += gsz[g]
            else:
                split[g] = 'train'

        allg = singles + smalls + larges
        if len(allg) >= 2:
            if all(split[g] == 'train' for g in allg):
                split[min(allg, key=lambda x: gsz[x])] = 'test'
            elif all(split[g] == 'test' for g in allg):
                split[min(allg, key=lambda x: gsz[x])] = 'train'

    dg = dg.copy()
    dg['split'] = dg['group_id'].map(split)
    torn = int((dg.groupby('group_id')['split'].nunique() > 1).sum())
    assert torn == 0, f'{torn} групп разорвано'

    path = str(SPLIT_CSV).format(seed=seed)
    dg.to_csv(path, index=False, encoding='utf-8')

    tr, te = (dg['split'] == 'train').sum(), (dg['split'] == 'test').sum()
    te_df = dg[dg['split'] == 'test']
    sing = int((te_df['group_size'] == 1).sum())
    ing = int((te_df['group_size'] > 1).sum())
    print(f'  сид {seed}: train {tr}  test {te}  '
          f'(независимых {sing}, с близнецами {ing})')
    per = te_df['type'].value_counts().to_dict()
    print('        test по типам: ' + ', '.join(
        f'{t}:{per.get(t, 0)}' for t in sorted(dg["type"].unique())))
    return dg


def main():
    docs = build_pool()
    dg = dup_groups(docs)
    print('\nгрупповые сплиты:')
    for seed in SEEDS:
        make_split(dg, seed)
    print('\nprep готов. дальше: run_pipeline.py (долгий, детач).')


if __name__ == '__main__':
    main()
