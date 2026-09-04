# -*- coding: utf-8 -*-
# rescore_embeddings.py
# Пересчёт grouped-LOO точности 13 вариантов эмбеддингов НА ТЕХ ЖЕ данных,
# что видело дообучение: набор 2234 (после чистки), с перемеченными и без
# удалённых. Иначе сравнение «дообучение 93.7% против векторов 85-88%»
# было бы нечестным -- у векторов числа считались на грязных метках.
#
# Считаем два варианта на одном и том же наборе документов:
#   dirty -- исходные метки папок;  clean -- метки после чистки.
#
#   python rescore_embeddings.py

import csv
import json

import numpy as np

from common import DUP_GROUPS, EXP13, LABELS, POOL_PATH, TEXTS_CACHE

VEC = EXP13 / 'vectors'
l2 = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def loo_grouped(e, y, g, K):
    sums = np.zeros((K, e.shape[1])); cnt = np.zeros(K)
    for i, c in enumerate(y):
        sums[c] += e[i]; cnt[c] += 1
    ok = np.zeros(len(y), bool)
    order = np.argsort(g, kind='stable')
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    for idx in np.split(order, bounds):
        gs = np.zeros((K, e.shape[1])); gc = np.zeros(K)
        for i in idx:
            gs[y[i]] += e[i]; gc[y[i]] += 1
        rest = sums - gs; rcnt = cnt - gc
        C = rest / (np.linalg.norm(rest, axis=1, keepdims=True) + 1e-12)
        S = e[idx] @ C.T
        S[:, rcnt < 1] = -2.0
        p = S.argmax(1)
        for j, i in enumerate(idx):
            ok[i] = p[j] == y[i]
    return ok


def main():
    rows = list(csv.DictReader(open(TEXTS_CACHE, encoding='utf-8')))
    dirty_all = [r['cls'] for r in rows]                       # длина 2279
    dg_rows = list(csv.DictReader(open(DUP_GROUPS, encoding='utf-8')))
    gid_all = np.array([int(r['group_id']) for r in dg_rows])

    pool = [json.loads(l) for l in open(POOL_PATH, encoding='utf-8')]
    keep_idx = np.array(sorted(int(p['entity_id']) for p in pool))
    clean_lab = {int(p['entity_id']): p['type'] for p in pool}

    y_dirty_s = [dirty_all[i] for i in keep_idx]
    y_clean_s = [clean_lab[i] for i in keep_idx]
    g = gid_all[keep_idx]
    # size групп на удержанном наборе -> singleton-статус
    _, inv, cnts = np.unique(g, return_inverse=True, return_counts=True)
    single = cnts[inv] == 1
    print(f'{len(keep_idx)} документов, групп {len(set(g))}, '
          f'независимых {int(single.sum())} ({single.mean():.0%})\n')

    variants = ['giga_480m', 'qwen3e_4b', 'nomic_v2', 'e5_large', 'bge_m3',
                'char_tfidf_svd_512', 'giga_3b', 'qwen3e_06b', 'nomic_v15',
                'char_raw_4096', 'rubert_base_dp', 'sbert_ru', 'rubert_tiny2']

    out = []
    for tag, ys in (('dirty', y_dirty_s), ('clean', y_clean_s)):
        cls = sorted(set(ys)); c2i = {c: i for i, c in enumerate(cls)}
        y = np.array([c2i[c] for c in ys]); K = len(cls)
        print(f'--- метки: {tag} ({K} классов) ---')
        print(f'{"вариант":20s}  общая  независ.  в группах')
        for v in variants:
            e = l2(np.load(VEC / f'{v}.npy').astype(np.float32)[keep_idx])
            ok = loo_grouped(e, y, g, K)
            r = {'labels': tag, 'variant': v,
                 'acc_grouped': float(ok.mean()),
                 'acc_singletons': float(ok[single].mean()),
                 'acc_inGroups': float(ok[~single].mean())}
            out.append(r)
            print(f'{v:20s}  {r["acc_grouped"]:.3f}   {r["acc_singletons"]:.3f}    '
                  f'{r["acc_inGroups"]:.3f}')
        print()

    json.dump(out, open('reports/embeddings_rescored.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('-> reports/embeddings_rescored.json')


if __name__ == '__main__':
    main()
