# phase_m6d_common_core.py
# Фаза M6d: контроль к Track A (M6).
#
# Проблема M6/Track A: парное согласие считалось на «уверенном ядре ПАРЫ»
# (документы с sim>=0.8 у обеих моделей). Размер этого ядра у разных пар
# различается втрое (18K у пар с Qwen3 против 46K у пар классических моделей),
# поэтому высокие ARI/низкие % переездов у пар с Qwen3 могли быть артефактом
# отбора «лёгких» документов, а не сходства моделей.
#
# Здесь согласие пересчитано на ОДНИХ И ТЕХ ЖЕ документах для всех пар:
#   full  — вся выборка (46 501), присвоение top-1 без порога;
#   core  — документы, уверенно присвоенные ВСЕМИ 7 моделями (sim>=0.8);
#   fixed_18k — контроль: ядро пары nomic_v15 x qwen3e_06b (самое узкое),
#               на нём считаются ВСЕ пары.
#
# Выход: reports/m6d_common_core.json
#
# Запуск: python phase_m6d_common_core.py

import glob
import io
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402

SIM_THRESHOLD = 0.8


def load_model_embeddings(model_key: str, df: pd.DataFrame):
    parts = sorted(glob.glob(str(EMB_DIR / f"{model_key}__shard*.npy")))
    parts = [p for p in parts if not p.endswith('_ids.npy')]
    if not parts:
        return None
    emb = np.concatenate([np.load(p) for p in parts])
    ids = np.concatenate([np.load(p.replace('.npy', '_ids.npy')) for p in parts])
    pos = {int(i): k for k, i in enumerate(ids)}
    order = [pos[int(i)] for i in df['id']]
    return emb[order]


def norm(embs):
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)


def assign_themes(embs, labels):
    embs = norm(embs)
    themes = sorted(np.unique(labels))
    cents = np.stack([norm(embs[labels == t].mean(axis=0, keepdims=True))[0]
                      for t in themes])
    sims = embs @ cents.T
    best = sims.argmax(axis=1)
    return np.array([themes[i] for i in best]), sims.max(axis=1)


def pair_stats(a1, a2, mask, labels_true):
    m = mask
    return {
        'n': int(m.sum()),
        'ari': float(adjusted_rand_score(a1[m], a2[m])),
        'moved_frac': float((a1[m] != a2[m]).mean()),
    }


def main():
    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels_true = df['theme'].to_numpy()
    keys = [k for k in MODELS]

    assigns, sims = {}, {}
    for k in keys:
        e = load_model_embeddings(k, df)
        if e is None:
            print(f"  {k}: нет эмбеддингов — пропуск")
            continue
        a, s = assign_themes(e, labels_true)
        assigns[k], sims[k] = a, s
        print(f"  {k}: coverage {float((s >= SIM_THRESHOLD).mean()):.1%}, "
              f"diff vs V3 {float((a != labels_true).mean()):.1%}")
    keys = list(assigns)

    n = len(df)
    full_mask = np.ones(n, dtype=bool)
    core_mask = np.ones(n, dtype=bool)
    for k in keys:
        core_mask &= sims[k] >= SIM_THRESHOLD
    narrow_mask = ((sims['nomic_v15'] >= SIM_THRESHOLD)
                   & (sims['qwen3e_06b'] >= SIM_THRESHOLD))
    print(f"\nвыборка {n}; ядро всех моделей {int(core_mask.sum())} "
          f"({core_mask.mean():.1%}); узкое ядро v15xqwen06b "
          f"{int(narrow_mask.sum())} ({narrow_mask.mean():.1%})")

    out = {'n_sample': n,
           'n_core_all': int(core_mask.sum()),
           'n_narrow': int(narrow_mask.sum()),
           'coverage': {k: float((sims[k] >= SIM_THRESHOLD).mean()) for k in keys},
           'diff_vs_v3': {k: float((assigns[k] != labels_true).mean()) for k in keys},
           'pairs': []}

    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            row = {'pair': f'{k1} vs {k2}'}
            for name, mask in (('full', full_mask), ('core_all', core_mask),
                               ('narrow_18k', narrow_mask)):
                row[name] = pair_stats(assigns[k1], assigns[k2], mask, labels_true)
            out['pairs'].append(row)
            print(f"  {k1} vs {k2}: full moved {row['full']['moved_frac']:.1%} "
                  f"(ARI {row['full']['ari']:.3f}) | core_all moved "
                  f"{row['core_all']['moved_frac']:.1%} (ARI {row['core_all']['ari']:.3f})"
                  f" | narrow moved {row['narrow_18k']['moved_frac']:.1%}")

    # согласие с эталонной разметкой V3 на тех же подвыборках
    out['diff_vs_v3_masks'] = {
        k: {'full': float((assigns[k] != labels_true).mean()),
            'core_all': float((assigns[k][core_mask] != labels_true[core_mask]).mean()),
            'narrow_18k': float((assigns[k][narrow_mask]
                                 != labels_true[narrow_mask]).mean())}
        for k in keys}

    REPORTS_DIR.mkdir(exist_ok=True)
    p = REPORTS_DIR / 'm6d_common_core.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nwritten {p}")


if __name__ == '__main__':
    main()
