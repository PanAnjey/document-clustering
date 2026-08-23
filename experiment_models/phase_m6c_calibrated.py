# phase_m6c_calibrated.py
# Фаза M6c: калиброванный unknown — цель пользователя «максимально плотные
# кластеры с минимальным unknown».
#
# Продакшен-порог 0.8 несовместим между моделями (M6). Здесь для КАЖДОЙ модели
# строится кривая «порог → unknown% / точность присвоения» (LOO-центроиды,
# метки V3) и находятся рабочие точки:
#   - unknown rate при точности присвоения >= 95% (минимальный порог);
#   - unknown rate при точности присвоения >= 90%;
#   - максимальная достижимая точность (порог 0.95).
#
# Выход: reports/m6c_calibrated.json

import glob
import io
import json

import numpy as np
import pandas as pd

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402

ACC_TARGETS = (0.95, 0.90)


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


def loo_assign(embs, labels):
    """LOO-центроиды: (assigned_theme, own_sim). Точный leave-one-out."""
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    themes = sorted(np.unique(labels))
    t2col = {t: k for k, t in enumerate(themes)}
    sums = {t: embs[labels == t].sum(axis=0) for t in themes}
    counts = {t: int((labels == t).sum()) for t in themes}
    cents = np.zeros_like(embs)
    for i, t in enumerate(labels):
        n = counts[t]
        c = (sums[t] - embs[i]) / max(n - 1, 1)
        cents[i] = c / (np.linalg.norm(c) + 1e-12)
    own = np.einsum('ij,ij->i', embs, cents)  # sim к LOO-центроиду своей темы
    # чужие центроиды (обычные, не LOO — вклад документа пренебрежим)
    theme_cents = np.stack([sums[t] / (np.linalg.norm(sums[t]) + 1e-12) for t in themes])
    sims = embs @ theme_cents.T
    sims[np.arange(len(labels)), [t2col[t] for t in labels]] = -1
    best_other = sims.max(axis=1)
    correct = own > best_other  # LOO top-1 hit
    return own, correct


def main():
    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels = df['theme'].to_numpy()
    print(f"sample: {len(df)}")

    thresholds = np.arange(0.30, 0.9901, 0.005)
    out = {}
    for key in MODELS:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key}: нет эмбеддингов — пропуск")
            continue
        own, correct = loo_assign(embs, labels)
        # кривая: для каждого порога — unknown% и accuracy среди присвоенных
        curve = []
        for t in thresholds:
            acc_mask = own >= t
            if acc_mask.sum() < 50:
                continue
            curve.append({
                't': float(t),
                'unknown': float(1 - acc_mask.mean()),
                'acc': float(correct[acc_mask].mean()),
            })
        res = {'curve': curve}
        for target in ACC_TARGETS:
            ok = [c for c in curve if c['acc'] >= target]
            # минимальный unknown при заданной точности = первая точка,
            # где accuracy достигает target (unknown растёт с порогом)
            res[f'unknown_at_acc{int(target*100)}'] = (
                min((c['unknown'] for c in ok), default=None))
            res[f't_at_acc{int(target*100)}'] = (
                min((c['t'] for c in ok
                     if c['unknown'] == res[f'unknown_at_acc{int(target*100)}']),
                    default=None))
        res['acc_at_t080'] = (
            float(correct[own >= 0.8].mean()) if (own >= 0.8).sum() > 50 else None)
        res['max_acc'] = max(c['acc'] for c in curve)
        out[key] = res
        u95 = res['unknown_at_acc95']
        t95 = res['t_at_acc95']
        print(f"  {key}: unknown@acc95 "
              f"{f'{u95:.1%} (t={t95:.3f})' if u95 is not None else 'недостижимо'}, "
              f"unknown@acc90 "
              f"{f'{res[chr(39)+chr(39)]}' if False else ''}"
              f"{'' if res['unknown_at_acc90'] is None else format(res['unknown_at_acc90'], '.1%')}, "
              f"max_acc {res['max_acc']:.1%}")

    (REPORTS_DIR / 'm6c_calibrated.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nwritten {REPORTS_DIR / 'm6c_calibrated.json'}")


if __name__ == '__main__':
    main()
