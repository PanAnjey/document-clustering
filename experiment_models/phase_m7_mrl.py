# phase_m7_mrl.py
# Фаза M7: влияние MRL-усечения размерности Qwen3-Embedding-4B на качество.
#
# Вопрос: какая потеря качества при усечении вектора 2560 -> D и выгодно ли
# это для хранилища/поиска. Инференс повторно не нужен: эмбеддинги уже есть
# (embeddings/qwen3e_4b__shard*.npy), MRL = срез первых D координат + L2-norm.
#
# Протокол = M4/M4b/M6c на той же выборке 46 501 (метки V3):
#   LOO top-1/top-3, own-sim, margin (точный leave-one-out, = phase_m4_eval),
#   kNN purity@10 (подвыборка 15K, rng 42, = phase_m4b),
#   MiniBatchKMeans ARI/NMI (k=25, random_state 42, = phase_m4b),
#   кривая «порог -> unknown/acc» -> unknown@acc90/95 (= phase_m6c),
#   silhouette (косинус, подвыборка 5K, = phase_m4).
#
# Выход: reports/m7_mrl.json.
#
# Запуск: python phase_m7_mrl.py [--dims 2560 1024 512 256] [--model qwen3e_4b]

import argparse
import glob
import io
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from em_config import EMB_DIR, REPORTS_DIR, SAMPLE_CSV  # noqa: E402

KNN_SUB = 15000
KNN_K = 10
KMEANS_N_INIT = 3
ACC_TARGETS = (0.95, 0.90)


def load_embeddings(model_key: str, df: pd.DataFrame):
    parts = sorted(glob.glob(str(EMB_DIR / f"{model_key}__shard*.npy")))
    parts = [p for p in parts if not p.endswith('_ids.npy')]
    emb = np.concatenate([np.load(p) for p in parts])
    ids = np.concatenate([np.load(p.replace('.npy', '_ids.npy')) for p in parts])
    pos = {int(i): k for k, i in enumerate(ids)}
    order = [pos[int(i)] for i in df['id']]
    return emb[order]


def norm(embs):
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)


def loo_own_correct(embs, labels):
    """(own_sim LOO, correct LOO) — = phase_m6c.loo_assign."""
    embs = norm(embs)
    themes = sorted(np.unique(labels))
    t2col = {t: k for k, t in enumerate(themes)}
    sums = {t: embs[labels == t].sum(axis=0) for t in themes}
    counts = {t: int((labels == t).sum()) for t in themes}
    cents = np.zeros_like(embs)
    for i, t in enumerate(labels):
        n = counts[t]
        c = (sums[t] - embs[i]) / max(n - 1, 1)
        cents[i] = c / (np.linalg.norm(c) + 1e-12)
    own = np.einsum('ij,ij->i', embs, cents)
    theme_cents = np.stack([sums[t] / (np.linalg.norm(sums[t]) + 1e-12)
                            for t in themes])
    sims = embs @ theme_cents.T
    sims[np.arange(len(labels)), [t2col[t] for t in labels]] = -1
    best_other = sims.max(axis=1)
    correct = own > best_other
    return own, correct, theme_cents


def loo_metrics(embs, labels):
    """LOO top-1/top-3, own-sim, margin — = phase_m4_eval.loo_metrics."""
    embs = norm(embs)
    themes = np.unique(labels)
    sums = {t: embs[labels == t].sum(axis=0) for t in themes}
    counts = {t: int((labels == t).sum()) for t in themes}
    theme_cents = {t: sums[t] / (np.linalg.norm(sums[t]) + 1e-12) for t in themes}
    theme_list = sorted(themes)
    C = np.stack([theme_cents[t] for t in theme_list])
    sims_all = embs @ C.T
    t2col = {t: k for k, t in enumerate(theme_list)}
    top1 = top3 = 0
    own_sims, margins = [], []
    for i, t in enumerate(labels):
        n = counts[t]
        if n < 2:
            continue
        c = (sums[t] - embs[i]) / (n - 1)
        c = c / (np.linalg.norm(c) + 1e-12)
        own = float(embs[i] @ c)
        others = sims_all[i].copy()
        others[t2col[t]] = -1
        best_other = float(others.max())
        own_sims.append(own)
        margins.append(own - best_other)
        top1 += own > best_other
        combined = others.copy()
        combined[t2col[t]] = own
        top3 += t2col[t] in np.argsort(combined)[-3:]
    n = len(own_sims)
    return (top1 / n, top3 / n, float(np.mean(own_sims)), float(np.mean(margins)))


def knn_purity(embs, labels, rng):
    n = len(embs)
    idx = rng.choice(n, size=min(KNN_SUB, n), replace=False)
    sub_e, sub_l = embs[idx], labels[idx]
    purities = []
    for start in range(0, len(idx), 2000):
        chunk = sub_e[start:start + 2000]
        sims = chunk @ sub_e.T
        rows = np.arange(len(chunk))
        sims[rows, start + rows] = -1
        nn = np.argpartition(-sims, KNN_K, axis=1)[:, :KNN_K]
        same = (sub_l[nn] == sub_l[start:start + 2000][:, None]).mean(axis=1)
        purities.append(same)
    return float(np.concatenate(purities).mean())


def kmeans_agreement(embs, labels):
    k = len(np.unique(labels))
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192,
                         n_init=KMEANS_N_INIT, random_state=42)
    pred = km.fit_predict(embs)
    return (float(adjusted_rand_score(labels, pred)),
            float(normalized_mutual_info_score(labels, pred)))


def threshold_curve(own, correct):
    """Кривая «порог -> unknown/acc» (= phase_m6c) на LOO own-sim."""
    thresholds = np.arange(0.30, 0.9901, 0.005)
    curve = []
    for t in thresholds:
        m = own >= t
        if m.sum() < 50:
            continue
        curve.append({'t': float(t), 'unknown': float(1 - m.mean()),
                      'acc': float(correct[m].mean())})
    res = {}
    for target in ACC_TARGETS:
        ok = [c for c in curve if c['acc'] >= target]
        res[f'unknown_at_acc{int(target * 100)}'] = (
            min((c['unknown'] for c in ok), default=None))
        res[f't_at_acc{int(target * 100)}'] = (
            min((c['t'] for c in ok
                 if c['unknown'] == res[f'unknown_at_acc{int(target * 100)}']),
                default=None))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3e_4b')
    ap.add_argument('--dims', type=int, nargs='+',
                    default=[2560, 1024, 512, 256])
    args = ap.parse_args()

    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels = df['theme'].to_numpy()
    emb_full = load_embeddings(args.model, df)
    print(f"{args.model}: эмбеддинги {emb_full.shape}")

    knn_rng = np.random.default_rng(42)   # один rng на все dim (как m4b — один вызов на модель)
    sil_rng = np.random.default_rng(42)
    out = {}
    for d in args.dims:
        if d > emb_full.shape[1]:
            print(f"  dim {d}: пропуск (> {emb_full.shape[1]})")
            continue
        embs = norm(emb_full[:, :d])
        loo1, loo3, own_sim, margin = loo_metrics(embs, labels)
        own_arr, correct, _ = loo_own_correct(embs, labels)
        knn = knn_purity(embs, labels, knn_rng)
        ari, nmi = kmeans_agreement(embs, labels)
        sub = sil_rng.choice(len(df), size=min(5000, len(df)), replace=False)
        sil = float(silhouette_score(embs[sub], labels[sub], metric='cosine'))
        th = threshold_curve(own_arr, correct)
        row = {
            'dim': d,
            'loo_top1': loo1, 'loo_top3': loo3,
            'own_sim': own_sim, 'margin': margin,
            'knn_purity_10': knn, 'kmeans_ari': ari, 'kmeans_nmi': nmi,
            'silhouette': sil,
            'unknown_at_acc90': th['unknown_at_acc90'],
            't_at_acc90': th['t_at_acc90'],
            'unknown_at_acc95': th['unknown_at_acc95'],
            't_at_acc95': th['t_at_acc95'],
        }
        out[d] = row
        u90 = row['unknown_at_acc90']
        u95 = row['unknown_at_acc95']
        print(f"  dim {d:>4}: LOO {loo1:.1%} top3 {loo3:.1%} kNN {knn:.3f} "
              f"ARI {ari:.3f} NMI {nmi:.3f} sil {sil:.3f} | "
              f"unknown@90 {u90:.1%} (t={row['t_at_acc90']:.3f}) "
              f"unknown@95 {u95:.1%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    p = REPORTS_DIR / 'm7_mrl.json'
    all_data = {}
    if p.exists():
        all_data = json.loads(io.open(p, encoding='utf-8').read())
    all_data[args.model] = out
    p.write_text(json.dumps(all_data, ensure_ascii=False, indent=2),
                 encoding='utf-8')
    print(f"\nwritten {p}")


if __name__ == '__main__':
    main()
