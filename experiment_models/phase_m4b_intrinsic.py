# phase_m4b_intrinsic.py
# Фаза M4b: справедливые метрики без использования центроидов v1.5.
# LOO top-1 в M4 даёт baseline «домашнее» преимущество (метки построены
# в его пространстве). Здесь метки используются ТОЛЬКО как эталон,
# центроиды v1.5 не участвуют:
#
#   knn_purity@10 — доля соседей своей темы среди 10 ближайших
#     (чистота локальной окрестности; подвыборка 15K);
#   ARI / NMI — согласованность MiniBatchKMeans (k=25, сферический)
#     с эталонными темами: можно ли восстановить таксономию
#     кластеризацией в НОВОМ пространстве.
#
# Запуск: python phase_m4b_intrinsic.py

import io
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from em_config import MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402
from phase_m4_eval import load_model_embeddings  # noqa: E402

KNN_SUB = 15000
KNN_K = 10
KMEANS_N_INIT = 3


def knn_purity(embs: np.ndarray, labels: np.ndarray, rng) -> float:
    n = len(embs)
    idx = rng.choice(n, size=min(KNN_SUB, n), replace=False)
    sub_e = embs[idx]
    sub_l = labels[idx]
    purities = []
    for start in range(0, len(idx), 2000):
        chunk = sub_e[start:start + 2000]
        sims = chunk @ sub_e.T
        rows = np.arange(len(chunk))
        sims[rows, start + rows] = -1  # исключаем самих себя
        nn = np.argpartition(-sims, KNN_K, axis=1)[:, :KNN_K]
        same = (sub_l[nn] == sub_l[start:start + 2000][:, None]).mean(axis=1)
        purities.append(same)
    return float(np.concatenate(purities).mean())


def kmeans_agreement(embs: np.ndarray, labels: np.ndarray) -> tuple:
    k = len(np.unique(labels))
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192,
                         n_init=KMEANS_N_INIT, random_state=42)
    pred = km.fit_predict(embs)
    return (float(adjusted_rand_score(labels, pred)),
            float(normalized_mutual_info_score(labels, pred)))


def main():
    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels = df['theme'].to_numpy()
    print(f"sample: {len(df)} документов, {len(np.unique(labels))} тем")
    rng = np.random.default_rng(42)

    rows = []
    for key in MODELS:
        embs = load_model_embeddings(key, df)
        if embs is None:
            continue
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        print(f"  {key}: ", end='', flush=True)
        purity = knn_purity(embs, labels, rng)
        print(f"knn@10 {purity:.3f} | ", end='', flush=True)
        ari, nmi = kmeans_agreement(embs, labels)
        print(f"ARI {ari:.3f} NMI {nmi:.3f}")
        rows.append({'model': key, 'dim': MODELS[key]['dim'],
                     'knn_purity_10': purity, 'kmeans_ari': ari, 'kmeans_nmi': nmi})

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm4b_intrinsic.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# M4b: интринсик-метрики (без центроидов v1.5, ~46.5K XML)', '']
    lines.append('| Модель | dim | kNN purity@10 | KMeans ARI | KMeans NMI |')
    lines.append('|---|---|---|---|---|')
    for r in rows:
        lines.append(f"| {r['model']} | {r['dim']} | {r['knn_purity_10']:.3f} | "
                     f"{r['kmeans_ari']:.3f} | {r['kmeans_nmi']:.3f} |")
    lines.append('')
    lines.append('kNN purity — доля соседей своей темы (подвыборка 15K). '
                 'ARI/NMI — согласованность MiniBatchKMeans(k=25) с эталонными темами. '
                 'Метки тем из пространства v1.5 — остаточное преимущество baseline '
                 'возможно, но слабее, чем в LOO (центроиды не используются).')
    out = REPORTS_DIR / 'm4b_intrinsic.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nwritten {out}")


if __name__ == '__main__':
    main()
