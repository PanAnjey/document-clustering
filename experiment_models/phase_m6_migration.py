# phase_m6_migration.py
# Фаза M6: миграция кластеров между embedding-моделями.
#
# Вопрос пользователя: «есть ли переползание документов в другие кластеры
# в зависимости от модели эмбеддингов?»
#
# Track A (центроидное присвоение — как продакшен assign тем):
#   Для каждой модели: центроиды 25 тем в её пространстве → присвоение
#   top-1 + продакшен-порог sim>=0.8 (иначе 'unknown').
#   Метрики: coverage; pairwise ARI/NMI между присвоениями;
#   матрица миграции nomic_v15 → каждая модель (топ переходов);
#   % документов, сменивших тему относительно исходной разметки V3.
#
# Track B (HDBSCAN-структура, зеркало продакшена cluster_engine):
#   HDBSCAN(metric='euclidean' на L2-норм. (= cosine), min_cluster_size=95,
#   cluster_selection_epsilon=sqrt(2*0.3) — эквивалент cosine eps=0.3).
#   Метрики: # кластеров, % шума, pairwise ARI/NMI между разбиениями,
#   ARI/NMI vs исходные темы V3.
#
# Запуск:
#   python phase_m6_migration.py --track a
#   python phase_m6_migration.py --track b --models nomic_v15 nomic_v2 qwen3e_4b qwen3e_06b

import argparse
import glob
import io
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402

SIM_THRESHOLD = 0.8          # продакшен-порог присвоения темы
MIN_CLUSTER_SIZE = 95        # продакшен cfg.MIN_CLUSTER_SIZE
# cosine eps=0.3 (1-0.70 SIMILARITY_THRESHOLD) -> euclidean на единичной сфере:
EPS_EUCLID = float(np.sqrt(2 * 0.3))


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
    """Центроиды тем в пространстве модели → (assigned, sim)."""
    embs = norm(embs)
    themes = sorted(np.unique(labels))
    cents = np.stack([norm(embs[labels == t].mean(axis=0, keepdims=True))[0] for t in themes])
    sims = embs @ cents.T
    best = sims.argmax(axis=1)
    assigned = np.array([themes[i] for i in best])
    return assigned, sims.max(axis=1)


def track_a(df, model_keys):
    labels_true = df['theme'].to_numpy()
    assigns, sims_all = {}, {}
    for key in model_keys:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key}: нет эмбеддингов — пропуск")
            continue
        a, s = assign_themes(embs, labels_true)
        assigns[key] = a
        sims_all[key] = s
        cov = float((s >= SIM_THRESHOLD).mean())
        diff_vs_v3 = float((a != labels_true).mean())
        print(f"  {key}: coverage(>=0.8) {cov:.1%}, отличается от разметки V3 {diff_vs_v3:.1%}")

    keys = list(assigns)
    # pairwise ARI/NMI по присвоениям (только присвоенные >=0.8 у обеих)
    pairs = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            m = (sims_all[k1] >= SIM_THRESHOLD) & (sims_all[k2] >= SIM_THRESHOLD)
            ari = adjusted_rand_score(assigns[k1][m], assigns[k2][m])
            nmi = normalized_mutual_info_score(assigns[k1][m], assigns[k2][m])
            moved = float((assigns[k1][m] != assigns[k2][m]).mean())
            pairs.append({'pair': f'{k1} vs {k2}', 'n_common': int(m.sum()),
                          'ari': ari, 'nmi': nmi, 'moved_frac': moved})
            print(f"  {k1} vs {k2}: ARI {ari:.3f} NMI {nmi:.3f} moved {moved:.1%} (n={m.sum()})")

    # Матрицы миграции: nomic_v15 (baseline) -> каждая модель
    migrations = {}
    base = 'nomic_v15'
    if base in assigns:
        for key in keys:
            if key == base:
                continue
            m = (sims_all[base] >= SIM_THRESHOLD) & (sims_all[key] >= SIM_THRESHOLD)
            a_from, a_to = assigns[base][m], assigns[key][m]
            trans = {}
            for x, y in zip(a_from, a_to):
                if x != y:
                    trans[(x, y)] = trans.get((x, y), 0) + 1
            top = sorted(trans.items(), key=lambda kv: -kv[1])[:15]
            migrations[key] = {
                'n_common': int(m.sum()),
                'moved_total': int((a_from != a_to).sum()),
                'top_transitions': [
                    {'from': x, 'to': y, 'n': n} for (x, y), n in top],
            }
    # per-model coverage/diff для отчёта
    per_model = {k: {'coverage': float((sims_all[k] >= SIM_THRESHOLD).mean()),
                     'diff_vs_v3': float((assigns[k] != labels_true).mean())}
                 for k in keys}
    return {'per_model': per_model, 'pairs': pairs, 'migration_from_nomic_v15': migrations}


def track_b(df, model_keys):
    import hdbscan  # noqa: E402
    labels_true = df['theme'].to_numpy()
    clabels = {}
    stats = {}
    for key in model_keys:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key}: нет эмбеддингов — пропуск")
            continue
        X = norm(embs)
        t0 = time.time()
        cl = hdbscan.HDBSCAN(
            metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE,
            cluster_selection_epsilon=EPS_EUCLID, core_dist_n_jobs=-1)
        lab = cl.fit_predict(X)
        dt = time.time() - t0
        clabels[key] = lab
        n_cl = len(set(lab) - {-1})
        noise = float((lab == -1).mean())
        ari_t = adjusted_rand_score(labels_true, lab)
        nmi_t = normalized_mutual_info_score(labels_true, lab)
        stats[key] = {'clusters': n_cl, 'noise': noise, 'time_s': round(dt, 1),
                      'ari_vs_v3': ari_t, 'nmi_vs_v3': nmi_t}
        print(f"  {key}: {n_cl} кластеров, шум {noise:.1%}, ARI(V3) {ari_t:.3f} "
              f"NMI(V3) {nmi_t:.3f} ({dt:.0f} с)")
    pairs = []
    keys = list(clabels)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            # ARI/NMI по всем точкам (шум = общий класс -1)
            ari = adjusted_rand_score(clabels[k1], clabels[k2])
            nmi = normalized_mutual_info_score(clabels[k1], clabels[k2])
            # и по точкам, кластеризованным в ОБЕИХ моделях
            m = (clabels[k1] != -1) & (clabels[k2] != -1)
            ari_c = adjusted_rand_score(clabels[k1][m], clabels[k2][m]) if m.sum() > 100 else None
            moved = float((clabels[k1] != clabels[k2]).mean())
            pairs.append({'pair': f'{k1} vs {k2}', 'ari': ari, 'nmi': nmi,
                          'ari_clustered_only': ari_c, 'moved_frac': moved,
                          'n_both_clustered': int(m.sum())})
            print(f"  {k1} vs {k2}: ARI {ari:.3f} NMI {nmi:.3f} "
                  f"ARI(cl) {ari_c if ari_c is None else round(ari_c,3)} moved {moved:.1%}")
    return {'hdbscan_stats': stats, 'hdbscan_pairs': pairs,
            'params': {'min_cluster_size': MIN_CLUSTER_SIZE, 'eps_euclid': EPS_EUCLID}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=['a', 'b', 'all'], default='all')
    ap.add_argument('--models', nargs='*', default=None)
    args = ap.parse_args()

    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    print(f"sample: {len(df)} документов, {df['theme'].nunique()} тем")
    model_keys = args.models or list(MODELS)

    out = {}
    if args.track in ('a', 'all'):
        print("\n=== Track A: центроидное присвоение (продакшен-порог 0.8) ===")
        out.update(track_a(df, model_keys))
    if args.track in ('b', 'all'):
        print("\n=== Track B: HDBSCAN-структура (продакшен-параметры) ===")
        out.update(track_b(df, model_keys))

    REPORTS_DIR.mkdir(exist_ok=True)
    mode = args.track if args.track != 'all' else 'ab'

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    (REPORTS_DIR / f'm6_migration_{mode}.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
        encoding='utf-8')
    print(f"\nwritten {REPORTS_DIR / f'm6_migration_{mode}.json'}")


if __name__ == '__main__':
    main()
