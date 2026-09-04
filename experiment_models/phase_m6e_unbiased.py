# phase_m6e_unbiased.py
# Фаза M6e: честное сравнение моделей — кластеризация с нуля без эталонных меток.
#
# Проблема M4/M4b/M6: все метрики опираются на V3-assign (получен в пространстве
# nomic_v15), что даёт baseline «домашнее» преимущество. LOO, ARI vs V3, Track A
# центроидное присвоение — все сравнивают с эталоном v1.5.
#
# Здесь: для КАЖДОЙ модели HDBSCAN с фиксированными параметрами на сырых
# эмбеддингах, ЕДИНСТВЕННАЯ оценка — intrinsic (без эталонных меток).
#
# Метрики:
#   n_clusters, noise_frac        — сколько кластеров найдено, сколько шума
#   intra_cos_mean/median         — средняя косинусная близость к центроиду кластера
#   silhouette                    — разделимость кластеров (−1..1)
#   davies_bouldin                — компактность + разделимость (меньше = лучше)
#   calinski_harabasz             — отношение межкл. дисперсии к внурикл. (больше = лучше)
#   cluster_sizes                 — статистика размеров
#
# Парное сравнение: pairwise ARI между HDBSCAN-разбиениями разных моделей.
#
# Запуск:
#   python phase_m6e_unbiased.py
#   python phase_m6e_unbiased.py --models nomic_v2 qwen3e_4b --threshold 0.85
#
# Зависимости: embeddings из M3 (embeddings/{model}__shard*.npy)

import argparse
import glob
import io
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV

DEFAULT_MIN_CLUSTER = 95
DEFAULT_THRESHOLD = 0.80


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


def intra_cluster_metrics(X, labels):
    """Средняя внутрикластерная косинусная близость к центроиду.
    labels: -1 = шум (исключается)."""
    unique = [c for c in np.unique(labels) if c != -1]
    if not unique:
        return {}
    cos_sims = []
    cluster_sizes = {}
    for c in unique:
        mask = labels == c
        n = int(mask.sum())
        cluster_sizes[int(c)] = n
        centroid = X[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = X[mask] @ centroid
        cos_sims.extend(sims.tolist())
    arr = np.array(cos_sims)
    return {
        'intra_cos_mean': float(arr.mean()),
        'intra_cos_median': float(np.median(arr)),
        'intra_cos_min': float(arr.min()),
        'cluster_sizes': cluster_sizes,
        'cluster_size_mean': float(np.mean(list(cluster_sizes.values()))),
        'cluster_size_median': float(np.median(list(cluster_sizes.values()))),
        'cluster_size_min': int(min(cluster_sizes.values())),
        'cluster_size_max': int(max(cluster_sizes.values())),
    }


def main():
    ap = argparse.ArgumentParser(
        description='M6e: unbiased HDBSCAN clustering — no reference labels')
    ap.add_argument('--models', nargs='*', default=None,
                    help='модели (по умолчанию все из MODELS)')
    ap.add_argument('--min-cluster', type=int, default=DEFAULT_MIN_CLUSTER,
                    help=f'min_cluster_size (default {DEFAULT_MIN_CLUSTER})')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                    help=f'cosine similarity threshold (default {DEFAULT_THRESHOLD})')
    ap.add_argument('--max-dim', type=int, default=0,
                    help='обрезать эмбеддинги до N координат (MRL, 0 = без обрезки)')
    ap.add_argument('--save-labels', action='store_true',
                    help='сохранить HDBSCAN-метки в reports/m6e_labels_{model}.npy')
    args = ap.parse_args()

    eps_euclid = float(np.sqrt(2 * (1 - args.threshold)))
    print(f"Параметры: min_cluster_size={args.min_cluster}, "
          f"cos_threshold={args.threshold}, eps_euclid={eps_euclid:.4f}")

    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    print(f"выборка: {len(df)} документов")

    model_keys = args.models or list(MODELS)

    import hdbscan

    clabels = {}
    stats = {}
    done_keys = []
    for key in model_keys:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key}: нет эмбеддингов — пропуск")
            continue
        # прогресс-файл для возобновления при таймауте
        progress_file = REPORTS_DIR / f'.m6e_progress_{key}'
        if progress_file.exists():
            print(f"  {key}: уже готов (пропуск)")
            continue
        progress_file.write_text('started', encoding='utf-8')

        X = norm(embs)
        if args.max_dim and args.max_dim < X.shape[1]:
            X = X[:, :args.max_dim]
            X = norm(X)
        t0 = time.time()
        cl = hdbscan.HDBSCAN(
            metric='euclidean', min_cluster_size=args.min_cluster,
            cluster_selection_epsilon=eps_euclid, core_dist_n_jobs=-1)
        lab = cl.fit_predict(X)
        dt = time.time() - t0
        clabels[key] = lab

        if args.save_labels:
            labs_dir = REPORTS_DIR
            labs_dir.mkdir(exist_ok=True)
            np.save(str(labs_dir / f'm6e_labels_{key}.npy'), lab)

        n_cl = len(set(lab) - {-1})
        noise_frac = float((lab == -1).mean())
        n_noise = int((lab == -1).sum())

        intra = intra_cluster_metrics(X, lab)

        # silhouette (только кластеризованные, ≥2 кластеров)
        clustered = lab != -1
        sil = None
        if clustered.sum() >= 100 and n_cl >= 2:
            sil = float(silhouette_score(X[clustered], lab[clustered],
                                         metric='cosine'))

        # Davies-Bouldin / CH (требуют >=2 кластеров)
        db, ch = None, None
        if n_cl >= 2 and clustered.sum() >= n_cl:
            db = float(davies_bouldin_score(X[clustered], lab[clustered]))
            ch = float(calinski_harabasz_score(X[clustered], lab[clustered]))

        stat = {
            'n_clusters': n_cl,
            'n_noise': n_noise,
            'noise_frac': noise_frac,
            'time_s': round(dt, 1),
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
        }
        stat.update(intra)
        stats[key] = stat

        noise_pct = f"{noise_frac:.1%}"
        sil_str = f"{sil:.3f}" if sil is not None else '—'
        db_str = f"{db:.3f}" if db is not None else '—'
        ch_str = f"{ch:.0f}" if ch is not None else '—'
        icm = intra.get('intra_cos_mean', 0)
        print(f"  {key}: {n_cl} кластеров, шум {noise_pct}, "
              f"внутрикл. cos {icm:.3f}, "
              f"sil {sil_str} DB {db_str} CH {ch_str} ({dt:.0f} с)")

    # Парное сравнение (ARI между HDBSCAN-разбиениями, БЕЗ порога)
    pairs = []
    keys_list = list(clabels)
    for i, k1 in enumerate(keys_list):
        for k2 in keys_list[i + 1:]:
            ari = float(adjusted_rand_score(clabels[k1], clabels[k2]))
            lab_both = (clabels[k1] != -1) & (clabels[k2] != -1)
            ari_cl = (float(adjusted_rand_score(clabels[k1][lab_both],
                                                clabels[k2][lab_both]))
                      if lab_both.sum() > 100 else None)
            moved = float((clabels[k1] != clabels[k2]).mean())
            pairs.append({
                'pair': f'{k1} vs {k2}',
                'ari': ari,
                'ari_clustered_only': ari_cl,
                'moved_frac': moved,
                'n_both_clustered': int(lab_both.sum()),
            })
            cl_str = f"{ari_cl:.3f}" if ari_cl is not None else '—'
            print(f"    {k1} ↔ {k2}: ARI {ari:.3f} ARI(cl) {cl_str} "
                  f"moved {moved:.1%}")

    # отметка о завершении модели
    progress_file.write_text('done', encoding='utf-8')

    # Инкрементальное сохранение после каждой модели
    out_part = {
        'params': {
            'min_cluster_size': args.min_cluster,
            'cos_threshold': args.threshold,
            'eps_euclid': round(eps_euclid, 4),
            'n_docs': len(df),
        },
        'per_model': stats,
        'pairwise': pairs,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    p_part = REPORTS_DIR / 'm6e_unbiased_partial.json'
    p_part.write_text(json.dumps(out_part, ensure_ascii=False, indent=2,
                                 default=lambda o: int(o) if isinstance(o, np.integer)
                                 else float(o) if isinstance(o, np.floating)
                                 else str(o)),
                      encoding='utf-8')
    # Отметка прогресса в общий файл
    done_keys.append(key)

    # Сводка
    out = out_part

    # Markdown-отчёт
    L = [f'# M6e: кластеризация с нуля (без эталонных меток V3)',
         '',
         f'Параметры HDBSCAN: min_cluster_size={args.min_cluster}, '
         f'cosine threshold={args.threshold} '
         f'(eps_euclid={eps_euclid:.4f}).',
         f'Выборка: {len(df)} XML-документов, каждая модель строит кластеры '
         f'самостоятельно без подсказок.',
         '',
         '## Основные метрики',
         '',
         '| Модель | dim | #кластеров | шум | '
         'внутрикл. cos (ср) | внутрикл. cos (мед) | '
         'silhouette | DavBouldin↓ | CalHar↑ |',
         '|---|---|---|---|---|---|---|---|---|',
         ]
    for key in model_keys:
        s = stats.get(key)
        if s is None:
            continue
        noise_pct = f"{s['noise_frac']:.1%}"
        sil = f"{s['silhouette']:.3f}" if s['silhouette'] is not None else '—'
        db = f"{s['davies_bouldin']:.3f}" if s['davies_bouldin'] is not None else '—'
        ch = f"{s['calinski_harabasz']:.0f}" if s['calinski_harabasz'] is not None else '—'
        L.append(f"| {key} | {MODELS[key]['dim']} | {s['n_clusters']} | "
                 f"{noise_pct} | {s['intra_cos_mean']:.4f} | "
                 f"{s['intra_cos_median']:.4f} | {sil} | {db} | {ch} |")
    L.append('')
    L.append('## Распределение размеров кластеров')
    L.append('')
    L.append('| Модель | мин. | медиана | среднее | макс. |')
    L.append('|---|---|---|---|---|')
    for key in model_keys:
        s = stats.get(key)
        if s is None:
            continue
        L.append(f"| {key} | {s['cluster_size_min']} | "
                 f"{s['cluster_size_median']:.0f} | "
                 f"{s['cluster_size_mean']:.0f} | {s['cluster_size_max']} |")
    L.append('')
    L.append('## Парное согласие HDBSCAN-разбиений (ARI)')
    L.append('')
    L.append('| Пара | ARI (все точки) | ARI (только кластеризованные) | '
             '% разошедшихся | общих в кластерах |')
    L.append('|---|---|---|---|---|')
    for p in pairs:
        ac = p['ari_clustered_only']
        acs = f"{ac:.3f}" if ac is not None else '—'
        L.append(f"| {p['pair']} | {p['ari']:.3f} | {acs} | "
                 f"{p['moved_frac']:.1%} | {p['n_both_clustered']} |")
    L.append('')
    L.append('---')
    L.append('')
    L.append('### Как читать')
    L.append('')
    L.append('У какой модели **меньше шума** и **выше внутрикластерный cos** '
             'при равном/большем числе кластеров — та и плотнее. '
             'Davies-Bouldin (меньше = лучше) и Calinski-Harabasz (больше = лучше) '
             'подтверждают. Silhouette (ближе к 1) — разделимость кластеров. '
             'ARI между моделями показывает, насколько структуры похожи '
             '(1 = идентичны, 0 = случайны).')
    L.append('')
    L.append('Этот тест честнее M4/M4b: эталонные метки V3 (полученные nomic_v15) '
             'НЕ используются — каждый HDBSCAN строится с нуля.')
    out_md = REPORTS_DIR / 'm6e_unbiased.md'
    out_md.write_text('\n'.join(L), encoding='utf-8')

    print(f"\nwritten {p}")
    print(f"written {out_md}")


if __name__ == '__main__':
    main()
