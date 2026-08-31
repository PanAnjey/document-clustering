# phase_m4c_ari_variance.py
# Разброс ARI/NMI от сида MiniBatchKMeans по всем моделям с готовыми эмбеддингами.
#
# Зачем: в M4b (phase_m4b_intrinsic.py) ARI и NMI считаются ОДНИМ сидом
# (random_state=42), и записанное число попадает в сводную таблицу как
# характеристика модели. Замер 2026-08-30 показал, что собственный разброс
# одной модели от одного лишь сида (ARI 0.04-0.07) сопоставим с разницей
# между моделями во всей таблице (0.093). То есть ранжирование моделей по
# одиночному ARI не подкреплено данными. Здесь считается среднее и разброс
# по нескольким сидам, чтобы в отчёте стояло число с погрешностью.
#
# NMI устойчивее ARI на порядок — этот прогон заодно показывает, насколько.
#
# Запуск: python phase_m4c_ari_variance.py [--seeds 42 7 123 2024 99]

import argparse
import glob
import io
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402

KMEANS_N_INIT = 3  # как в phase_m4b_intrinsic
KMEANS_BATCH = 8192


def load_model_embeddings(model_key: str, df: pd.DataFrame):
    """Склеивает штатные шарды модели → (N, dim) в порядке строк df."""
    parts = [p for p in sorted(glob.glob(str(EMB_DIR / f"{model_key}__shard*.npy")))
             if not p.endswith('_ids.npy')]
    if not parts:
        return None
    emb = np.concatenate([np.load(p) for p in parts])
    ids = np.concatenate([np.load(p.replace('.npy', '_ids.npy')) for p in parts])
    pos = {int(i): k for k, i in enumerate(ids)}
    return emb[[pos[int(i)] for i in df['id']]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123, 2024, 99])
    args = ap.parse_args()

    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels = df['theme'].to_numpy()
    k = len(np.unique(labels))
    print(f"sample: {len(df)} документов, {k} тем, сиды {args.seeds}\n", flush=True)

    rows = []
    for key in MODELS:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key:16s} эмбеддингов нет — пропуск", flush=True)
            continue
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        aris, nmis = [], []
        for s in args.seeds:
            km = MiniBatchKMeans(n_clusters=k, batch_size=KMEANS_BATCH,
                                 n_init=KMEANS_N_INIT, random_state=s)
            pred = km.fit_predict(embs)
            aris.append(float(adjusted_rand_score(labels, pred)))
            nmis.append(float(normalized_mutual_info_score(labels, pred)))
        r = {
            'model': key, 'dim': MODELS[key]['dim'], 'seeds': args.seeds,
            'ari_mean': float(np.mean(aris)), 'ari_std': float(np.std(aris)),
            'ari_min': min(aris), 'ari_max': max(aris), 'ari_all': aris,
            'nmi_mean': float(np.mean(nmis)), 'nmi_std': float(np.std(nmis)),
            'nmi_min': min(nmis), 'nmi_max': max(nmis), 'nmi_all': nmis,
            'ari_seed42': aris[args.seeds.index(42)] if 42 in args.seeds else None,
        }
        rows.append(r)
        print(f"  {key:16s} ARI {r['ari_mean']:.3f} ± {r['ari_std']:.3f} "
              f"({r['ari_min']:.3f}-{r['ari_max']:.3f}) | "
              f"NMI {r['nmi_mean']:.3f} ± {r['nmi_std']:.3f} "
              f"({r['nmi_min']:.3f}-{r['nmi_max']:.3f})", flush=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm4c_ari_variance.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    ari_spread = max(r['ari_mean'] for r in rows) - min(r['ari_mean'] for r in rows)
    nmi_spread = max(r['nmi_mean'] for r in rows) - min(r['nmi_mean'] for r in rows)
    worst_ari = max(r['ari_max'] - r['ari_min'] for r in rows)
    worst_nmi = max(r['nmi_max'] - r['nmi_min'] for r in rows)

    lines = [
        '# M4c: разброс ARI/NMI от сида k-means', '',
        f"Сиды MiniBatchKMeans: {args.seeds}. Векторы и все прочие параметры "
        f"те же, что в M4b — меняется только `random_state`.", '',
        '| Модель | dim | ARI сред. | ARI разброс | ARI при сиде 42 (в отчёте) '
        '| NMI сред. | NMI разброс |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in sorted(rows, key=lambda x: -x['ari_mean']):
        s42 = f"{r['ari_seed42']:.3f}" if r['ari_seed42'] is not None else '—'
        lines.append(
            f"| {r['model']} | {r['dim']} | {r['ari_mean']:.3f} ± {r['ari_std']:.3f} "
            f"| {r['ari_min']:.3f}–{r['ari_max']:.3f} | {s42} "
            f"| {r['nmi_mean']:.3f} ± {r['nmi_std']:.3f} "
            f"| {r['nmi_min']:.3f}–{r['nmi_max']:.3f} |")
    lines += [
        '',
        f"Разброс средних между моделями: ARI {ari_spread:.3f}, NMI {nmi_spread:.3f}.",
        f"Наибольший разброс ОДНОЙ модели от сида: ARI {worst_ari:.3f}, NMI {worst_nmi:.3f}.",
        '',
        'Если разброс одной модели сопоставим с разбросом между моделями, '
        'ранжировать модели по этой метрике нельзя.',
    ]
    (REPORTS_DIR / 'm4c_ari_variance.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nРазброс средних между моделями: ARI {ari_spread:.3f}, NMI {nmi_spread:.3f}")
    print(f"Наибольший разброс одной модели: ARI {worst_ari:.3f}, NMI {worst_nmi:.3f}")
    print(f"Отчёт: {REPORTS_DIR / 'm4c_ari_variance.md'}")


if __name__ == '__main__':
    main()
