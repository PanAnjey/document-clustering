# phase_m4_eval.py
# Фаза M4: оценка качества тематического пространства каждой модели
# на выборке phase_m2 (метки — темы V3, 26 + кластер 90).
#
# Метрики:
#   LOO top-1 / top-3 — точность ближайшего центроида темы с точным
#     leave-one-out (центроид пересчитывается без самого документа);
#   own-sim  — средняя близость документа к центроиду своей темы;
#   margin   — own-sim − max sim к чужим центроидам;
#   silhouette (косинусный, подвыборка 5000) — общая разделимость тем;
#   скорость/VRAM — из meta.json фазы M3.
#
# Запуск: python phase_m4_eval.py

import glob
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from em_config import EMB_DIR, MODELS, REPORTS_DIR, SAMPLE_CSV  # noqa: E402


def load_model_embeddings(model_key: str, df: pd.DataFrame):
    """Склеивает шарды → (N, dim) в порядке строк df. None, если шардов нет."""
    parts = sorted(glob.glob(str(EMB_DIR / f"{model_key}__shard*.npy")))
    parts = [p for p in parts if not p.endswith('_ids.npy')]
    if not parts:
        return None
    emb = np.concatenate([np.load(p) for p in parts])
    ids = np.concatenate([np.load(p.replace('.npy', '_ids.npy')) for p in parts])
    pos = {int(i): k for k, i in enumerate(ids)}
    order = [pos[int(i)] for i in df['id']]
    return emb[order]


def loo_metrics(embs: np.ndarray, labels: np.ndarray):
    """Точный leave-one-out по центроидам тем (эффект самовключения убран)."""
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    themes = np.unique(labels)
    sums = {t: embs[labels == t].sum(axis=0) for t in themes}
    counts = {t: int((labels == t).sum()) for t in themes}

    cents = []
    for i, t in enumerate(labels):
        n = counts[t]
        if n < 2:
            cents.append(None)
            continue
        c = (sums[t] - embs[i]) / (n - 1)
        cents.append(c / (np.linalg.norm(c) + 1e-12))
    # центроиды «чужих» тем общие для всех
    theme_cents = {t: sums[t] / (np.linalg.norm(sums[t]) + 1e-12) for t in themes}
    theme_list = sorted(themes)
    C = np.stack([theme_cents[t] for t in theme_list])

    sims_all = embs @ C.T  # (N, n_themes) — к «чужим» корректно, к своему приближённо
    top1 = top3 = 0
    own_sims, margins = [], []
    n_skipped = 0
    t2col = {t: k for k, t in enumerate(theme_list)}
    for i, t in enumerate(labels):
        if cents[i] is None:
            n_skipped += 1
            continue
        own = float(embs[i] @ cents[i])
        others = sims_all[i].copy()
        others[t2col[t]] = -1  # свою тему считаем через LOO-центроид
        best_other = float(others.max())
        own_sims.append(own)
        margins.append(own - best_other)
        if own > best_other:
            top1 += 1
        # top-3: своя тема среди трёх ближайших?
        combined = others.copy()
        combined[t2col[t]] = own
        if t in theme_list and np.argsort(combined)[-3:].max() >= 0:
            top3 += int(t2col[t] in np.argsort(combined)[-3:])
    n_valid = len(own_sims)
    return {
        'loo_top1': top1 / max(n_valid, 1),
        'loo_top3': top3 / max(n_valid, 1),
        'own_sim': float(np.mean(own_sims)),
        'margin': float(np.mean(margins)),
        'n_valid': n_valid, 'n_skipped': n_skipped,
    }


def main():
    df = pd.read_csv(SAMPLE_CSV, encoding='utf-8')
    labels = df['theme'].to_numpy()
    print(f"sample: {len(df)} документов, {len(np.unique(labels))} тем")

    rows = []
    for key in MODELS:
        embs = load_model_embeddings(key, df)
        if embs is None:
            print(f"  {key}: эмбеддингов нет — пропуск")
            continue
        print(f"  {key}: {embs.shape}")
        m = loo_metrics(embs, labels)

        # silhouette на подвыборке (косинус)
        rng = np.random.default_rng(42)
        sub = rng.choice(len(df), size=min(5000, len(df)), replace=False)
        m['silhouette'] = float(silhouette_score(
            embs[sub], labels[sub], metric='cosine'))

        # скорость/VRAM из meta шардов
        metas = []
        for mp in sorted(glob.glob(str(EMB_DIR / f"{key}__shard*_meta.json"))):
            metas.append(json.loads(io.open(mp, encoding='utf-8').read()))
        m['docs_per_s'] = sum(x['docs_per_s'] for x in metas) if metas else None
        m['peak_vram'] = max((x['peak_vram_gb'] for x in metas), default=None)

        rows.append({'model': key, 'dim': MODELS[key]['dim'], **m})
        print(f"    LOO top-1 {m['loo_top1']:.2%} top-3 {m['loo_top3']:.2%} "
              f"margin {m['margin']:.3f} silhouette {m['silhouette']:.3f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm4_eval.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# M4: документный тест (LOO по центроидам тем, ~50K XML)', '']
    lines.append('| Модель | dim | LOO top-1 | LOO top-3 | own-sim | margin '
                 '| silhouette | док/с | VRAM GB |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        dps = f"{r['docs_per_s']:.0f}" if r['docs_per_s'] else '—'
        vram = f"{r['peak_vram']:.1f}" if r['peak_vram'] else '—'
        lines.append(f"| {r['model']} | {r['dim']} | {r['loo_top1']:.1%} | "
                     f"{r['loo_top3']:.1%} | {r['own_sim']:.3f} | {r['margin']:.3f} | "
                     f"{r['silhouette']:.3f} | {dps} | {vram} |")
    lines.append('')
    lines.append('Метки тем получены в пространстве nomic_v15 (V3 assign) — '
                 'у baseline лёгкое «домашнее» преимущество.')
    out = REPORTS_DIR / 'm4_eval.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nwritten {out}")


if __name__ == '__main__':
    main()
