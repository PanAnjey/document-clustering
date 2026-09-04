# -*- coding: utf-8 -*-
# score.py  -- агрегирует zeroshot_seed*.jsonl / finetuned_seed*.jsonl:
# точность общая / на независимых (group_size==1) / на документах с
# близнецами (group_size>1), macro-F1, разбор по классам, среднее±разброс
# по сидам. Пишет results.json + reports/*.md + графики. Быстрый.
#
#   python score.py

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import BASE, FT_JSONL, LABELS, RESULTS_JSON, SEEDS, ZS_JSONL

# для контекста в отчёте: предыдущие замеры той же модели/рецепта
PREV = {
    'naive':   {'ft_overall': 0.903, 'zs_overall': 0.692},   # train2_full, случайный сплит, утечка
    'grouped': {'ft_sing': 0.858, 'ft_overall': 0.697,        # ../train2_grouped, грубая чистка (52 метки)
                'zs_sing': 0.647, 'zs_overall': 0.544},
}
OLD_FT_OVERALL = PREV['naive']['ft_overall']
OLD_ZS_4B = PREV['naive']['zs_overall']
REPORTS = BASE / 'reports'
REPORTS.mkdir(exist_ok=True)


def load(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    return rows


def metrics(rows):
    """acc overall / singleton / ingroup, macro-F1, per-class recall, confusions."""
    tot = len(rows)
    cor = sum(r['pred_type'] == r['true_type'] for r in rows)
    sing = [r for r in rows if r['group_size'] == 1]
    ing = [r for r in rows if r['group_size'] > 1]
    acc_s = (sum(r['pred_type'] == r['true_type'] for r in sing) / len(sing)) if sing else None
    acc_i = (sum(r['pred_type'] == r['true_type'] for r in ing) / len(ing)) if ing else None

    tp = Counter(); fp = Counter(); fn = Counter(); n = Counter()
    for r in rows:
        t, p = r['true_type'], r['pred_type']
        n[t] += 1
        if p == t:
            tp[t] += 1
        else:
            fn[t] += 1
            if p is not None:
                fp[p] += 1
    f1s = []
    per_class = {}
    for c in LABELS:
        if n[c] == 0:
            continue
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / n[c]
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class[c] = {'n': n[c], 'recall': rec, 'precision': prec, 'f1': f1}
    conf = Counter()
    for r in rows:
        if r['pred_type'] != r['true_type']:
            conf[(r['true_type'], str(r['pred_type']))] += 1
    return {
        'n': tot, 'acc': cor / tot,
        'acc_singleton': acc_s, 'n_singleton': len(sing),
        'acc_ingroup': acc_i, 'n_ingroup': len(ing),
        'macro_f1': sum(f1s) / len(f1s) if f1s else 0.0,
        'per_class': per_class,
        'top_confusions': conf.most_common(12),
    }


def agg(key, dicts):
    vals = [d[key] for d in dicts if d[key] is not None]
    if not vals:
        return None
    return {'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals),
            'sd': st.pstdev(vals) if len(vals) > 1 else 0.0, 'seeds': vals}


def main():
    per_seed = {}
    for seed in SEEDS:
        zs = str(ZS_JSONL).format(seed=seed)
        ft = str(FT_JSONL).format(seed=seed)
        if not (Path(zs).exists() and Path(ft).exists()):
            print(f'сид {seed}: нет данных, пропуск')
            continue
        per_seed[seed] = {'zeroshot': metrics(load(zs)), 'finetuned': metrics(load(ft))}

    if not per_seed:
        print('нет результатов'); return

    out = {'seeds': list(per_seed), 'per_seed': per_seed, 'aggregate': {}}
    for phase in ('zeroshot', 'finetuned'):
        ds = [per_seed[s][phase] for s in per_seed]
        out['aggregate'][phase] = {k: agg(k, ds) for k in
                                   ('acc', 'acc_singleton', 'acc_ingroup', 'macro_f1')}

    # разбор по классам: средний recall по сидам, до/после
    classes = sorted({c for s in per_seed for c in per_seed[s]['finetuned']['per_class']})
    per_class_delta = []
    for c in classes:
        zr = [per_seed[s]['zeroshot']['per_class'].get(c, {}).get('recall')
              for s in per_seed if c in per_seed[s]['zeroshot']['per_class']]
        fr = [per_seed[s]['finetuned']['per_class'].get(c, {}).get('recall')
              for s in per_seed if c in per_seed[s]['finetuned']['per_class']]
        ns = [per_seed[s]['finetuned']['per_class'].get(c, {}).get('n')
              for s in per_seed if c in per_seed[s]['finetuned']['per_class']]
        if not fr:
            continue
        per_class_delta.append({
            'class': c, 'n_test_mean': sum(ns) / len(ns),
            'recall_zeroshot': sum(zr) / len(zr) if zr else None,
            'recall_finetuned': sum(fr) / len(fr),
        })
    out['per_class'] = per_class_delta

    json.dump(out, open(RESULTS_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---- текстовый отчёт ----
    L = []
    L.append('# «Правильное» дообучение на TRAIN2 — результаты\n')
    L.append(f'Сиды разбиения: {list(per_seed)}. Групповой сплит (без утечки '
             f'шаблонов), 52 заведомо ошибочные метки удалены. База Qwen3.5-4B, '
             f'LoRA r=16, 1 эпоха — рецепт как в train2_full.\n')
    L.append('## Точность (среднее по сидам, в скобках min–max)\n')
    L.append('| Замер | Общая | На независимых | На док. с близнецами | macro-F1 |')
    L.append('|---|---|---|---|---|')
    for phase, name in (('zeroshot', 'Qwen3.5-4B вхолодную'),
                        ('finetuned', 'Qwen3.5-4B + LoRA')):
        a = out['aggregate'][phase]
        def cell(k):
            v = a[k]
            return f'{v["mean"]:.1%} ({v["min"]:.1%}–{v["max"]:.1%})' if v else '—'
        L.append(f'| {name} | {cell("acc")} | {cell("acc_singleton")} | '
                 f'{cell("acc_ingroup")} | {cell("macro_f1")} |')
    L.append('')
    ftm = out['aggregate']['finetuned']['acc']['mean']
    zsm = out['aggregate']['zeroshot']['acc']['mean']
    L.append(f'Прирост от дообучения: **+{(ftm - zsm) * 100:.1f} п.п.** '
             f'({zsm:.1%} → {ftm:.1%}).')
    g = PREV['grouped']
    L.append(f'Предыдущие замеры той же модели: наивный (случайный сплит, утечка) '
             f'{PREV["naive"]["ft_overall"]:.1%} общая; групповой сплит + грубая '
             f'чистка (52 метки) {g["ft_sing"]:.1%} на независимых / {g["ft_overall"]:.1%} общая.')
    fts = out['aggregate']['finetuned']['acc_singleton']
    if fts:
        L.append(f'Честное число для сравнения с эмбеддингами — точность на '
                 f'независимых документах: **{fts["mean"]:.1%}** '
                 f'({fts["min"]:.1%}–{fts["max"]:.1%}).')
    L.append('\n## По классам (recall, среднее по сидам)\n')
    L.append('| Класс | n test | вхолодную | + LoRA | Δ |')
    L.append('|---|--:|--:|--:|--:|')
    for r in sorted(per_class_delta, key=lambda x: (x['recall_finetuned'] - (x['recall_zeroshot'] or 0))):
        z = r['recall_zeroshot']
        d = r['recall_finetuned'] - (z or 0)
        L.append(f'| {r["class"]} | {r["n_test_mean"]:.0f} | '
                 f'{"—" if z is None else f"{z:.0%}"} | {r["recall_finetuned"]:.0%} | '
                 f'{d:+.0%} |')
    L.append('\n## Топ-путаницы дообученной модели (сид ' + str(list(per_seed)[0]) + ')\n')
    for (t, p), c in per_seed[list(per_seed)[0]]['finetuned']['top_confusions'][:10]:
        L.append(f'- {t} → {p}: ×{c}')
    (REPORTS / 'summary.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))

    # ---- графики ----
    _chart_overall(out)
    _chart_perclass(per_class_delta)
    print(f'\n-> {RESULTS_JSON}\n-> {REPORTS / "summary.md"}\n-> {REPORTS}/chart_*.png')


def _chart_overall(out):
    cats = [('acc', 'Общая'), ('acc_singleton', 'На независимых'),
            ('acc_ingroup', 'С близнецами')]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(cats))
    for off, phase, lab, col in ((-0.19, 'zeroshot', 'вхолодную', '#b0b0b0'),
                                 (0.19, 'finetuned', '+ LoRA', '#2c6355')):
        a = out['aggregate'][phase]
        ys = [a[k]['mean'] if a[k] else 0 for k, _ in cats]
        lo = [a[k]['mean'] - a[k]['min'] if a[k] else 0 for k, _ in cats]
        hi = [a[k]['max'] - a[k]['mean'] if a[k] else 0 for k, _ in cats]
        ax.bar([i + off for i in x], ys, 0.36, label=lab, color=col,
               yerr=[lo, hi], capsize=4)
        for i, y in zip(x, ys):
            ax.text(i + off, y + 0.02, f'{y:.0%}', ha='center', fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([c[1] for c in cats])
    ax.set_ylim(0, 1); ax.set_ylabel('точность'); ax.legend()
    ax.set_title('TRAIN2, групповой сплит: до / после LoRA (3 сида)')
    fig.tight_layout(); fig.savefig(REPORTS / 'chart_overall.png', dpi=130)
    plt.close(fig)


def _chart_perclass(pcd):
    pcd = sorted(pcd, key=lambda x: (x['recall_finetuned'] - (x['recall_zeroshot'] or 0)))
    labs = [r['class'] for r in pcd]
    y = range(len(labs))
    zs = [(r['recall_zeroshot'] or 0) for r in pcd]
    ft = [r['recall_finetuned'] for r in pcd]
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.42 * len(labs))))
    ax.hlines(list(y), zs, ft, color='#c3c7be', zorder=1)
    ax.scatter(zs, list(y), s=34, color='#b0b0b0', label='вхолодную', zorder=2)
    ax.scatter(ft, list(y), s=34, color='#2c6355', label='+ LoRA', zorder=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlim(-0.03, 1.03); ax.set_xlabel('recall'); ax.legend(loc='lower right')
    ax.set_title('Recall по классам, до / после LoRA (среднее по сидам)')
    fig.tight_layout(); fig.savefig(REPORTS / 'chart_perclass.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
