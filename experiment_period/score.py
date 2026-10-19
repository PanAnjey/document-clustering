# score.py — метрики по pred_*.json против эталона eval.json -> report_metrics.md + errors_<tag>.json
import json, sys
from pathlib import Path
HERE = Path(__file__).parent
ev = json.load(open(HERE / 'eval.json', encoding='utf-8'))
G = {e['id']: e for e in ev}
tags = [p.stem[5:] for p in sorted(HERE.glob('pred_*.json'))]


def strata(e):
    s = ['all']
    s.append('xml' if e.get('xml') else ('hard' if e.get('hard') else 'easy'))
    s.append('has' if e['gold'] else 'none')
    if e['gold']:
        s.append('range' if e['gold'][0] != e['gold'][1] and not (e['gold'][0][8:] == '01') else ('single' if e['gold'][0] == e['gold'][1] else 'month'))
    if e['ncand'] > 1: s.append('multi')
    return s


def eq(p, g):
    if g is None: return p is None
    return isinstance(p, list) and p == g


def start_ok(p, g):
    return isinstance(p, list) and g is not None and p[0] == g[0]


rows = {}
for t in tags:
    d = json.load(open(HERE / f'pred_{t}.json', encoding='utf-8'))
    P = {r['id']: r['pred'] for r in d['rows']}
    agg = {}; errs = []
    for e in ev:
        p = P.get(e['id'])
        ok = eq(p, e['gold'])
        for s in strata(e):
            a = agg.setdefault(s, [0, 0, 0]); a[0] += 1; a[1] += ok; a[2] += start_ok(p, e['gold'])
        if not ok: errs.append({'id': e['id'], 'text': e['text'], 'gold': e['gold'], 'pred': p})
    bad = sum(1 for r in d['rows'] if r['pred'] == 'bad')
    rows[t] = (agg, d['seconds'], d['n'], bad)
    json.dump(errs, open(HERE / f'errors_{t}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

cols = ['all', 'easy', 'hard', 'xml', 'has', 'none', 'month', 'range', 'single', 'multi']
L = ['| модель | ' + ' | '.join(cols) + ' | не разобрано | стр/с |', '|' + '---|' * (len(cols) + 3)]
for t in tags:
    agg, sec, n, bad = rows[t]
    L.append(f'| {t} | ' + ' | '.join(f'{100*agg[c][1]/agg[c][0]:.1f}%' if c in agg else '–' for c in cols) + f' | {bad} | {n/sec:.1f} |')
L.append('')
L.append('Размеры страт: ' + ', '.join(f'{c}={rows[tags[0]][0][c][0]}' for c in cols if c in rows[tags[0]][0]))
L.append('')
L.append('Доля совпадения только даты начала (мягкая метрика), страта all: ' + ', '.join(f'{t} {100*rows[t][0]["all"][2]/rows[t][0]["all"][0]:.1f}%' for t in tags))
out = '\n'.join(L)
(HERE / 'report_metrics.md').write_text(out, encoding='utf-8')
print(out)
