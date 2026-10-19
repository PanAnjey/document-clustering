# build_data.py — eval.json (400 строк, эталон по правилам gold.py, без строк «месяц без года») и train.json (силвер-разметка для Giga-тегера).
import json, pickle, random, re, sys
from pathlib import Path
from gold import gold, gold_frag
HERE = Path(__file__).parent
ANY = re.compile(r'январ|феврал|март|апрел|мая|май|июн|июл|август|сентябр|октябр|ноябр|декабр|квартал|период|\d{1,2}\.\d{1,2}\.\d{2,4}|20\d\d', re.I)
MONTHW = re.compile(r'январ|феврал|март|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр', re.I)
rows = json.load(open(HERE / 'rows400.json', encoding='utf-8'))
ev = []; drop = 0
for r in rows:
    v, sp = gold(r['text'])
    if not v and re.search(r'(?:за|период|на)\s+\w*\s*(' + MONTHW.pattern + ')', r['text'], re.I):
        drop += 1; continue      # месяц без года — однозначной даты нет
    ev.append({'id': r['id'], 'text': r['text'], 'gold': list(v) if v else None, 'ncand': len(sp)})
json.dump(ev, open(HERE / 'eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('eval', len(ev), 'с периодом', sum(1 for e in ev if e['gold']), 'multi-кандидат', sum(1 for e in ev if e['ncand'] > 1), 'исключено', drop)
# train
res = pickle.load(open(HERE.parent / 'experiment_invoice_lines/work/tab_0.pkl', 'rb'))
allr = [x for _, r, _ in res for x in r]
norm = lambda s: re.sub(r'\d+', '#', re.sub(r'\s+', ' ', s.lower().replace('¶', '')).strip())
evk = {norm(e['text']) for e in ev}
u = {}
for x in allr:
    t = re.sub(r'\s+', ' ', x['item'].replace('¶', ' ')).strip()[:300]
    if norm(t) not in evk: u.setdefault(norm(t), t)
keys = sorted(u); random.Random(11).shuffle(keys)
# сложный набор: диапазоны, одиночные даты, строки с несколькими кандидатами
hard = []
for k in keys:
    t = u[k]; v, sp = gold(t)
    if v and (len(sp) > 1 or any(s[2] in ('range', 'date') for s in sp)):
        hard.append(k)
hard = hard[:150]; hk = set(hard)
for k in hard:
    t = u[k]; v, sp = gold(t)
    ev.append({'id': 1000 + len(ev), 'text': t, 'gold': list(v), 'ncand': len(sp), 'hard': 1})
json.dump(ev, open(HERE / 'eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('eval+hard', len(ev), 'hard', len(hard), 'multi', sum(1 for e in ev if e['ncand'] > 1))
keys = [k for k in keys if k not in hk]
tr = []
for k in keys:
    t = u[k]; g = gold_frag(t)
    if not g and MONTHW.search(t): continue
    tr.append({'text': t, 'span': [g[1], g[2]] if g else None})
    if len(tr) >= 3500: break
json.dump(tr, open(HERE / 'train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('train', len(tr), 'с периодом', sum(1 for x in tr if x['span']))
