# run_regex.py — «типовой» набор регулярных выражений для периода, написанный независимо от gold.py (другая структура, без общих функций).
# Два уровня: regex_basic (то, что обычно пишут сразу) и regex_extended (+ кварталы, диапазон месяцев, день-день месяц, 2-значный год, «Дата начала/окончания»).
import calendar, json, re, time
from pathlib import Path
HERE = Path(__file__).parent
MON = {'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'мая': 5, 'май': 5, 'мае': 5, 'июн': 6, 'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12}
M = r'(янв|фев|мар|апр|мая|май|мае|июн|июл|авг|сен|окт|ноя|дек)[а-я]*'
iso = lambda y, m, d: f'{y:04d}-{m:02d}-{d:02d}'
last = lambda y, m: calendar.monthrange(y, m)[1]
mn = lambda w: MON[w[:3].lower()] if w[:3].lower() in MON else MON[w.lower()]
yr = lambda s: int(s) + 2000 if len(s) == 2 else int(s)


def basic(t):
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})\s*(?:г\.?)?\s*(?:по|-|–)\s*(\d{2})\.(\d{2})\.(\d{4})', t)          # 01.03.2026 - 31.03.2026
    if m: return [iso(int(m[3]), int(m[2]), int(m[1])), iso(int(m[6]), int(m[5]), int(m[4]))]
    m = re.search(M + r'\s+(20\d\d)', t, re.I)                                                                     # март 2026
    if m: y = int(m[2]); mo = mn(m[1]); return [iso(y, mo, 1), iso(y, mo, last(y, mo))]
    m = re.search(r'(?<!от )(?<!№ )(?<!\d)(\d{2})\.(\d{2})\.(20\d\d)', t)                                                 # одиночная дата
    if m: d = iso(int(m[3]), int(m[2]), int(m[1])); return [d, d]
    return None


def extended(t):
    t2 = t.replace('«', '').replace('»', '').replace('"', '')
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*г?\.?\s*(?:\d{1,2}:\d{2}(?::\d{2})?)?\s*(?:по|до|-|–|—)\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})', t2)
    if m: return [iso(yr(m[3]), int(m[2]), int(m[1])), iso(yr(m[6]), int(m[5]), int(m[4]))]
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.?\s*(?:по|-|–)\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})', t2)                    # 01.03-31.03.2026
    if m: y = yr(m[5]); return [iso(y, int(m[2]), int(m[1])), iso(y, int(m[4]), int(m[3]))]
    m = re.search(r'начал\w*:?\s*(\d{1,2})\.(\d{1,2})\.(\d{4}).{0,25}?окончани\w*:?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})', t2, re.I)
    if m: return [iso(int(m[3]), int(m[2]), int(m[1])), iso(int(m[6]), int(m[5]), int(m[4]))]
    m = re.search(r'(\d{1,2})\s*(?:по|-|–)\s*(\d{1,2})\s+' + M + r'\s+(20\d\d)', t2, re.I)                          # 25 по 31 мая 2026
    if m: y = int(m[4]); mo = mn(m[3]); return [iso(y, mo, int(m[1])), iso(y, mo, int(m[2]))]
    m = re.search(r'(\d{1,2})\s+' + M + r'\s+(20\d\d)\s*г?\.?\s*(?:по|-|–)\s*(\d{1,2})\s+' + M + r'\s+(20\d\d)', t2, re.I)
    if m: return [iso(int(m[3]), mn(m[2]), int(m[1])), iso(int(m[6]), mn(m[5]), int(m[4]))]
    m = re.search(M + r'\s*(20\d\d)?\s*(?:по|-|–)\s*' + M + r'\s*(20\d\d)', t2, re.I)                                # январь - март 2026
    if m: y2 = int(m[4]); y1 = int(m[2]) if m[2] else y2; return [iso(y1, mn(m[1]), 1), iso(y2, mn(m[3]), last(y2, mn(m[3])))]
    m = re.search(r'(\d|I{1,3}|IV)\s*(?:-?й)?\s*кв\w*\.?\s*(20\d\d)', t2, re.I)                                    # 2 квартал 2026
    if m:
        q = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}.get(m[1].upper()) or int(m[1]); y = int(m[2])
        if 1 <= q <= 4: return [iso(y, 3 * q - 2, 1), iso(y, 3 * q, last(y, 3 * q))]
    m = re.search(r'(?<!\d)(\d{1,2})\s+' + M + r'\s+(20\d\d)', t2, re.I)
    if m: d = iso(int(m[3]), mn(m[2]), int(m[1])); return [d, d]
    m = re.search(M + r'(?:\s+(?:месяц|м-ц|мес\.?))?\s*(20\d\d|\d{2}(?=\s*г))', t2, re.I)                          # март 2026 / март 26г / июнь м-ц 2026
    if m: y = yr(m[2]); mo = mn(m[1]); return [iso(y, mo, 1), iso(y, mo, last(y, mo))]
    m = re.search(r'(?<![\w№]\s)(?<!от )(?<!№ )(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{2,4})', t2)
    if m:
        try: d = iso(yr(m[3]), int(m[2]), int(m[1])); return [d, d]
        except Exception: pass
    return None


ev = json.load(open(HERE / 'eval.json', encoding='utf-8'))
for tag, f in (('regex_basic', basic), ('regex_extended', extended)):
    t0 = time.time(); rows = []
    for e in ev:
        try: p = f(e['text'])
        except Exception: p = None
        rows.append({'id': e['id'], 'pred': p})
    json.dump({'tag': tag, 'seconds': max(time.time() - t0, 1e-3), 'n': len(rows), 'rows': rows}, open(HERE / f'pred_{tag}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('ok')
