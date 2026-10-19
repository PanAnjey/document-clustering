# link.py — общие функции связывания (вынесены из run_rubert.py): атомы дат, разбор, нормализация, сборка периода.
import calendar, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gold import MRX, mnum   # noqa

SEP = r'(?:г(?:ода)?\.?)?\s*[»"]?\s*(?:по|до|[-–—])\s*[_«"]*\s*(?:\d{1,2}:\d{2}(?::\d{2})?)?\s*'
D4 = r'(?:20\d\d|\d{2}(?=\s*г))'
ATOMS = [
    r'\d{1,2}\s?\.\s?\d{1,2}\s?\.\s?(?:\d{4}|\d{2})',                         # полная дата
    r'(?:\d|I{1,3}|IV)\s*(?:-?й)?\s*кв(?:артал[а-я]*|\.)\s*20\d\d',           # квартал
    r'[«"]?\d{1,2}[»"]?\s+' + MRX + r'(?:\s*20\d\d)?',                           # 25 мая [2026]
    MRX + r'(?:\s+(?:месяц|м-ц|мес\.?))?\s*' + D4,                               # май 2026
    r'\d{1,2}\.\d{1,2}(?=\.?\s*' + SEP + r'\d)',                                 # 01.03 по 31.03.2026
    r'\d{1,2}(?=\s*' + SEP + r'\d{1,2}\s+' + MRX + r')',                         # 25 по 31 мая
    MRX + r'(?=\s*' + SEP + MRX + r'\s*20\d\d)',                                 # января – марта 2026
]
ATOM_RX = re.compile('|'.join(f'(?:{a})' for a in ATOMS), re.I)
LEAD_FROM = re.compile(r'(?:от|№|n)\s*$', re.I)


LEAD_FROM = re.compile(r'(?:от|№|n)\s*$', re.I)


def parse_atom(s):
    """текст сущности -> dict(y,m,d,q) с отсутствующими полями None"""
    r = dict(y=None, m=None, d=None, q=None)
    m = re.search(r'(\d{1,2})\s?\.\s?(\d{1,2})(?:\s?\.\s?(\d{4}|\d{2}))?', s)
    if m:
        r.update(d=int(m.group(1)), m=int(m.group(2)))
        if m.group(3): y = int(m.group(3)); r['y'] = y + 2000 if y < 100 else y
        return r
    m = re.search(r'(\d|I{1,3}|IV)\s*(?:-?й)?\s*кв', s, re.I)
    if m:
        r['q'] = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}.get(m.group(1).upper()) or int(m.group(1))
        y = re.search(r'20\d\d', s); r['y'] = int(y.group(0)) if y else None
        return r
    mm = re.search(MRX, s, re.I)
    if mm: r['m'] = mnum(mm.group(1))
    y = re.search(r'(20\d\d|\b\d{2}\b(?=\s*г))', s)
    if y and mm: yy = int(y.group(1)); r['y'] = yy + 2000 if yy < 100 else yy
    d = re.match(r'\W*(\d{1,2})\b', s)
    if d and (mm or not re.search(r'\d\s?\.', s)): r['d'] = int(d.group(1))
    return r


def span_of(r, end):
    y, m, d, q = r['y'], r['m'], r['d'], r['q']
    if q: return (y, 3 * q - 2, 1) if not end else (y, 3 * q, calendar.monthrange(y, 3 * q)[1])
    if d is None: d = 1 if not end else calendar.monthrange(y, m)[1]
    return (y, m, d)


def iso(t): return f'{t[0]:04d}-{t[1]:02d}-{t[2]:02d}'


def link_entities(text, ents):
    """Сущности-даты -> период. «с X по Y»: две соседние даты, между ними только «по/до/–» (и ничего содержательного) -> диапазон;
    иначе одиночная сущность (высшая уверенность): месяц/квартал целиком или конкретный день."""
    ents = sorted(e for e in ents if re.search(r'^\d{1,2}$|\d\s?\.\s?\d|' + MRX + r'|кв', text[e[0]:e[1]], re.I))   # отбрасываем «09:42» и прочее без даты
    if not ents: return None
    # одна сущность, внутри которой уже диапазон месяцев: «январь - март 2026»
    for a, e, p in ents:
        m = re.fullmatch(MRX + r'\s*(?:по|до|[-–—])\s*' + MRX + r'\s*(20\d\d)', text[a:e].strip(), re.I)
        if m: y = int(m.group(3)); return [iso((y, mnum(m.group(1)), 1)), iso((y, mnum(m.group(2)), calendar.monthrange(y, mnum(m.group(2)))[1]))]
    for (a1, e1, p1), (a2, e2, p2) in zip(ents, ents[1:]):
        gap = text[e1:a2]
        if len(gap) <= 40 and (re.fullmatch(r'\s*(?:г(?:ода)?\.?)?\s*[»"]?\s*(?:\d{1,2}:\d{2}(?::\d{2})?)?\s*(?:по|до|[-–—])\s*[_«"]*\s*(?:\d{1,2}:\d{2}(?::\d{2})?)?\s*', gap)
                               or re.fullmatch(r'[\s\w.:г]*окончани\w*\s*:?\s*', gap, re.I)):    # «Дата начала: X Дата окончания: Y»
            A, B = parse_atom(text[a1:e1]), parse_atom(text[a2:e2])
            for k in ('y', 'm'):                              # недостающее у начала берём у конца
                if A[k] is None and A['q'] is None: A[k] = B[k]
            if A['y'] is None: A['y'] = B['y']
            if B['y'] is None: B['y'] = A['y']
            try:
                if A['y'] and B['y'] and (A['m'] or A['q']) and (B['m'] or B['q']): return [iso(span_of(A, False)), iso(span_of(B, True))]
            except Exception: pass
    best = max(ents, key=lambda x: x[2])
    A = parse_atom(text[best[0]:best[1]])
    try:
        if A['y'] and (A['m'] or A['q']):
            if A['d'] and A['m']: s = iso((A['y'], A['m'], A['d'])); return [s, s]
            return [iso(span_of(A, False)), iso(span_of(A, True))]
    except Exception: pass
    return None


