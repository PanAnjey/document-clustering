# gold.py — правило-основанный эталон периода оказания услуги + кандидаты-фрагменты (для эмбеддинг-модели).
# Период = (start, end) в ISO. Месяц -> весь месяц, квартал, год, "с X по Y", одиночная дата -> start=end. Нет периода -> None.
import calendar, re
MONTHS = {'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'ма': 5, 'июн': 6, 'июл': 7, 'август': 8, 'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12}
MRX = r'(январ[а-яё]*|феврал[а-яё]*|март[а-яё]*|апрел[а-яё]*|мая|май|мае|июн[а-яё]*|июл[а-яё]*|август[а-яё]*|сентябр[а-яё]*|октябр[а-яё]*|ноябр[а-яё]*|декабр[а-яё]*)'
YRX = r'(20\d\d)'
DRX = r'(\d{1,2})\s?\.\s?(\d{1,2})\s?\.\s?(\d{4}|\d{2})'


def mnum(w):
    w = w.lower()
    return 5 if w in ('мая', 'май', 'мае') else next(v for k, v in MONTHS.items() if k != 'ма' and w.startswith(k))


def iso(y, m, d):
    return f'{y:04d}-{m:02d}-{d:02d}'


def full_month(y, m):
    return iso(y, m, 1), iso(y, m, calendar.monthrange(y, m)[1])


def yy(s):
    y = int(s); return y + 2000 if y < 100 else y


def spans(t):
    """Все кандидаты периода в строке: список (фрагмент, (start,end), kind)."""
    out = []; used = []
    SEP = r'\s*(?:г(?:ода)?\.?)?\s*(?:\d{1,2}:\d{2})?\s*[»"]?\s*(?:по|[-–—])\s*[_«"]*\s*'
    def add(m, val, kind):
        if any(not (m.end() <= a or m.start() >= b) for a, b in used): return
        pre = t[max(0, m.start() - 12):m.start()].lower()
        if re.search(r'(?:от|№|n)\s*$', pre) and kind == 'date': return    # дата договора/документа
        used.append((m.start(), m.end())); out.append((m.group(0).strip(), val, kind))
    D = r'[«"]?(\d{1,2})[»"]?'
    Y = r'(?:\s*(20\d\d|\d{2}(?!\d)))?'
    for m in re.finditer(r'(?:с\s*)?' + DRX + SEP + DRX, t, re.I):
        try: add(m, (iso(yy(m.group(3)), int(m.group(2)), int(m.group(1))), iso(yy(m.group(6)), int(m.group(5)), int(m.group(4)))), 'range')
        except ValueError: pass
    # с «01» апреля 2026 г. по «30» апреля 2026 г.
    # 01.12-31.12.2025 / с 01.03 по 31.03.2026
    for m in re.finditer(r'(?:с\s*)?(\d{1,2})\.(\d{1,2})\.?' + SEP + DRX, t, re.I):
        try:
            y = yy(m.group(5)); add(m, (iso(y, int(m.group(2)), int(m.group(1))), iso(y, int(m.group(4)), int(m.group(3)))), 'range')
        except ValueError: pass
    # Дата начала: X ... Дата окончания: Y
    for m in re.finditer(r'дата\s+нач\w+\s*:?\s*' + DRX + r'.{0,25}?дата\s+окончания\s*:?\s*' + DRX, t, re.I):
        try: add(m, (iso(yy(m.group(3)), int(m.group(2)), int(m.group(1))), iso(yy(m.group(6)), int(m.group(5)), int(m.group(4)))), 'range')
        except ValueError: pass
    for m in re.finditer(r'(?:с\s*)?' + D + r'\s*' + MRX + r'\s*(20\d\d)?' + SEP + D + r'\s*' + MRX + r'\s*(20\d\d)', t, re.I):
        y2 = int(m.group(6)); y1 = int(m.group(3)) if m.group(3) else y2
        add(m, (iso(y1, mnum(m.group(2)), int(m.group(1))), iso(y2, mnum(m.group(5)), int(m.group(4)))), 'range')
    # с февраля 2026 по май 2026 / январь-март 2026
    for m in re.finditer(r'(?:с\s*)?' + MRX + r'\s*(20\d\d)?\s*(?:г\.?)?' + r'\s*(?:по|[-–—])\s*' + MRX + r'\s*(20\d\d)', t, re.I):
        y2 = int(m.group(4)); y1 = int(m.group(2)) if m.group(2) else y2
        add(m, (full_month(y1, mnum(m.group(1)))[0], full_month(y2, mnum(m.group(3)))[1]), 'range')
    for m in re.finditer(r'(\d{1,2})\s*(?:по|[-–—])\s*(\d{1,2})\s+' + MRX + r'\s*(20\d\d)', t, re.I):
        add(m, (iso(int(m.group(4)), mnum(m.group(3)), int(m.group(1))), iso(int(m.group(4)), mnum(m.group(3)), int(m.group(2)))), 'range')
    for m in re.finditer(r'(\d|I{1,3}|IV)\s*(?:-?й)?\s*кв(?:артал|\.)?[а-я]*\s*(20\d\d)', t, re.I):
        q = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}.get(m.group(1).upper()) or int(m.group(1)); y = int(m.group(2))
        if 1 <= q <= 4: add(m, (iso(y, 3 * q - 2, 1), full_month(y, 3 * q)[1]), 'month')
    for m in re.finditer(D + r'\s+' + MRX + r'\s*(20\d\d)', t, re.I):
        d = iso(int(m.group(3)), mnum(m.group(2)), int(m.group(1))); add(m, (d, d), 'date')
    # месяц [месяц|м-ц] год(4 или 2 цифры)
    for m in re.finditer(MRX + r'(?:\s+(?:месяц|м-ц|мес\.?))?\s*(20\d\d|\d{2}(?=\s*г))', t, re.I):
        add(m, full_month(yy(m.group(2)), mnum(m.group(1))), 'month')
    for m in re.finditer(DRX, t):
        try: d = iso(yy(m.group(3)), int(m.group(2)), int(m.group(1))); add(m, (d, d), 'date')
        except ValueError: pass
    for m in re.finditer(r'(?:за|на|в)\s+(20\d\d)\s*(?:год|г)', t, re.I):
        y = int(m.group(1)); add(m, (iso(y, 1, 1), iso(y, 12, 31)), 'month')
    return sorted(out, key=lambda x: t.find(x[0]))


def gold(t):
    """Приоритет: диапазон > «месяц/квартал/год» > одиночная дата. Возвращает ((start,end)|None, кандидаты)."""
    sp = spans(t)
    if not sp: return None, []
    for kind in ('range', 'month', 'date'):
        c = [s for s in sp if s[2] == kind]
        if c: return c[0][1], sp


def gold_frag(t):
    """(start, end, char_from, char_to) выбранного фрагмента или None."""
    v, sp = gold(t)
    if not v: return None
    frag = next(s[0] for s in sp if s[1] == v)
    i = t.find(frag)
    return v, i, i + len(frag)
