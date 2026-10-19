# run_rubert.py — redmadrobot-rnd/rubert-base-pii-ner + связывание сущностей-дат в период.
# У исходной модели класса DATE нет (проверено: 0 дат на eval.json), поэтому энкодер дообучается токенной головой O/B-DATE/I-DATE,
# метки: атомарные даты (dd.mm.yyyy, «месяц год», «квартал год», «25» перед «по …» и т.п.), входящие в выбранный эталоном период; даты договоров -> O.
# Связывание в код вынесено отдельно (link_entities): две даты через «по»/«–» -> диапазон, одиночная -> месяц/день.
import calendar, json, random, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gold import gold_frag, MRX, mnum   # noqa
PATH = 'D:/MODELS/Transformers/redmadrobot-rnd_rubert-base-pii-ner'
GPU = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
random.seed(5); torch.manual_seed(5)

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


def atoms(t):
    out = []
    for m in ATOM_RX.finditer(t):
        if LEAD_FROM.search(t[max(0, m.start() - 6):m.start()]) and re.match(r'\d', m.group(0)): continue
        out.append((m.start(), m.end()))
    return out


# ---------- разметка
def label_chars(t, dates):
    """dates: список (a,b) символьных диапазонов-DATE -> список токен-независимых меток по символам"""
    lab = np.zeros(len(t), dtype=np.int8)
    for a, b in dates: lab[a:b] = 1
    return lab


def silver_dates(t):
    g = gold_frag(t)
    if not g: return []
    _, a, b = g
    return [(s, e) for s, e in atoms(t) if s < b and e > a]


MON_G = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
MON_N = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']


def synth(base):
    """Синтетика с диапазонами: вставляем шаблон периода в строку без периода. Возвращает (text, [(a,b)...])."""
    y = random.choice([2025, 2026]); m = random.randint(1, 12); m2 = min(12, m + random.randint(0, 3))
    d1 = random.randint(1, 20); d2 = random.randint(d1 + 1, calendar.monthrange(y, m)[1])
    ld = calendar.monthrange(y, m2)[1]
    f = lambda d, mm, yy: f'{d:02d}.{mm:02d}.{yy}'
    T = random.choice(range(9))
    if T == 0: s = f'за период с {f(1, m, y)} по {f(ld if m2 == m else calendar.monthrange(y, m)[1], m2, y)}'; parts = [f(1, m, y), f(ld if m2 == m else calendar.monthrange(y, m)[1], m2, y)]
    elif T == 1: parts = [f(d1, m, y), f(d2, m, y)]; s = f'({parts[0]} - {parts[1]})'
    elif T == 2: parts = [f'{d1}', f'{d2} {MON_G[m-1]} {y}']; s = f'с {parts[0]} по {parts[1]}'
    elif T == 3: parts = [f'{MON_G[m-1]} {y}', f'{MON_N[m2-1]} {y}']; s = f'за период с {parts[0]} по {parts[1]}'
    elif T == 4: parts = [f'«{d1:02d}» {MON_G[m-1]} {y}', f'«{d2:02d}» {MON_G[m-1]} {y}']; s = f'в период с {parts[0]} г. по {parts[1]} г.'
    elif T == 5: parts = [f'{d1:02d}.{m:02d}', f(d2, m, y)]; s = f'за период {parts[0]}-{parts[1]}'
    elif T == 6: parts = [f(d1, m, y)]; s = f'на мероприятие {parts[0]}'
    elif T == 7: parts = [f'{MON_N[m-1]} {y}']; s = f'за {parts[0]}'
    else: parts = [f(d1, m, y), f(d2, m, y)]; s = f'отчетный период с {parts[0]} по {parts[1]}'
    contract = f'по договору № {random.randint(1,999)}/{random.randint(10,99)} от {f(random.randint(1,28), random.randint(1,12), random.randint(2015,2025))}' if random.random() < .5 else ''
    pre = base.rstrip(' .,;') + ' '
    if contract: pre += contract + ' '
    text = pre + s
    dates = []; pos = len(pre)
    for p in parts:
        i = text.find(p, pos); dates.append((i, i + len(p))); pos = i + len(p)
    return text, dates


tok = AutoTokenizer.from_pretrained(PATH)
tr = json.load(open(HERE / 'train.json', encoding='utf-8'))
samples = [(x['text'], silver_dates(x['text'])) for x in tr]
negs = [x['text'] for x in tr if not x['span']]
samples += [synth(random.choice(negs)) for _ in range(1400)]
random.shuffle(samples)
print('обучающих текстов', len(samples), 'с датами', sum(1 for _, d in samples if d), flush=True)


def encode(texts, dates_list=None):
    enc = tok(texts, truncation=True, max_length=192, padding=True, return_offsets_mapping=True, return_tensors='pt')
    if dates_list is None: return enc
    labs = np.full(enc['input_ids'].shape, -100)
    for i, (t, dts) in enumerate(zip(texts, dates_list)):
        cl = label_chars(t, dts); prev = 0
        for j, (a, b) in enumerate(enc['offset_mapping'][i].tolist()):
            if b <= a: continue
            if cl[a:b].any():
                labs[i, j] = 2 if (prev and j > 0 and labs[i, j - 1] in (1, 2)) else 1
                prev = 1
            else: labs[i, j] = 0; prev = 0
    return enc, torch.tensor(labs)


model = AutoModelForTokenClassification.from_pretrained(PATH, num_labels=3, ignore_mismatched_sizes=True,
                                                        id2label={0: 'O', 1: 'B-DATE', 2: 'I-DATE'}, label2id={'O': 0, 'B-DATE': 1, 'I-DATE': 2}).to(GPU)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
EP, BS = 3, 32; steps = EP * (len(samples) // BS + 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-5, total_steps=steps, pct_start=.1)
t0 = time.time(); model.train()
for ep in range(EP):
    random.shuffle(samples); tot = 0
    for i in range(0, len(samples), BS):
        b = samples[i:i + BS]; enc, lab = encode([t for t, _ in b], [d for _, d in b])
        out = model(input_ids=enc['input_ids'].to(GPU), attention_mask=enc['attention_mask'].to(GPU), labels=lab.to(GPU))
        out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); tot += out.loss.item()
    print(f'эпоха {ep+1}: loss {tot/(len(samples)/BS):.4f}, {time.time()-t0:.0f} c', flush=True)
model.eval()


# ---------- инференс + связывание
def entities(texts, bs=32):
    res = []
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]; enc = encode(b)
        with torch.no_grad():
            lg = model(input_ids=enc['input_ids'].to(GPU), attention_mask=enc['attention_mask'].to(GPU)).logits.softmax(-1).cpu().numpy()
        for k, t in enumerate(b):
            ents = []; cur = None
            for j, (a, e) in enumerate(enc['offset_mapping'][k].tolist()):
                if e <= a: continue
                p = lg[k, j]; lab = int(p.argmax())
                if lab == 1 or (lab == 2 and cur is None): cur = [a, e, [p[1] + p[2]]]; ents.append(cur)
                elif lab == 2: cur[1] = e; cur[2].append(p[1] + p[2])
                else: cur = None
            res.append([(a, e, float(np.mean(s))) for a, e, s in ents])
    return res


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


ev = json.load(open(HERE / 'eval.json', encoding='utf-8'))
t1 = time.time(); E = entities([e['text'] for e in ev]); dt = time.time() - t1
out = [{'id': e['id'], 'pred': link_entities(e['text'], en), 'ents': [e['text'][a:b] for a, b, _ in en]} for e, en in zip(ev, E)]
json.dump({'tag': 'rubert_pii_ft', 'seconds': dt, 'n': len(out), 'rows': out}, open(HERE / 'pred_rubert_pii_ft.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
json.dump({'tag': 'rubert_pii_asis', 'seconds': dt, 'n': len(out), 'rows': [{'id': e['id'], 'pred': None} for e in ev]},
          open(HERE / 'pred_rubert_pii_asis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(f'готово: {len(out)} строк за {dt:.1f} c')
