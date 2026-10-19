# run_deeppavlov.py — DeepPavlov ner_ontonotes_bert_mult (метка DATE) без дообучения + связывание (link.py).
# Запуск: .venv_dp/Scripts/python.exe run_deeppavlov.py   (окружение Python 3.10 с deeppavlov; модель в ~/.deeppavlov)
import json, re, sys, time
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from deeppavlov import build_model
from link import ATOM_RX, LEAD_FROM, link_entities

ev = json.load(open(HERE / 'eval.json', encoding='utf-8'))
m = build_model('ner_ontonotes_bert_mult', download=False, install=False)
CONTRACT = re.compile(r'(?:от|№|n|ДС|Доп\.?|заказ\w*|акт\w*)\s*[\w./-]{0,12}\s*$', re.I)


def entities(text, toks, tags):
    """BIO-теги DeepPavlov -> символьные области DATE -> атомарные даты внутри них (ATOM_RX); даты договоров («от 01.10.2013») отбрасываются."""
    pos = 0; spans = []; cur = None
    for tok, tag in zip(toks, tags):
        i = text.find(tok, pos)
        if i < 0: continue
        pos = i + len(tok)
        if tag == 'B-DATE' or (tag == 'I-DATE' and cur is None): cur = [i, pos]; spans.append(cur)
        elif tag == 'I-DATE': cur[1] = pos
        else: cur = None
    out = []
    for a, b in spans:
        for am in ATOM_RX.finditer(text[a:b]):
            s, e = a + am.start(), a + am.end()
            if re.match(r'\d', am.group(0)) and CONTRACT.search(text[max(0, s - 22):s]): continue
            cue = re.search(r'(?:за|период|в|на)\s*(?:период\s*)?(?:с\s*)?$', text[max(0, s - 12):s], re.I)   # «за март 2026», «период 01.04–30.04»
            out.append((s, e, 1.0 if cue else 0.5))
    return out


t0 = time.time(); rows = []
B = 32
for i in range(0, len(ev), B):
    chunk = ev[i:i + B]; toks, tags = m([e['text'] for e in chunk])
    for e, tk, tg in zip(chunk, toks, tags):
        en = entities(e['text'], tk, tg)
        rows.append({'id': e['id'], 'pred': link_entities(e['text'], en), 'ents': [e['text'][a:b] for a, b, _ in en]})
dt = time.time() - t0
json.dump({'tag': 'deeppavlov_ontonotes_mult', 'seconds': dt, 'n': len(rows), 'rows': rows}, open(HERE / 'pred_deeppavlov_ontonotes_mult.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(f'готово: {len(rows)} строк за {dt:.0f} c')
