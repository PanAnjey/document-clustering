# run_giga.py — Giga-Embeddings-480M как извлекатель периода. Генерировать он не умеет (энкодер), поэтому:
# замороженный энкодер -> скрытые состояния токенов -> линейный тегер «токен входит в период» (логрегрессия), обучен на силвер-разметке train.json.
# Выбранный тегером фрагмент приводится к датам тем же детерминированным нормализатором (gold.spans), что и у эталона -> сравнение честное по выбору фрагмента.
# Режимы: giga_tag (тегер), giga_zs (zero-shot: кандидат-фрагменты по сходству с запросом, без обучения).
import json, sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gold import spans   # noqa
PATH = 'D:/MODELS/Transformers/ai-sage_Giga-Embeddings-instruct-480M-0826'
GPU = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
mdl = AutoModel.from_pretrained(PATH, trust_remote_code=True).to(GPU).eval()


def hidden(texts, bs=32):
    """-> список (H[ntok,d] float16, offsets[ntok,2]) по каждой строке"""
    out = []
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]
        enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors='pt', return_offsets_mapping=True)
        off = enc.pop('offset_mapping'); enc = enc.to(GPU)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            h = mdl(**enc).last_hidden_state.float().cpu().numpy()
        am = enc['attention_mask'].cpu().numpy()
        for j in range(len(b)):
            k = am[j].astype(bool); out.append((h[j][k].astype(np.float32), off[j].numpy()[k]))
    return out


def sent_emb(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors='pt').to(GPU)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            h = mdl(**enc).last_hidden_state.float()
        m = enc['attention_mask'].unsqueeze(-1).float(); e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


ev = json.load(open(HERE / 'eval.json', encoding='utf-8'))
tr = json.load(open(HERE / 'train.json', encoding='utf-8'))
t0 = time.time()
# --- обучение тегера
Htr = hidden([x['text'] for x in tr])
X, y = [], []
for x, (H, off) in zip(tr, Htr):
    lab = np.zeros(len(H), dtype=int)
    if x['span']:
        a, b = x['span']; lab = ((off[:, 0] < b) & (off[:, 1] > a) & (off[:, 1] > off[:, 0])).astype(int)
    X.append(H); y.append(lab)
clf = LogisticRegression(max_iter=300, C=1.0, class_weight='balanced').fit(np.concatenate(X), np.concatenate(y))
print(f'тегер обучен {time.time()-t0:.0f} c, токенов {sum(len(v) for v in y)}', flush=True)


def to_period(text, a, b):
    """символьный фрагмент [a,b) -> даты (нормализатор gold.spans по расширенному окну)"""
    win = text[max(0, a - 12):b + 14]
    sp = spans(win)
    if not sp: return None
    # фрагмент должен пересекаться с выбранным
    for frag, val, kind in sp:
        i = win.find(frag) + max(0, a - 12)
        if i < b and i + len(frag) > a: return list(val)
    return None


def run_tag(thr):
    t1 = time.time(); H = hidden([e['text'] for e in ev]); res = {}
    for e, (h, off) in zip(ev, H):
        p = clf.predict_proba(h)[:, 1]; ok = (p >= thr) & (off[:, 1] > off[:, 0])
        # непрерывные группы токенов -> лучший по средней вероятности
        best = None; i = 0
        while i < len(ok):
            if ok[i]:
                j = i
                while j + 1 < len(ok) and ok[j + 1]: j += 1
                sc = p[i:j + 1].mean()
                if best is None or sc > best[0]: best = (sc, int(off[i, 0]), int(off[j, 1]))
                i = j + 1
            else: i += 1
        res[e['id']] = to_period(e['text'], best[1], best[2]) if best else None
    return res, time.time() - t1


for thr in (0.5,):
    res, dt = run_tag(thr)
    out = [{'id': e['id'], 'pred': res[e['id']]} for e in ev]
    json.dump({'tag': 'giga480m_tagger', 'seconds': dt, 'n': len(out), 'rows': out}, open(HERE / 'pred_giga480m_tagger.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print(f'tagger thr={thr}: {dt:.1f} c')

# --- zero-shot: кандидаты (грубый детектор) + сходство контекста с запросом
Q = sent_emb(['период оказания услуги: за какой период времени оказана услуга или выполнены работы'])[0]
t1 = time.time(); res = {}; cand_txt = []; cand_own = []
for e in ev:
    t = e['text']
    for frag, val, kind in spans(t):
        i = t.find(frag); cand_txt.append(t[max(0, i - 40):i + len(frag) + 10]); cand_own.append((e['id'], val))
E = sent_emb(cand_txt); sims = E @ Q
best = {}
for (rid, val), s in zip(cand_own, sims):
    if rid not in best or s > best[rid][0]: best[rid] = (s, val)
# порог «нет периода»: не нужен, если кандидатов нет; иначе берём лучший (кандидат есть -> период есть)
out = [{'id': e['id'], 'pred': list(best[e['id']][1]) if e['id'] in best else None} for e in ev]
dt = time.time() - t1
json.dump({'tag': 'giga480m_zeroshot', 'seconds': dt, 'n': len(out), 'rows': out}, open(HERE / 'pred_giga480m_zeroshot.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(f'zero-shot: {dt:.1f} c')
