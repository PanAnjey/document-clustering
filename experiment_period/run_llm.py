# run_llm.py — извлечение периода генеративной моделью.
#   transformers:  python run_llm.py --model D:/MODELS/Transformers/Qwen_Qwen3.5-9B --tag qwen35_9b --gpu cuda:1
#   OpenAI-совм.:  python run_llm.py --api http://localhost:1234/v1 --model qwen3.8-27b --tag qwen38_27b
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from prompt import SYSTEM, USER, parse   # noqa

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--api', default=''); ap.add_argument('--gpu', default='cuda:0'); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--eval', default='eval.json'); ap.add_argument('--out-suffix', default=''); ap.add_argument('--user-suffix', default=''); ap.add_argument('--max-tokens', type=int, default=64)
a = ap.parse_args()
rows = json.load(open(HERE / a.eval, encoding='utf-8'))
if a.limit: rows = rows[:a.limit]
res = {}; t0 = time.time()

if a.api:
    def call(r):
        body = {'model': a.model, 'max_tokens': a.max_tokens, 'temperature': 0,
                'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(text=r['text']) + a.user_suffix}]}
        req = urllib.request.Request(a.api.rstrip('/') + '/chat/completions', data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                     headers={'Content-Type': 'application/json; charset=utf-8'})
        for k in range(5):
            try:
                m = json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]['message']; return m.get('content') or ''
            except Exception as e:
                if k == 4: raise
                time.sleep(10)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (r, raw) in enumerate(zip(rows, ex.map(call, rows))):
            res[r['id']] = raw
            if i % 50 == 0: print(i, f'{time.time()-t0:.0f} c', flush=True)
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.tensor([0], device=a.gpu); torch.cuda.synchronize(a.gpu)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.gpu, trust_remote_code=True).eval()
    print(f'загружена {time.time()-t0:.0f} c', flush=True); t0 = time.time()

    def build(text):
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(text=text)}]
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['text']))
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        enc = tok([build(rows[i]['text']) for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to(a.gpu)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, g[:, enc['input_ids'].shape[1]:]):
            res[rows[i]['id']] = tok.decode(seq, skip_special_tokens=True)
        if (b // a.batch) % 5 == 0: print(b + len(idx), f'{time.time()-t0:.0f} c', flush=True)

dt = time.time() - t0
out = [{'id': r['id'], 'raw': res[r['id']], 'pred': parse(res[r['id']])} for r in rows]
json.dump({'tag': a.tag, 'seconds': dt, 'n': len(out), 'rows': out}, open(HERE / f'pred_{a.tag}{a.out_suffix}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(f'готово {a.tag}: {len(out)} строк, {dt:.0f} c ({len(out)/dt:.2f} стр/с)')
