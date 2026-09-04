# -*- coding: utf-8 -*-
# stage_eval.py SEED
# Оценка дообученной модели сида на его test-выборке. Протокол идентичен
# stage_zeroshot.py (тот же промпт, greedy, max_new_tokens=30) -- честное
# «после». Копия train2_full/run_finetuned_eval.py.

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common import (ADAPTER_DIR, FT_JSONL, MAX_TEXT_CHARS_EVAL, MODEL_PATH, PROMPT,
                    extract_json, load_split, log)


def main():
    seed = int(sys.argv[1])
    adapter = Path(str(ADAPTER_DIR).format(seed=seed)) / 'final'
    out_path = str(FT_JSONL).format(seed=seed)
    _, test = load_split(seed)
    log(f'[eval seed={seed}] {len(test)} тестовых документов, адаптер {adapter}')

    done = set()
    try:
        for line in open(out_path, encoding='utf-8'):
            done.add(json.loads(line)['entity_id'])
    except FileNotFoundError:
        pass
    if done:
        log(f'[eval seed={seed}] уже посчитано {len(done)}, продолжаю')

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager')
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.eval()

    correct = total = 0
    with open(out_path, 'a', encoding='utf-8') as out_f:
        for i, d in enumerate(test):
            if d['entity_id'] in done:
                continue
            prompt = PROMPT.format(text=d['text'][:MAX_TEXT_CHARS_EVAL])
            enc = tok.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            parsed = extract_json(gen)
            pred = parsed.get('type') if parsed else None
            ok = pred == d['type']
            total += 1
            correct += int(ok)
            out_f.write(json.dumps({
                'entity_id': d['entity_id'], 'true_type': d['type'], 'pred_type': pred,
                'group_size': d['group_size'], 'raw': gen if parsed is None else None,
            }, ensure_ascii=False) + '\n')
            out_f.flush()
            if (i + 1) % 25 == 0:
                log(f'[eval seed={seed}] {i + 1}/{len(test)}  acc≈{correct / max(total, 1):.3f}')

    log(f'[eval seed={seed}] DONE новых {total}, верных {correct}')


if __name__ == '__main__':
    main()
