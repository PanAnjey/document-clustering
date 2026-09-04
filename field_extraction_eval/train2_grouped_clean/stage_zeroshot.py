# -*- coding: utf-8 -*-
# stage_zeroshot.py SEED
# Zero-shot база Qwen3.5-4B (без LoRA) на test-выборке сида -- честное «до».
# Копия протокола train2_full/run_zeroshot_4b.py, только выборка -- групповой
# сплит и в вывод добавлен group_size (для разбивки singletons / in-group).

import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common import (MAX_TEXT_CHARS_EVAL, MODEL_PATH, PROMPT, ZS_JSONL, extract_json,
                    load_split, log)


def main():
    seed = int(sys.argv[1])
    out_path = str(ZS_JSONL).format(seed=seed)
    _, test = load_split(seed)
    log(f'[zeroshot seed={seed}] {len(test)} тестовых документов -> {out_path}')

    done = set()
    try:
        for line in open(out_path, encoding='utf-8'):
            done.add(json.loads(line)['entity_id'])
    except FileNotFoundError:
        pass
    if done:
        log(f'[zeroshot seed={seed}] уже посчитано {len(done)}, продолжаю')

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager',
    )
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
                log(f'[zeroshot seed={seed}] {i + 1}/{len(test)}  acc≈{correct / max(total, 1):.3f}')

    log(f'[zeroshot seed={seed}] DONE новых {total}, верных {correct}')


if __name__ == '__main__':
    main()
