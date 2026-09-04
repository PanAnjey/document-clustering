# -*- coding: utf-8 -*-
# star_repr.py -- скрытые состояния Qwen3.5-4B для звёздной карты «до / после».
# Для каждого документа вычищенного пула: тот же промпт классификации, что при
# обучении/оценке, forward без генерации, берём hidden state ПОСЛЕДНЕГО слоя на
# ПОСЛЕДНЕМ токене (это фактически признак, по которому модель называет тип).
# Дважды: база и слитый с LoRA сид 42.
#
#   python star_repr.py            # обе модели
#   python star_repr.py base|ft    # только одну

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common import (ADAPTER_DIR, BASE, MAX_TEXT_CHARS_EVAL, MODEL_PATH, PROMPT,
                    load_pool)

OUT = BASE / 'reports'
OUT.mkdir(exist_ok=True)
FT_SEED = 42


def repr_for(model, tok, docs):
    vecs = []
    for i, d in enumerate(docs):
        enc = tok.apply_chat_template(
            [{'role': 'user', 'content': PROMPT.format(text=d['text'][:MAX_TEXT_CHARS_EVAL])}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors='pt')
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        vecs.append(out.hidden_states[-1][0, -1, :].float().cpu().numpy())
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(docs)}', flush=True)
    return np.stack(vecs).astype(np.float32)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    docs = load_pool()
    print(f'{len(docs)} документов', flush=True)
    # порядок и метки — сохранить рядом
    meta = [{'entity_id': d['entity_id'], 'type': d['type'],
             'group_size': d['group_size']} for d in docs]
    json.dump(meta, open(OUT / 'star_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    if which in ('both', 'base'):
        print('=== база ===', flush=True)
        m = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager')
        m.eval()
        np.save(OUT / 'star_repr_base.npy', repr_for(m, tok, docs))
        del m
        torch.cuda.empty_cache()

    if which in ('both', 'ft'):
        print('=== дообученная (сид 42) ===', flush=True)
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager')
        m = PeftModel.from_pretrained(
            base, str(Path(str(ADAPTER_DIR).format(seed=FT_SEED)) / 'final')).merge_and_unload()
        m.eval()
        np.save(OUT / 'star_repr_ft.npy', repr_for(m, tok, docs))
        del m, base
        torch.cuda.empty_cache()

    print('DONE ->', OUT / 'star_repr_{base,ft}.npy', flush=True)


if __name__ == '__main__':
    main()
