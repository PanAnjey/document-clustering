# -*- coding: utf-8 -*-
# stage_train.py SEED
# LoRA-дообучение Qwen3.5-4B на train-выборке сида. Рецепт БЕЗ ИЗМЕНЕНИЙ
# относительно train2_full/train_lora.py (r=16, alpha=32, 1 эпоха, int8, bs=1,
# grad-accum=8, lr=2e-4 cosine) -- чтобы «до/после» и сравнение со старым
# числом 90.3% были чистыми. Отличается только источник train (групповой сплит,
# без утечки шаблонов и без 52 ошибочных меток).

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

from common import (ADAPTER_DIR, MAX_SEQ_LEN, MAX_TEXT_CHARS_TRAIN, MODEL_PATH,
                    PROMPT, load_split, log)


def main():
    seed = int(sys.argv[1])
    out_dir = Path(str(ADAPTER_DIR).format(seed=seed))
    train, _ = load_split(seed)
    log(f'[train seed={seed}] {len(train)} обучающих примеров -> {out_dir}')

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build_example(doc):
        prompt = PROMPT.format(text=doc['text'][:MAX_TEXT_CHARS_TRAIN])
        completion = json.dumps({'type': doc['type']}, ensure_ascii=False)
        return prompt, completion

    def chat_ids(messages, add_gen):
        return tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=add_gen)['input_ids']

    def tokenize(doc):
        prompt, completion = build_example(doc)
        mp = [{'role': 'user', 'content': prompt}]
        prompt_ids = chat_ids(mp, True)
        full_ids = chat_ids(mp + [{'role': 'assistant', 'content': completion}], False)
        if len(full_ids) > MAX_SEQ_LEN:
            overflow = len(full_ids) - MAX_SEQ_LEN
            d2 = dict(doc)
            d2['text'] = doc['text'][:max(150, MAX_TEXT_CHARS_TRAIN - overflow * 3)]
            prompt, completion = build_example(d2)
            mp = [{'role': 'user', 'content': prompt}]
            prompt_ids = chat_ids(mp, True)
            full_ids = chat_ids(mp + [{'role': 'assistant', 'content': completion}], False)
            full_ids = full_ids[:MAX_SEQ_LEN]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(full_ids))):
            labels[i] = -100
        return {'input_ids': full_ids, 'labels': labels,
                'attention_mask': [1] * len(full_ids)}

    base = Dataset.from_list(train)
    ds = base.map(tokenize, remove_columns=base.column_names)
    lens = sorted(len(x) for x in ds['input_ids'])
    log(f'[train seed={seed}] длины (мин/медиана/макс): '
        f'{lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}')

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        attn_implementation='eager',
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    ))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy='epoch', report_to=[],
        optim='adamw_torch', warmup_ratio=0.03, lr_scheduler_type='cosine',
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(
                          tok, model=model, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(str(out_dir / 'final'))
    tok.save_pretrained(str(out_dir / 'final'))
    log(f'[train seed={seed}] DONE -> {out_dir / "final"}')


if __name__ == '__main__':
    main()
