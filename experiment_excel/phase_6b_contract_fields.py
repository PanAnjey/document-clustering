# phase_6b_contract_fields.py
# Полный 4-полевой LLM-прогон (ПРЕДМЕТ/ТИП/ОСНОВАНИЕ/КОНТРАГЕНТ) по
# temp_contract_files. Возобновляемый, шардирование по id для 2 GPU.
#
# Фиксы из валидации: ИНН-10/12 чексумма, ИНН-в-тексте, self-фильтр, [ \t] парсер.
#
# Таблица: temp_contract_fields (doc_id PK, predmet, tip, osnovanie,
#   kp_name, kp_inn, kp_kpp, kp_region, kp_self, inn_in_text, raw)
#
# Запуск:
#   python phase_6b_contract_fields.py --gpu cuda:0 --shard 0 --shards 2
#   python phase_6b_contract_fields.py --gpu cuda:1 --shard 1 --shards 2

import argparse
import re
import sys
import time
import traceback

import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL  # noqa: sys.path+utf8
from phase_5v_validate_contract import (
    SYSTEM_PROMPT, _USER_TMPL, _FIELDS, _FIELD_RX,
    inn_valid, is_self, parse_kp, _make_prompts, parse_raw,
)

BATCH = 8


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS temp_contract_fields (
                doc_id INTEGER PRIMARY KEY,
                predmet TEXT, tip TEXT, osnovanie TEXT,
                kp_name TEXT, kp_inn TEXT, kp_kpp TEXT, kp_region TEXT,
                kp_self BOOLEAN, inn_in_text BOOLEAN, raw TEXT
            )""")
    conn.commit()


def fetch_pending(conn, shard, shards, limit=0):
    with conn.cursor() as cur:
        q = f"""
            SELECT f.id, f.txt
            FROM temp_contract_files f
            LEFT JOIN temp_contract_fields r ON r.doc_id = f.id
            WHERE r.doc_id IS NULL AND mod(f.id, {int(shards)}) = {int(shard)}
            ORDER BY f.id
        """
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        return cur.fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    ensure_table(conn)
    rows = fetch_pending(conn, args.shard, args.shards, args.limit)
    if not rows:
        print(f'✅ Нечего обрабатывать (шард {args.shard}/{args.shards})')
        return
    print(f'К обработке: {len(rows)} (шард {args.shard}/{args.shards})')

    import torch
    import hf_summarizer
    from config import cfg
    if args.gpu:
        cfg.LLM_GPU_DEVICE = args.gpu
    hf_summarizer.load_model()
    tokenizer = hf_summarizer._tokenizer_instance
    model = hf_summarizer._model_instance

    t0 = time.time()
    done = 0
    try:
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            ids = [r[0] for r in chunk]
            texts = [r[1] or '' for r in chunk]
            prompts = _make_prompts(texts, tokenizer)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=cfg.LLM_MAX_CONTEXT_TOKENS
                               ).to(cfg.LLM_GPU_DEVICE)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=140, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            out = []
            for i, did in enumerate(ids):
                gen = out_ids[i][inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
                f = parse_raw(raw)
                name, inn, kpp, region = parse_kp(f['КОНТРАГЕНТ'] or '')
                self_flag = is_self(name)
                if self_flag:
                    name = None
                inn_in_text = bool(inn) and inn in texts[i].replace(' ', '')
                out.append((int(did), f['ПРЕДМЕТ'], f['ТИП'], f['ОСНОВАНИЕ'],
                            name, inn, kpp, region, self_flag, inn_in_text,
                            raw[:600]))
            with conn.cursor() as cur:
                execute_values(cur,
                    """INSERT INTO temp_contract_fields
                       (doc_id, predmet, tip, osnovanie, kp_name, kp_inn,
                        kp_kpp, kp_region, kp_self, inn_in_text, raw)
                       VALUES %s ON CONFLICT (doc_id) DO NOTHING""",
                    out, page_size=100)
            conn.commit()
            done += len(chunk)
            dt = time.time() - t0
            rate = done / dt if dt else 0
            print(f'\r  LLM: {done}/{len(rows)} ({rate:.2f} д/с)', end='', flush=True)
    finally:
        print()
        hf_summarizer.unload_model()
    dt = time.time() - t0
    print(f'✅ {done} док. за {dt / 60:.1f} мин ({done / dt:.2f} д/с)')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
