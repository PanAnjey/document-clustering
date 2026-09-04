# phase_x6_embed_pdf.py
# Этап X6: эмбеддинги LLM-предметов неформализованных документов
# (temp_llm_subject_pdftext / temp_llm_subject_pdftables) той же моделью и
# параметрами, что XML (nomic-embed-text, bf16, L2-norm) — для совместной
# карты «галактик» XML + PDF.
#
# Таблица: temp_pdf_embeddings (source TEXT, doc_id INT, embedding BYTEA,
#                             PRIMARY KEY (source, doc_id))
#
# Запуск:
#   python phase_x6_embed_pdf.py --device cuda:0 --shards 2 --shard-id 0
#   python phase_x6_embed_pdf.py --device cuda:1 --shards 2 --shard-id 1

import argparse
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import DB_URL  # noqa: E402
from xml_embed_helper import encode_texts  # noqa: E402

T_PDF_EMB = 'temp_pdf_embeddings'
SOURCES = [
    ('pdftext', 'temp_llm_subject_pdftext'),
    ('pdftables', 'temp_llm_subject_pdftables'),
]
CHUNK = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard-id', type=int, default=0)
    args = ap.parse_args()
    if args.device:
        from config import cfg
        cfg.EMB_GPU_DEVICE = args.device
        print(f"device: {args.device} (shard {args.shard_id}/{args.shards})")

    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {T_PDF_EMB} (
            source TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            embedding BYTEA NOT NULL,
            PRIMARY KEY (source, doc_id))""")
    conn.commit()

    for src, table in SOURCES:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT s.train_id, s.predmet FROM {table} s
                WHERE s.predmet IS NOT NULL
                  AND (s.train_id %% %s) = %s
                  AND NOT EXISTS (SELECT 1 FROM {T_PDF_EMB} e
                                  WHERE e.source = %s AND e.doc_id = s.train_id)
                ORDER BY s.train_id""", (args.shards, args.shard_id, src))
            rows = cur.fetchall()
        print(f"{src}: к эмбеддингу {len(rows)}")
        t0 = time.time()
        done = 0
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start:start + CHUNK]
            ids = [r[0] for r in chunk]
            emb = encode_texts([r[1] or '' for r in chunk], batch_size=64)
            payload = [(src, int(ids[k]), emb[k].astype(np.float32).tobytes())
                       for k in range(len(chunk))]
            with conn.cursor() as cur:
                execute_values(cur,
                               f"INSERT INTO {T_PDF_EMB} (source, doc_id, embedding) "
                               f"VALUES %s ON CONFLICT (source, doc_id) DO NOTHING",
                               payload, page_size=1000)
            conn.commit()
            done += len(chunk)
            print(f"  {src}: {done}/{len(rows)} | {done/(time.time()-t0):.0f} док/с",
                  flush=True)
    conn.close()
    print('Phase X6 embed completed')


if __name__ == '__main__':
    main()
