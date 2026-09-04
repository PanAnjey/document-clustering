# phase_x4_clean_subject.py
# Этап X4: V3 — тематический текст, очищенный от договорных ссылок и
# календарных периодов (strip_contract_refs), эмбеддинг и присвоение темам.
#
# Мотивация: V2 показал, что 58.8% XML уходит в тему 26 («вознаграждение
# по договору») из-за boilerplate «по Договору №… / согласно ДС №…» в
# наименованиях позиций. Очистка должна перераспределить эти документы
# в содержательные темы (9/10/7/0/17…).
#
# Таблицы:
#   temp_xml_embeddings.subject_clean_embedding BYTEA (добавляется колонка)
#   temp_xml_subject_clean_assign (xml_id PK, assigned_theme, similarity, is_unknown)
#
# Запуск: python phase_x4_clean_subject.py [--skip-embed] [--limit N]

import argparse
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    DB_URL, T_CANONICAL, T_EMBEDDINGS, SIMILARITY_THRESHOLD, CENTROIDS_TABLE,
)
from xml_canonicalizer import strip_contract_refs  # noqa: E402
from xml_embed_helper import encode_texts  # noqa: E402
from phase_x2_embed_assign import apply_marked_override  # noqa: E402

T_CLEAN_ASSIGN = 'temp_xml_subject_clean_assign'
FETCH_CHUNK = 4000


def run_embed(conn, limit=0, shards=1, shard_id=0):
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {T_EMBEDDINGS} "
                    f"ADD COLUMN IF NOT EXISTS subject_clean_embedding BYTEA")
        cur.execute(f"""
            SELECT c.id, c.subject_text FROM {T_CANONICAL} c
            JOIN {T_EMBEDDINGS} e ON e.xml_id = c.id
            WHERE c.parse_ok AND c.subject_text IS NOT NULL
              AND (c.id %% %s) = %s
              AND e.subject_clean_embedding IS NULL
            ORDER BY c.id
        """ + (f" LIMIT {limit}" if limit else ""), (shards, shard_id))
        rows = cur.fetchall()
    conn.commit()
    if not rows:
        print("embed[V3]: нечего делать")
        return
    print(f"embed[V3]: {len(rows)} документов")
    t0 = time.time()
    done = 0
    for start in range(0, len(rows), FETCH_CHUNK):
        chunk = rows[start:start + FETCH_CHUNK]
        ids = [r[0] for r in chunk]
        cleaned = [strip_contract_refs(r[1] or '') for r in chunk]
        emb = encode_texts(cleaned, batch_size=64)
        payload = [(emb[k].astype(np.float32).tobytes(), int(ids[k]))
                   for k in range(len(chunk))]
        with conn.cursor() as cur:
            execute_values(cur,
                           f"UPDATE {T_EMBEDDINGS} SET subject_clean_embedding = data.emb "
                           f"FROM (VALUES %s) AS data(emb, xml_id) "
                           f"WHERE {T_EMBEDDINGS}.xml_id = data.xml_id",
                           payload, page_size=500)
        conn.commit()
        done += len(chunk)
        dt = time.time() - t0
        print(f"  embed[V3] {done}/{len(rows)} | {done / dt:.0f} док/с | "
              f"eta {((len(rows) - done) / max(done / dt, 1)) / 60:.1f} мин", flush=True)


def run_assign(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT theme_label, embedding FROM {CENTROIDS_TABLE} ORDER BY theme_label")
        rows = cur.fetchall()
    labels = [r[0] for r in rows]
    cents = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {T_CLEAN_ASSIGN}")
        cur.execute(f"""CREATE TABLE {T_CLEAN_ASSIGN} (
            xml_id INTEGER PRIMARY KEY,
            assigned_theme INTEGER NOT NULL,
            similarity REAL NOT NULL,
            is_unknown BOOLEAN NOT NULL)""")
        cur.execute(f"SELECT xml_id, subject_clean_embedding FROM {T_EMBEDDINGS} "
                    f"WHERE subject_clean_embedding IS NOT NULL ORDER BY xml_id")
        rows = cur.fetchall()
    conn.commit()

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    print(f"assign[V3]: {len(ids)} эмбеддингов")

    sims = emb @ cents.T
    best_idx = sims.argmax(axis=1)
    best = sims[np.arange(len(ids)), best_idx]
    assigned = np.array([labels[i] for i in best_idx])
    unknown = best < SIMILARITY_THRESHOLD

    payload = [(int(i), int(a), float(s), bool(u))
               for i, a, s, u in zip(ids, assigned, best, unknown)]
    with conn.cursor() as cur:
        execute_values(cur,
                       f"INSERT INTO {T_CLEAN_ASSIGN} (xml_id, assigned_theme, similarity, is_unknown) "
                       f"VALUES %s", payload, page_size=1000)
    conn.commit()
    n_unk = int(unknown.sum())
    print(f"  coverage @ {SIMILARITY_THRESHOLD}: {1 - n_unk / len(best):.2%}, "
          f"unknown {n_unk} ({n_unk / len(best):.1%})")
    print(f"  sim avg={best.mean():.3f} min={best.min():.3f} max={best.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-embed', action='store_true')
    ap.add_argument('--embed-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard-id', type=int, default=0)
    args = ap.parse_args()
    if args.device:
        from config import cfg
        cfg.EMB_GPU_DEVICE = args.device
        print(f"device override: {args.device} (shard {args.shard_id}/{args.shards})")
    conn = psycopg2.connect(DB_URL)
    if not args.skip_embed:
        run_embed(conn, limit=args.limit, shards=args.shards, shard_id=args.shard_id)
    if args.embed_only:
        conn.close()
        print(f'Phase X4 embed shard {args.shard_id}/{args.shards} completed')
        return
    run_assign(conn)
    apply_marked_override(conn, T_CLEAN_ASSIGN)
    conn.close()
    print('Phase X4 completed')


if __name__ == '__main__':
    main()
