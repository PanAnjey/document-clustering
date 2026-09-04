# phase_x2_embed_assign.py
# Этап X2 эксперимента XML: эмбеддинги (nomic-embed-text, cuda:1, 768d float32
# bytea) ДВУХ представлений документа и присвоение 26 темам предыдущего
# эксперимента (temp_theme2_centroids_v2) по cosine sim, порог 0.8 —
# методология идентична experiment_excel/phase_3r_assign.py.
#
#   V1 = canonical_text (полный канон: стороны, адреса, суммы, основания, Доп)
#   V2 = subject_text   (тематический: предметные строки + НаимДокОпр + СодОпер)
#
# Таблицы:
#   temp_xml_embeddings     (xml_id PK, embedding BYTEA, subject_embedding BYTEA)
#   temp_xml_assign         (V1: xml_id PK, assigned_theme, similarity, is_unknown)
#   temp_xml_subject_assign (V2: то же)
#
# Запуск:
#   python phase_x2_embed_assign.py              # embed (с резюмом) + assign
#   python phase_x2_embed_assign.py --skip-embed # только assign
#   python phase_x2_embed_assign.py --limit 5000 # отладка

import argparse
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    DB_URL, T_CANONICAL, T_EMBEDDINGS, T_ASSIGN, T_SUBJECT_ASSIGN,
    EMBED_BATCH_SIZE, EMBED_DIM, SIMILARITY_THRESHOLD, CENTROIDS_TABLE,
    MARKED_THEME,
)
from xml_embed_helper import encode_texts  # noqa: E402


def apply_marked_override(conn, assign_table):
    """Документы с маркированными товарами (N3=1 в ИдФайл) → отдельный
    кластер MARKED_THEME независимо от similarity (п.1 правок)."""
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {assign_table} a
            SET assigned_theme = %s, is_unknown = FALSE
            FROM {T_CANONICAL} c
            WHERE c.id = a.xml_id AND c.is_marked""", (MARKED_THEME,))
        n = cur.rowcount
    conn.commit()
    print(f"  marked override: {n} док. → кластер {MARKED_THEME} (ГИС МТ)")

FETCH_CHUNK = 4000

ASSIGN_DDL = """
    xml_id INTEGER PRIMARY KEY,
    assigned_theme INTEGER NOT NULL,
    similarity REAL NOT NULL,
    is_unknown BOOLEAN NOT NULL
"""


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {T_EMBEDDINGS} (
                xml_id INTEGER PRIMARY KEY,
                embedding BYTEA NOT NULL,
                subject_embedding BYTEA
            )""")
    conn.commit()


def run_embed(conn, limit=0, shards=1, shard_id=0):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT c.id, c.canonical_text, c.subject_text FROM {T_CANONICAL} c
            WHERE c.parse_ok AND c.canonical_text IS NOT NULL
              AND (c.id %% %s) = %s
              AND NOT EXISTS (SELECT 1 FROM {T_EMBEDDINGS} e WHERE e.xml_id = c.id)
            ORDER BY c.id
        """ + (f" LIMIT {limit}" if limit else ""), (shards, shard_id))
        rows = cur.fetchall()
    if not rows:
        print("embed: нечего делать (всё уже заэмбежено)")
        return
    print(f"embed: {len(rows)} документов (V1 + V2)")
    t0 = time.time()
    done = 0
    for start in range(0, len(rows), FETCH_CHUNK):
        chunk = rows[start:start + FETCH_CHUNK]
        ids = [r[0] for r in chunk]
        emb_v1 = encode_texts([r[1] or '' for r in chunk], batch_size=64)
        emb_v2 = encode_texts([r[2] or '' for r in chunk], batch_size=64)
        payload = [(int(ids[k]),
                    emb_v1[k].astype(np.float32).tobytes(),
                    emb_v2[k].astype(np.float32).tobytes())
                   for k in range(len(chunk))]
        with conn.cursor() as cur:
            execute_values(cur,
                           f"INSERT INTO {T_EMBEDDINGS} (xml_id, embedding, subject_embedding) "
                           f"VALUES %s ON CONFLICT (xml_id) DO NOTHING",
                           payload, page_size=500)
        conn.commit()
        done += len(chunk)
        dt = time.time() - t0
        print(f"  embed {done}/{len(rows)} | {done / dt:.0f} док/с | "
              f"eta {((len(rows) - done) / max(done / dt, 1)) / 60:.1f} мин", flush=True)


def _assign_one(conn, emb_col, assign_table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT theme_label, embedding FROM {CENTROIDS_TABLE} ORDER BY theme_label")
        rows = cur.fetchall()
    labels = [r[0] for r in rows]
    cents = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {assign_table}")
        cur.execute(f"CREATE TABLE {assign_table} ({ASSIGN_DDL})")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT xml_id, {emb_col} FROM {T_EMBEDDINGS} "
                    f"WHERE {emb_col} IS NOT NULL ORDER BY xml_id")
        rows = cur.fetchall()
    print(f"assign[{assign_table} ← {emb_col}]: {len(rows)} эмбеддингов, "
          f"центроидов {len(labels)}")

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    sims = emb @ cents.T
    best_idx = sims.argmax(axis=1)
    best = sims[np.arange(len(ids)), best_idx]
    assigned = np.array([labels[i] for i in best_idx])
    unknown = best < SIMILARITY_THRESHOLD

    payload = [(int(i), int(a), float(s), bool(u))
               for i, a, s, u in zip(ids, assigned, best, unknown)]
    with conn.cursor() as cur:
        execute_values(cur,
                       f"INSERT INTO {assign_table} (xml_id, assigned_theme, similarity, is_unknown) "
                       f"VALUES %s", payload, page_size=1000)
    conn.commit()

    n_unk = int(unknown.sum())
    print(f"  coverage @ {SIMILARITY_THRESHOLD}: {1 - n_unk / len(best):.2%} "
          f"({len(best) - n_unk}/{len(best)}), unknown {n_unk} ({n_unk / len(best):.1%})")
    print(f"  sim avg={best.mean():.3f} min={best.min():.3f} max={best.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-embed', action='store_true')
    ap.add_argument('--embed-only', action='store_true',
                    help='только эмбеддинг (для шардов); assign запускается отдельно')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default=None,
                    help="GPU девайс, напр. cuda:0 / cuda:1 (переопределяет cfg.EMB_GPU_DEVICE)")
    ap.add_argument('--shards', type=int, default=1, help='число шардов (по id %% N)')
    ap.add_argument('--shard-id', type=int, default=0, help='номер данного шарда 0..N-1')
    args = ap.parse_args()

    if args.device:
        from config import cfg
        cfg.EMB_GPU_DEVICE = args.device
        print(f"device override: {args.device} (shard {args.shard_id}/{args.shards})")

    conn = psycopg2.connect(DB_URL)
    ensure_tables(conn)
    if not args.skip_embed:
        run_embed(conn, limit=args.limit, shards=args.shards, shard_id=args.shard_id)
    if args.embed_only:
        conn.close()
        print(f'Phase X2 embed shard {args.shard_id}/{args.shards} completed')
        return
    _assign_one(conn, 'embedding', T_ASSIGN)
    _assign_one(conn, 'subject_embedding', T_SUBJECT_ASSIGN)
    apply_marked_override(conn, T_ASSIGN)
    apply_marked_override(conn, T_SUBJECT_ASSIGN)
    conn.close()
    print('Phase X2 completed')


if __name__ == '__main__':
    main()
