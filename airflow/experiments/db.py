"""db.py — connection + schema helpers для исследовательских прогонов.

Каждый прогон эксперимента изолируется через отдельную PostgreSQL-схему
`exp_<run_id>`. Существующий `DatabaseManager` (database.py) работает
против `public` и не модифицируется. Варианты, вызываемые Airflow,
используют helpers ниже, чтобы:

1. Создать/удалить схему прогона с копией структуры таблиц `documents`
   и `documents_<fmt>` (+ эмбеддинг-таблицы).
2. Клонировать золотой Stage 1 (`gold_stage1.documents`) в схему прогона.
3. Подменить `search_path` текущего соединения так, чтобы unqualified
   `INSERT INTO documents ...` автоматически попадали в схему прогона.
"""
import re
import threading
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values

from config import cfg
from logger_utils import logger

SCHEMA_PREFIX = "exp_"
RUN_ID_RE = re.compile(r"^[a-z0-9_]+$")
PG_DIM = None  # ленивое получение

_conn = None
_conn_lock = threading.Lock()


def get_conn():
    """Возвращает соединение с public (НЕ для прогона).

    Для прогона используйте `set_search_path` поверх полученного
    соединения или управляйте своим соединением в варианте.
    """
    global _conn
    with _conn_lock:
        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(
                host=cfg.PG_HOST, port=cfg.PG_PORT,
                dbname=cfg.PG_DB, user=cfg.PG_USER, password=cfg.PG_PASSWORD,
            )
            _conn.autocommit = False
        return _conn


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id={run_id!r}, allowed: [a-z0-9_]+")
    if not run_id.startswith(SCHEMA_PREFIX):
        # Допускаем как 'exp_xxx', так и 'xxx' — нормализуем к схеме.
        schema = f"{SCHEMA_PREFIX}{run_id}"
    else:
        schema = run_id
    if len(schema) > 63:
        raise ValueError(f"run_id/schema too long (>63 chars): {schema}")
    return schema


def _emb_dimension() -> int:
    global PG_DIM
    if PG_DIM is None:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT atttypmod::int
                FROM pg_attribute
                WHERE attrelid = 'public.text_embeddings'::regclass
                  AND attname = 'embedding'
            """)
            row = cur.fetchone()
            PG_DIM = row[0] if row and row[0] > 0 else cfg.EMB_DIMENSION
    return PG_DIM


def set_search_path(conn, run_id: str) -> None:
    """Подменяет search_path соединения так, чтобы unqualified запросы
    (INSERT INTO documents ...; INSERT INTO documents_<fmt> ...)
    попадали в схему прогона.

    `public` остаётся в search_path, поэтому pgvector-операторы и
    общие типы доступны без квалификации.
    """
    schema = _validate_run_id(run_id)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO %s, public", (schema,))
        # SET ... TO не принимает параметр как строку для schema_name —
        # но psycopg2 заменяет %s на корректно экранированную строку,
        # и SET search_path TO 'foo', public валиден в PostgreSQL.
    conn.commit()


def _list_format_types_from_gold() -> list:
    """Список всех format_type, присутствующих в gold_stage1.documents."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT format_type
            FROM gold_stage1.documents
            WHERE format_type IS NOT NULL
            GROUP BY format_type
            ORDER BY 1
        """)
        return [row[0] for row in cur.fetchall()]


def create_exp_schema(run_id: str, *, with_format_tables: bool = True) -> None:
    """Создаёт схему exp_<run_id> с таблицами `documents` (пустая) и,
    если with_format_tables=True, пустыми таблицами `documents_<fmt>`
    и `text_embeddings_<fmt>`/`image_embeddings_<fmt>` для каждого
    подформата, известного по gold_stage1.documents.

    НЕ заполняет ни одну таблицу. Заполнение `documents` — клон из
    `gold_stage1.documents` через `clone_gold_to_exp(run_id)` (опционально,
    делается при отке if re_sort=false).
    """
    schema = _validate_run_id(run_id)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

        dim = _emb_dimension()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".documents (
                id SERIAL PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                format_type TEXT,
                text TEXT,
                image_path TEXT,
                error TEXT,
                enriched_text TEXT,
                topic TEXT,
                doc_type TEXT,
                purpose TEXT,
                stage_2_done BOOLEAN DEFAULT FALSE,
                stage_3_done BOOLEAN DEFAULT FALSE,
                text_embedded BOOLEAN DEFAULT FALSE,
                image_embedded BOOLEAN DEFAULT FALSE,
                final_class TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        if with_format_tables:
            for fmt in _list_format_types_from_gold():
                if not re.match(r"^[a-z0-9_]+$", fmt):
                    logger.warning(f"skip format (invalid name for SQL): {fmt!r}")
                    continue
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{schema}".documents_{fmt} (
                        id INTEGER PRIMARY KEY REFERENCES "{schema}".documents(id) ON DELETE CASCADE,
                        text TEXT,
                        image_path TEXT,
                        error TEXT,
                        stage_2_done BOOLEAN DEFAULT FALSE,
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{schema}".text_embeddings_{fmt} (
                        doc_id INTEGER PRIMARY KEY REFERENCES "{schema}".documents(id) ON DELETE CASCADE,
                        embedding vector({dim})
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{schema}".image_embeddings_{fmt} (
                        doc_id INTEGER PRIMARY KEY REFERENCES "{schema}".documents(id) ON DELETE CASCADE,
                        embedding vector({dim})
                    )
                """)
    conn.commit()
    logger.info(f"[exp] schema created: {schema}")


def drop_exp_schema(run_id: str) -> None:
    """Удаляет схему прогона каскадно. Источник (`SourceFiles/`),
    золотой `gold_stage1` и `public` НЕ затрагиваются."""
    schema = _validate_run_id(run_id)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    conn.commit()
    logger.info(f"[exp] schema dropped: {schema}")


def clone_gold_to_exp(run_id: str) -> int:
    """Копирует содержимое `gold_stage1.documents` в `exp_<run_id>.documents`.

    Схема должна быть уже создана через `create_exp_schema(run_id)`.
    Копируются только `(file_path, format_type)`. Текст/эмбеддинги НЕ
    переносятся — они вычисляются внутри эксперимента.

    Возвращает количество скопированных строк.
    """
    schema = _validate_run_id(run_id)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE TABLE "{schema}".documents RESTART IDENTITY CASCADE')
        cur.execute(f"""
            INSERT INTO "{schema}".documents (file_path, format_type)
            SELECT file_path, format_type
            FROM gold_stage1.documents
            WHERE file_path IS NOT NULL AND format_type IS NOT NULL
            ON CONFLICT (file_path) DO NOTHING
        """)
        cur.execute(f'SELECT count(*) FROM "{schema}".documents')
        total = cur.fetchone()[0]
    conn.commit()
    logger.info(f"[exp] cloned gold_stage1 → {schema}.documents: {total} rows")
    return total


def schema_exists(run_id: str) -> bool:
    schema = _validate_run_id(run_id)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.schemata WHERE schema_name = %s
        """, (schema,))
        return cur.fetchone() is not None