# database.py
# Версия: 3.0
# Дата: 2026-06-08
# Описание: Модуль взаимодействия с PostgreSQL (pgvector).
#            v3.0: Расширенная таблица documents, инкрементальное сохранение по этапам,
#                  resume для этапов 3 и 4, батчевые операции.

import threading

from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Dict, Optional, Set
from pathlib import Path

from config import cfg
from logger_utils import logger

_connection_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_db_instance: Optional["DatabaseManager"] = None


# RLock (reentrant): get_db() holds this lock and then constructs DatabaseManager(),
# which calls _get_pool() that re-acquires the same lock. A plain Lock would self-deadlock
# the calling thread (and, in the async web server, freeze the entire event loop).
_pool_lock = threading.RLock()

def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _connection_pool
    with _pool_lock:
        if _connection_pool is None or _connection_pool.closed:
            _connection_pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                host=cfg.PG_HOST,
                port=cfg.PG_PORT,
                database=cfg.PG_DB,
                user=cfg.PG_USER,
                password=cfg.PG_PASSWORD
            )
    return _connection_pool


def get_db() -> "DatabaseManager":
    global _db_instance
    with _pool_lock:
        if _db_instance is None:
            _db_instance = DatabaseManager()
    return _db_instance


def close_db():
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self._from_pool = False
        self.connect()

    def connect(self):
        try:
            p = _get_pool()
            self.conn = p.getconn()
            self._from_pool = True
            try:
                register_vector(self.conn)
                self._create_tables()
            except Exception:
                try:
                    p.putconn(self.conn)
                except Exception:
                    pass
                self.conn = None
                self._from_pool = False
                raise
        except Exception as e:
            logger.critical(f"Failed to connect to DB: {e}")
            raise

    def _create_tables(self):
        with self.conn.cursor() as cur:
            # ===== Основная таблица документов (общая для всех форматов) =====
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
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
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("DROP TABLE IF EXISTS summaries;")

            existing_cols = set()
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'documents';
            """)
            for row in cur.fetchall():
                existing_cols.add(row[0])

            migrations = [
                ("format_type", "TEXT"),  # Phase 1: тип формата (pdf_text, word_docx и т.д.)
                ("text", "TEXT"),
                ("image_path", "TEXT"),
                ("error", "TEXT"),
                ("enriched_text", "TEXT"),
                ("topic", "TEXT"),
                ("doc_type", "TEXT"),
                ("purpose", "TEXT"),
                ("stage_2_done", "BOOLEAN DEFAULT FALSE"),
                ("stage_3_done", "BOOLEAN DEFAULT FALSE"),
                ("text_embedded", "BOOLEAN DEFAULT FALSE"),
                ("image_embedded", "BOOLEAN DEFAULT FALSE"),
                ("updated_at", "TIMESTAMP DEFAULT NOW()"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    cur.execute(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                    logger.info(f"Added column {col_name} to documents table")

            self.conn.commit()
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_stage2
                ON documents (stage_2_done) WHERE stage_2_done = FALSE;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_stage3
                ON documents (stage_3_done) WHERE stage_3_done = FALSE;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_text_emb
                ON documents (text_embedded) WHERE text_embedded = FALSE;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_image_emb
                ON documents (image_embedded) WHERE image_embedded = FALSE;
            """)

            dim = cfg.EMB_DIMENSION

            # ===== Общие таблицы эмбеддингов (обратная совместимость) =====
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector({dim})
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS image_embeddings (
                    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector({dim})
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_emb ON text_embeddings
                USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_emb ON image_embeddings
                USING hnsw (embedding vector_cosine_ops);
            """)

            # ===== Таблицы извлечения по форматам (Phase 1: PDF) =====
            self._create_format_extraction_tables(cur, dim)

            cur.execute("""
                DROP TABLE IF EXISTS summaries;
            """)

            self.conn.commit()

    def _create_format_extraction_tables(self, cur, dim):
        """Создаёт таблицы извлечения и эмбеддингов для каждого формата.
        
        Phase 1: PDF форматы (PoC)
        Phase 2: Word/Excel/Image/XML форматы
        """
        
        # Все поддерживаемые форматы (Phase 1 + Phase 2)
        all_formats = [
            # PDF
            "pdf_text", "pdf_scan",
            "pdf_tables", "pdf_tables_fin", "pdf_tables_tech",
            "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other",
            # Word
            "word_docx", "word_doc", "word_rtf", "word_txt", "word_odt",
            # Excel
            "excel_xlsx", "excel_csv", "excel_ods",
            # Image
            "image_jpg", "image_png", "image_gif", "image_bmp", "image_tiff", "image_webp",
            # XML
            "xml_xml", "xml_xsd", "xml_xsl", "xml_wsdl",
        ]
        
        for fmt in all_formats:
            table_name = f"documents_{fmt}"
            
            # Таблица извлечения для формата
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    text TEXT,
                    image_path TEXT,
                    error TEXT,
                    stage_2_done BOOLEAN DEFAULT FALSE,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Backup-таблица для формата (создаётся пустой, заполняется после этапа 1)
            backup_name = f"{table_name}_backup"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {backup_name} (
                    id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    text TEXT,
                    image_path TEXT,
                    error TEXT,
                    stage_2_done BOOLEAN DEFAULT FALSE,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Таблицы эмбеддингов для формата (текстовые + графические)
            text_emb_name = f"text_embeddings_{fmt}"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {text_emb_name} (
                    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector({dim})
                );
            """)
            
            img_emb_name = f"image_embeddings_{fmt}"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {img_emb_name} (
                    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector({dim})
                );
            """)
            
            # Индексы для эмбеддингов формата
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{text_emb_name}_emb 
                ON {text_emb_name} USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{img_emb_name}_emb 
                ON {img_emb_name} USING hnsw (embedding vector_cosine_ops);
            """)

    # ===== Stage 1: Insert sorted documents =====

    def insert_documents_batch(self, records: List[tuple]) -> List[int]:
        if not records:
            return []
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO documents (file_path, format_type)
                    VALUES %s
                    ON CONFLICT (file_path) DO UPDATE SET
                        format_type = EXCLUDED.format_type,
                        updated_at = NOW()
                    RETURNING id;
                    """,
                    records
                )
                ids = [row[0] for row in cur.fetchall()]
                self.conn.commit()
                return ids
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Insert batch failed ({len(records)} records), retrying individually: {e}")
            saved = []
            for fp, ft in records:
                try:
                    doc_id = self.upsert_document(fp, ft)
                    saved.append(doc_id)
                except Exception as e2:
                    logger.error(f"Insert failed for {fp}: {e2}")
            logger.info(f"Insert: {len(saved)}/{len(records)} saved individually")
            return saved

    def upsert_document(self, file_path: str, file_type: str) -> int:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (file_path, format_type)
                    VALUES (%s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        format_type = EXCLUDED.format_type,
                        updated_at = NOW()
                    RETURNING id;
                """, (file_path, file_type))
                doc_id = cur.fetchone()[0]
                self.conn.commit()
                return doc_id
        except Exception:
            self.conn.rollback()
            raise

    # ===== Stage 2: Update extraction data =====

    def update_extraction(self, file_path: str, text: Optional[str],
                          image_path: Optional[str], error: Optional[str]):
        if isinstance(text, str):
            text = text.replace('\x00', '')
        if isinstance(error, str):
            error = error.replace('\x00', '')
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET
                        text = %s,
                        image_path = %s,
                        error = %s,
                        stage_2_done = TRUE,
                        updated_at = NOW()
                    WHERE file_path = %s;
                """, (text, image_path, error, file_path))
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def update_extraction_batch(self, records: List[tuple]):
        if not records:
            return
        clean = []
        for rec in records:
            fp, text, img, err = rec
            if isinstance(text, str):
                text = text.replace('\x00', '')
            if isinstance(err, str):
                err = err.replace('\x00', '')
            clean.append((fp, text, img, err))
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    UPDATE documents SET
                        text = data.text,
                        image_path = data.image_path,
                        error = data.error,
                        stage_2_done = TRUE,
                        updated_at = NOW()
                    FROM (VALUES %s) AS data(file_path, text, image_path, error)
                    WHERE documents.file_path = data.file_path;
                    """,
                    clean
                )
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Extraction batch failed ({len(records)} records), retrying individually: {e}")
            saved = 0
            for fp, text, img, err in clean:
                try:
                    self.update_extraction(fp, text, img, err)
                    saved += 1
                except Exception as e2:
                    logger.error(f"Extraction save failed for {fp}: {e2}")
            logger.info(f"Extraction: {saved}/{len(records)} saved individually")

    def update_file_path(self, old_path: str, new_path: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET file_path = %s, updated_at = NOW()
                    WHERE file_path = %s;
                """, (new_path, old_path))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Failed to update file path {old_path} -> {new_path}: {e}")

    # ===== Stage 3: Update summary data =====

    def update_summary(self, file_path: str, enriched_text: Optional[str],
                        topic: Optional[str], doc_type: Optional[str],
                        purpose: Optional[str]):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET
                        enriched_text = %s,
                        topic = %s,
                        doc_type = %s,
                        purpose = %s,
                        stage_3_done = TRUE,
                        updated_at = NOW()
                    WHERE file_path = %s;
                """, (enriched_text, topic, doc_type, purpose, file_path))
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def update_summary_batch(self, records: List[tuple]):
        if not records:
            return
        clean = []
        for rec in records:
            fp, enriched, topic, doc_type, purpose = rec
            if isinstance(enriched, str):
                enriched = enriched.replace('\x00', '')
            if isinstance(topic, str):
                topic = topic.replace('\x00', '')
            if isinstance(doc_type, str):
                doc_type = doc_type.replace('\x00', '')
            if isinstance(purpose, str):
                purpose = purpose.replace('\x00', '')
            clean.append((fp, enriched, topic, doc_type, purpose))
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    UPDATE documents SET
                        enriched_text = data.enriched_text,
                        topic = data.topic,
                        doc_type = data.doc_type,
                        purpose = data.purpose,
                        stage_3_done = TRUE,
                        updated_at = NOW()
                    FROM (VALUES %s) AS data(file_path, enriched_text, topic, doc_type, purpose)
                    WHERE documents.file_path = data.file_path;
                    """,
                    clean
                )
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Summary batch failed ({len(records)} records), retrying individually: {e}")
            saved = 0
            for fp, enriched, topic, doc_type, purpose in clean:
                try:
                    self.update_summary(fp, enriched, topic, doc_type, purpose)
                    saved += 1
                except Exception as e2:
                    logger.error(f"Summary save failed for {fp}: {e2}")
            logger.info(f"Summary: {saved}/{len(records)} saved individually")

    # ===== Stage 4: Embeddings =====

    def save_text_embeddings_batch(self, records: List[tuple], format_type: str = None):
        if not records:
            return
        table = self.get_embedding_table_name(format_type or "pdf_text", "text")
        if not table:
            logger.warning(f"No text embedding table for format {format_type}")
            return
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {table} (doc_id, embedding)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                    """,
                    records
                )
                doc_ids = [r[0] for r in records]
                cur.execute("""
                    UPDATE documents SET text_embedded = TRUE, updated_at = NOW()
                    WHERE id = ANY(%s);
                """, (doc_ids,))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Text embeddings batch failed ({len(records)} records, format={format_type}): {e}")
            saved = 0
            for doc_id, emb in records:
                try:
                    self.save_text_embedding(doc_id, emb, format_type)
                    saved += 1
                except Exception as e2:
                    logger.error(f"Text embedding save failed for doc_id {doc_id}: {e2}")
            logger.info(f"Text embeddings: {saved}/{len(records)} saved individually")

    def save_text_embedding(self, doc_id: int, embedding: np.ndarray, format_type: str = None):
        if embedding is None:
            return
        table = self.get_embedding_table_name(format_type or "pdf_text", "text")
        if not table:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {table} (doc_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                """, (doc_id, embedding))
                cur.execute("""
                    UPDATE documents SET text_embedded = TRUE, updated_at = NOW()
                    WHERE id = %s;
                """, (doc_id,))
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def save_image_embedding(self, doc_id: int, embedding: np.ndarray, format_type: str = None):
        if embedding is None:
            return
        table = self.get_embedding_table_name(format_type or "pdf_scan", "image")
        if not table:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {table} (doc_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                """, (doc_id, embedding))
                cur.execute("""
                    UPDATE documents SET image_embedded = TRUE, updated_at = NOW()
                    WHERE id = %s;
                """, (doc_id,))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise

    def save_image_embeddings_batch(self, records: List[tuple], format_type: str = None):
        if not records:
            return
        table = self.get_embedding_table_name(format_type or "pdf_scan", "image")
        if not table:
            logger.warning(f"No image embedding table for format {format_type}")
            return
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {table} (doc_id, embedding)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                    """,
                    records
                )
                doc_ids = [r[0] for r in records]
                cur.execute("""
                    UPDATE documents SET image_embedded = TRUE, updated_at = NOW()
                    WHERE id = ANY(%s);
                """, (doc_ids,))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.warning(f"Image embeddings batch failed ({len(records)} records, format={format_type}): {e}")
            saved = 0
            for doc_id, emb in records:
                try:
                    self.save_image_embedding(doc_id, emb, format_type)
                    saved += 1
                except Exception as e2:
                    logger.error(f"Image embedding save failed for doc_id {doc_id}: {e2}")
            logger.info(f"Image embeddings: {saved}/{len(records)} saved individually")

    # ===== Resume: Get unprocessed documents =====

    def get_doc_id_by_path(self, file_path: str) -> Optional[int]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE file_path = %s;", (file_path,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_summarized_paths(self) -> Set[str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT file_path FROM documents WHERE stage_3_done = TRUE;")
            return {row[0] for row in cur.fetchall()}

    def get_embedded_text_paths(self) -> Set[str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT file_path FROM documents WHERE text_embedded = TRUE;")
            return {row[0] for row in cur.fetchall()}

    def get_embedded_image_paths(self) -> Set[str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT file_path FROM documents WHERE image_embedded = TRUE;")
            return {row[0] for row in cur.fetchall()}

    def load_summaries(self, file_paths: List[str]) -> Dict[str, Dict]:
        if not file_paths:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT file_path, enriched_text, topic, doc_type, purpose
                FROM documents WHERE file_path = ANY(%s) AND stage_3_done = TRUE;
            """, (file_paths,))
            result = {}
            for row in cur.fetchall():
                result[row[0]] = {
                    "enriched_text": row[1],
                    "topic": row[2],
                    "doc_type": row[3],
                    "purpose": row[4],
                }
            return result

    # ===== Load all data for clustering =====

    def get_all_data_with_embeddings(self) -> List[Dict]:
        logger.info("Loading documents with embeddings from Database...")
        data = []
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, format_type,
                       enriched_text, image_path,
                       topic, doc_type, purpose, error
                FROM documents
                WHERE stage_2_done = TRUE AND (error IS NULL OR error = '');
            """)
            rows = cur.fetchall()

        for row in rows:
            doc_id, file_path, format_type = row[0], row[1], row[2]
            item = {
                "id": doc_id,
                "source": file_path,
                "type": format_type,
            }
            raw_text = row[3]
            item["text"] = raw_text if raw_text else ""
            if row[4]:
                item["image"] = row[4]
            if row[5]:
                item["topic"] = row[5]
            if row[6]:
                item["doc_type"] = row[6]
            if row[7]:
                item["purpose"] = row[7]
            if row[8]:
                item["error"] = row[8]

            with self.conn.cursor() as cur:
                text_table = self.get_embedding_table_name(format_type or "pdf_text", "text")
                if text_table:
                    try:
                        cur.execute(f"SELECT embedding FROM {text_table} WHERE doc_id = %s;", (doc_id,))
                        emb_row = cur.fetchone()
                        if emb_row and emb_row[0] is not None:
                            item["text_embedding"] = emb_row[0]
                    except Exception:
                        pass

                image_table = self.get_embedding_table_name(format_type or "pdf_scan", "image")
                if image_table:
                    try:
                        cur.execute(f"SELECT embedding FROM {image_table} WHERE doc_id = %s;", (doc_id,))
                        emb_row = cur.fetchone()
                        if emb_row and emb_row[0] is not None:
                            item["image_embedding"] = emb_row[0]
                    except Exception:
                        pass

            data.append(item)

        logger.info(f"Loaded {len(data)} documents with embeddings.")
        return data

    # ===== Statistics =====

    def get_stats(self) -> Dict:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE stage_2_done) as stage2,
                    COUNT(*) FILTER (WHERE stage_3_done) as stage3,
                    COUNT(*) FILTER (WHERE text_embedded) as text_emb,
                    COUNT(*) FILTER (WHERE image_embedded) as img_emb,
                    COUNT(*) FILTER (WHERE error IS NOT NULL) as errors
                FROM documents;
            """)
            row = cur.fetchone()
            return {
                "total": row[0],
                "stage2": row[1],
                "stage3": row[2],
                "text_emb": row[3],
                "img_emb": row[4],
                "errors": row[5],
            }

    # ===== Cleanup for rollback =====

    def clear_stage3(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET
                    enriched_text = NULL,
                    topic = NULL,
                    doc_type = NULL,
                    purpose = NULL,
                    stage_3_done = FALSE,
                    updated_at = NOW()
                WHERE stage_3_done = TRUE;
            """)
            deleted = cur.rowcount
            self.conn.commit()
            logger.info(f"Cleared stage 3 data for {deleted} documents")

    def clear_stage4(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM text_embeddings;")
            cur.execute("DELETE FROM image_embeddings;")
            # Очищаем format-specific таблицы эмбеддингов
            all_formats = [
                "pdf_text", "pdf_scan", "pdf_tables", "pdf_tables_fin", "pdf_tables_tech", "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other",
                "word_docx", "word_doc", "word_rtf", "word_txt", "word_odt",
                "excel_xlsx", "excel_csv", "excel_ods",
                "image_jpg", "image_png", "image_gif", "image_bmp", "image_tiff", "image_webp",
                "xml_xml", "xml_xsd", "xml_xsl", "xml_wsdl",
            ]
            for fmt in all_formats:
                for emb_type in ("text", "image"):
                    table = self.get_embedding_table_name(fmt, emb_type)
                    if table:
                        try:
                            cur.execute(f"DELETE FROM {table};")
                        except Exception as e:
                            logger.warning(f"Failed to clear {table}: {e}")
            cur.execute("""
                UPDATE documents SET
                    text_embedded = FALSE,
                    image_embedded = FALSE,
                    updated_at = NOW();
            """)
            self.conn.commit()
            logger.info("Cleared all embeddings")

    def clear_stage1(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM text_embeddings;")
            cur.execute("DELETE FROM image_embeddings;")
            cur.execute("DELETE FROM documents;")
            self.conn.commit()
            logger.info("Cleared stage 1 data from database")

    def clear_stage2(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET
                    text = NULL,
                    image_path = NULL,
                    error = NULL,
                    stage_2_done = FALSE,
                    updated_at = NOW()
                WHERE stage_2_done = TRUE;
            """)
            deleted = cur.rowcount
            self.conn.commit()
            logger.info(f"Cleared stage 2 data for {deleted} documents")

    def clear_all(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM text_embeddings;")
            cur.execute("DELETE FROM image_embeddings;")
            cur.execute("DELETE FROM documents;")
            self.conn.commit()
            logger.info("Cleared all data from database")

    # ===== Методы для работы с таблицами по форматам (Phase 1: PDF PoC) =====

    @staticmethod
    def get_table_name_for_format(format_type: str, table_prefix: str = "documents") -> str:
        """Возвращает имя таблицы для данного типа формата."""
        return f"{table_prefix}_{format_type}"

    @staticmethod
    def get_embedding_table_name(format_type: str, emb_type: str) -> Optional[str]:
        """Возвращает имя таблицы эмбеддингов для формата.
        
        Args:
            format_type: тип формата (pdf_text, word_docx и т.д.)
            emb_type: 'text' или 'image'
        
        Returns:
            Имя таблицы или None если формат не поддерживает данный тип эмбеддингов
        """
        # Phase 1: PDF форматы поддерживают оба типа эмбеддингов
        pdf_formats = {"pdf_text", "pdf_scan", "pdf_tables",
                     "pdf_tables_fin", "pdf_tables_tech",
                     "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other"}
        
        if format_type in pdf_formats:
            return f"{emb_type}_embeddings_{format_type}"
        
        # Будущие форматы (Phase 2)
        image_formats = {"image_jpg", "image_png", "image_gif", "image_bmp", "image_tiff", "image_webp"}
        if emb_type == "text" and format_type in image_formats:
            return f"text_embeddings_{format_type}"
        if emb_type == "image" and format_type in image_formats:
            return f"image_embeddings_{format_type}"
        
        # Текстовые форматы — текстовые эмбеддинги + image для ODT/ODP
        text_formats = {"word_docx", "word_doc", "word_rtf", "word_txt", "word_odt",
                        "excel_xlsx", "excel_csv", "excel_ods",
                        "xml_xml", "xml_xsd", "xml_xsl", "xml_wsdl"}
        if emb_type == "text" and format_type in text_formats:
            return f"text_embeddings_{format_type}"
        
        # ODT/ODP поддерживают image эмбеддинги
        if emb_type == "image" and format_type in {"word_odt", "word_doc"}:
            return f"image_embeddings_{format_type}"
        
        return None

    def create_backup_tables(self):
        """Создаёт backup-таблицы для всех форматов после этапа 1."""
        with self.conn.cursor() as cur:
            # Все поддерживаемые форматы (Phase 1 + Phase 2)
            # pdf_tables_* — подтипы финансовой/тех/договорной/etc документации;
            # pdf_tables — общий код для unmatched (fallback под LLM на этапе 2).
            all_formats = [
                # PDF
                "pdf_text", "pdf_scan",
                "pdf_tables", "pdf_tables_fin", "pdf_tables_tech",
                "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other",
                # Word
                "word_docx", "word_doc", "word_rtf", "word_txt", "word_odt",
                # Excel
                "excel_xlsx", "excel_csv", "excel_ods",
                # Image
                "image_jpg", "image_png", "image_gif", "image_bmp", "image_tiff", "image_webp",
# XML
                "xml_xml", "xml_xsd", "xml_xsl", "xml_wsdl",
            ]

            for fmt in all_formats:
                table_name = f"documents_{fmt}"
                backup_name = f"{table_name}_backup"
                
                try:
                    cur.execute(f"""
                        DELETE FROM {backup_name};
                        INSERT INTO {backup_name} SELECT * FROM {table_name};
                    """)
                    logger.info(f"Backup создан для {table_name}")
                except Exception as e:
                    logger.warning(f"Не удалось создать backup для {table_name}: {e}")
            
            self.conn.commit()

    def rollback_by_format(self, format_type: str, file_paths: List[str], restore_dir: Path):
        """Откат обработки формата: возврат файлов + очистка БД.

        Args:
            format_type: тип формата (pdf_text, word_docx и т.д.)
            file_paths: список путей файлов для отката
            restore_dir: директория для возврата файлов (Sorted/{format})
        """
        if not file_paths:
            logger.info(f"Откат формата {format_type}: пустой список файлов")
            return

        with self.conn.cursor() as cur:
            # Получаем doc_id из documents по file_path
            cur.execute("""
                SELECT id, file_path FROM documents WHERE file_path = ANY(%s);
            """, (file_paths,))
            rows = cur.fetchall()
            doc_ids = [row[0] for row in rows]
            actual_paths = [row[1] for row in rows]

            if not doc_ids:
                logger.info(f"Откат формата {format_type}: нет записей для отката")
                return

            # Удаляем эмбеддинги из format-specific таблиц
            for emb_type in ("text", "image"):
                table = self.get_embedding_table_name(format_type, emb_type)
                if table:
                    try:
                        cur.execute(f"DELETE FROM {table} WHERE doc_id = ANY(%s);", (doc_ids,))
                    except Exception as e:
                        logger.warning(f"Failed to clear {table}: {e}")

            # Очищаем данные извлечения в основной таблице documents
            cur.execute("""
                UPDATE documents SET
                    text = NULL, image_path = NULL, error = NULL,
                    enriched_text = NULL, topic = NULL, doc_type = NULL, purpose = NULL,
                    stage_2_done = FALSE, stage_3_done = FALSE,
                    text_embedded = FALSE, image_embedded = FALSE,
                    updated_at = NOW()
                WHERE file_path = ANY(%s);
            """, (actual_paths,))

            deleted = cur.rowcount
            self.conn.commit()

        logger.info(f"Откат формата {format_type}: очищено {deleted} записей в БД")

    def get_format_stats(self) -> Dict[str, int]:
        """Возвращает статистику обработанных файлов по форматам."""
        stats = {}

        with self.conn.cursor() as cur:
            all_formats = [
                "pdf_text", "pdf_scan", "pdf_tables", "pdf_tables_fin", "pdf_tables_tech", "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other",
                "word_docx", "word_doc", "word_rtf", "word_txt", "word_odt",
                "excel_xlsx", "excel_csv", "excel_ods",
                "image_jpg", "image_png", "image_gif", "image_bmp", "image_tiff", "image_webp",
                "xml_xml", "xml_xsd", "xml_xsl", "xml_wsdl",
            ]

            for fmt in all_formats:
                try:
                    cur.execute("""
                        SELECT COUNT(*) FROM documents
                        WHERE format_type = %s AND stage_2_done = TRUE;
                    """, (fmt,))
                    stats[fmt] = cur.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Не удалось получить статистику для {fmt}: {e}")

        return stats

    def close(self):
        if self.conn:
            if self._from_pool:
                try:
                    p = _get_pool()
                    p.putconn(self.conn)
                except Exception:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
            else:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.conn = None