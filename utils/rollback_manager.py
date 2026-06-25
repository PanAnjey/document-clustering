# utils/rollback_manager.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Управление откатом операций пайплайна.

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Dict, Optional

from config import cfg
from logger_utils import logger


class RollbackManager:
    """Управление откатом операций на этапах 2 и 3."""
    
    def __init__(self):
        self._stage2_results: Dict[str, Dict] = {}
    
    def rollback_embeddings(self, format_type: str, sorted_data: List[Dict]) -> int:
        """Откат ТОЛЬКО подэтапа эмбеддингов для формата."""
        from database import DatabaseManager
        
        format_files = [d for d in sorted_data if d.get("type") == format_type]
        
        if not format_files:
            logger.info(f"Нет файлов формата {format_type} для отката эмбеддингов")
        
        file_paths = [d["source"] for d in format_files]

        try:
            db = DatabaseManager()
        except Exception as e:
            logger.error(f"Не удалось подключиться к БД для отката эмбеддингов: {e}")
            return 0
        
        if db:
            try:
                # Берём актуальные пути из БД (файлы могли переместиться)
                with db.conn.cursor() as cur:
                    cur.execute("SELECT file_path FROM documents WHERE file_path = ANY(%s)", (file_paths,))
                    db_paths = [row[0] for row in cur.fetchall()] or file_paths
                
                self._delete_embeddings_db(db, db_paths)
                self._delete_npy_for_files(db_paths)
                
                logger.info(f"Откат эмбеддингов формата {format_type}: очищены векторы и флаги")
            except Exception as e:
                logger.error(f"Ошибка при откате эмбеддингов формата {format_type}: {e}")
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        else:
            self._delete_npy_for_files(file_paths)

        # Чистим эмбеддинги в накопителе результатов, сохраняя извлечённый текст
        for d in self._stage2_results.values():
            if d.get("type") == format_type:
                d.pop("text_embedding", None)
                d.pop("image_embedding", None)

        return len(file_paths)
    
    def rollback_extraction(self, format_type: str, sorted_data: List[Dict]) -> int:
        """Откат подэтапа извлечения для формата (вместе с зависимыми эмбеддингами)."""
        from database import DatabaseManager
        
        rolled_back = 0
        
        format_files = [d for d in sorted_data if d.get("type") == format_type]
        
        if not format_files:
            logger.info(f"Нет файлов формата {format_type} для отката извлечения")
            
            # Чистим накопитель и выходим
            for src in [k for k, d in self._stage2_results.items() if d.get("type") == format_type]:
                self._stage2_results.pop(src, None)
            
            return 0
        
        try:
            db = DatabaseManager()
        except Exception as e:
            logger.error(f"Не удалось подключиться к БД для отката: {e}")
            return 0
        
        try:
            file_paths = [d["source"] for d in format_files]
            
            with db.conn.cursor() as cur:
                # Удаляем эмбеддинги (по doc_id)
                cur.execute("""
                    DELETE FROM text_embeddings WHERE doc_id IN (
                        SELECT id FROM documents WHERE file_path = ANY(%s)
                    );
                """, (file_paths,))
                
                cur.execute("""
                    DELETE FROM image_embeddings WHERE doc_id IN (
                        SELECT id FROM documents WHERE file_path = ANY(%s)
                    );
                """, (file_paths,))
                
                # Очищаем извлечение + флаги
                cur.execute("""
                    UPDATE documents 
                    SET text = NULL, image_path = NULL, error = NULL,
                        enriched_text = NULL, topic = NULL, doc_type = NULL, purpose = NULL,
                        stage_2_done = FALSE, stage_3_done = FALSE,
                        text_embedded = FALSE, image_embedded = FALSE,
                        updated_at = NOW()
                    WHERE file_path = ANY(%s)
                """, (file_paths,))
                
                db.conn.commit()
                
                logger.info(f"Очищены данные извлечения и эмбеддингов для формата {format_type}")

            self._delete_npy_for_files(file_paths)

            # Возвращаем файлы из FailedExtraction и ErrorFiles обратно в Sorted/{format}
            target_dir = self._get_format_sorted_dir(format_type)
            
            if not target_dir:
                logger.error(f"Не найдена директория для формата {format_type}")
                return 0
            
            target_dir.mkdir(parents=True, exist_ok=True)

            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT file_path FROM documents
                    WHERE file_path = ANY(%s)
                """, (file_paths,))
                
                db_paths = [row[0] for row in cur.fetchall()]

            failed_dir = str(cfg.FAILED_EXTRACTION_DIR).lower().rstrip('\\')
            errors_dir = str(cfg.ERRORS_DIR).lower().rstrip('\\')

            for db_path_str in db_paths:
                current_path = Path(db_path_str)
                current_parent = str(current_path.parent).lower().rstrip('\\')

                if current_parent not in (failed_dir, errors_dir):
                    continue
                
                new_path = target_dir / current_path.name
                
                counter = 1
                original_new_path = new_path
                
                while new_path.exists():
                    stem = original_new_path.stem
                    ext = original_new_path.suffix
                    new_path = target_dir / f"{stem}_{counter}{ext}"
                    counter += 1

                try:
                    shutil.move(str(current_path), str(new_path))
                    
                    with db.conn.cursor() as cur:
                        cur.execute("DELETE FROM documents WHERE file_path = %s;", (str(current_path),))
                    
                    with db.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO documents (file_path, format_type, created_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (file_path) DO UPDATE SET format_type = EXCLUDED.format_type;
                        """, (str(new_path), format_type))
                    
                    db.conn.commit()
                    rolled_back += 1
                    
                    logger.debug(f"Возвращён файл: {current_path.name} -> {new_path}")
                except Exception as e:
                    logger.error(f"Не удалось вернуть файл {current_path.name}: {e}")

            logger.info(f"Откат извлечения формата {format_type}: возвращено {rolled_back} файлов")

        except Exception as e:
            logger.error(f"Ошибка при откате формата {format_type}: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass
        
        # Очищаем накопитель результатов для формата
        for src in [k for k, d in self._stage2_results.items() if d.get("type") == format_type]:
            self._stage2_results.pop(src, None)
        
        return rolled_back
    
    def restore_error_files(self, format_type: str, sorted_data: List[Dict]) -> int:
        """Возвращает файлы из FailedExtraction/ErrorFiles обратно в Sorted/{format} без очистки БД."""
        from database import DatabaseManager
        
        try:
            db = DatabaseManager()
        except Exception as e:
            logger.error(f"Не удалось подключиться к БД для восстановления: {e}")
            return 0
        
        restored = 0
        
        try:
            target_dir = cfg.TARGETS.get(format_type)
            
            if not target_dir:
                logger.error(f"Не найдена директория для формата {format_type}")
                return 0
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Запрашиваем актуальные пути файлов этого формата из БД
            format_file_paths = [d["source"] for d in sorted_data if d.get("type") == format_type]
            
            if not format_file_paths:
                logger.info(f"Нет файлов формата {format_type} для восстановления")
                return 0
            
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT file_path FROM documents
                    WHERE file_path = ANY(%s)
                """, (format_file_paths,))
                
                db_paths = [row[0] for row in cur.fetchall()]
            
            failed_dir = str(cfg.FAILED_EXTRACTION_DIR).lower().rstrip('\\')
            errors_dir = str(cfg.ERRORS_DIR).lower().rstrip('\\')
            
            for db_path in db_paths:
                current_path = Path(db_path)
                current_parent = str(current_path.parent).lower().rstrip('\\')
                
                if current_parent not in (failed_dir, errors_dir):
                    continue
                
                new_path = target_dir / current_path.name
                
                counter = 1
                original_new_path = new_path
                
                while new_path.exists():
                    stem = original_new_path.stem
                    ext = original_new_path.suffix
                    new_path = target_dir / f"{stem}_{counter}{ext}"
                    counter += 1
                
                try:
                    shutil.move(str(current_path), str(new_path))
                    
                    db.update_file_path(str(current_path), str(new_path))
                    
                    restored += 1
                    
                    logger.debug(f"Восстановлен файл: {current_path.name} -> {new_path}")
                except Exception as e:
                    logger.error(f"Не удалось восстановить файл {current_path.name}: {e}")
            
            logger.info(f"Восстановление формата {format_type}: возвращено {restored} файлов из FailedExtraction/ErrorFiles")
            
        except Exception as e:
            logger.error(f"Ошибка при восстановлении формата {format_type}: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass
        
        return restored
    
    def _delete_npy_for_files(self, file_paths: List[str]) -> None:
        """Удаляет .npy эмбеддинги для указанных файлов."""
        emb_dir = cfg.EMBEDDINGS_DIR
        
        if not emb_dir.exists():
            return
        
        removed = 0
        
        for fp in file_paths:
            stem = Path(fp).stem
            
            for suffix in (f"{stem}_text.npy", f"{stem}_image.npy"):
                p = emb_dir / suffix
                
                if p.exists():
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
        
        if removed:
            logger.debug(f"Удалено .npy эмбеддингов: {removed}")
    
    def _delete_embeddings_db(self, db, file_paths: List[str]) -> None:
        """Удаляет строки эмбеддингов в БД и сбрасывает флаги."""
        with db.conn.cursor() as cur:
            cur.execute("""
                DELETE FROM text_embeddings WHERE doc_id IN (
                    SELECT id FROM documents WHERE file_path = ANY(%s)
                );
            """, (file_paths,))
            
            cur.execute("""
                DELETE FROM image_embeddings WHERE doc_id IN (
                    SELECT id FROM documents WHERE file_path = ANY(%s)
                );
            """, (file_paths,))
            
            cur.execute("""
                UPDATE documents
                SET text_embedded = FALSE, image_embedded = FALSE, updated_at = NOW()
                WHERE file_path = ANY(%s)
            """, (file_paths,))
            
            db.conn.commit()
    
    def _get_format_sorted_dir(self, format_type: str) -> Optional[Path]:
        """Директория Sorted/{format} для формата."""
        rel = cfg.FORMAT_TARGETS.get(format_type)
        
        if rel:
            return cfg.ROOT / rel
        
        return cfg.TARGETS.get(format_type)
    
    def set_stage2_result(self, source_path: str, data: Dict) -> None:
        """Сохранение результата этапа 2 в память."""
        self._stage2_results[source_path] = data
    
    def get_stage2_result(self, source_path: str) -> Optional[Dict]:
        """Получение результата этапа 2 из памяти."""
        return self._stage2_results.get(source_path)
    
    def clear_stage2_results(self, format_type: str) -> None:
        """Очистка результатов для формата."""
        for src in [k for k, d in self._stage2_results.items() if d.get("type") == format_type]:
            self._stage2_results.pop(src, None)


# Глобальный экземпляр (для совместимости с old code)
_rollback_manager: Optional[RollbackManager] = None


def get_rollback_manager() -> RollbackManager:
    """Получение глобального экземпляра RollbackManager."""
    global _rollback_manager
    
    if _rollback_manager is None:
        _rollback_manager = RollbackManager()
    
    return _rollback_manager
