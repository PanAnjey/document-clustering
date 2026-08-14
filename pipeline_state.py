# pipeline_state.py
# Управление состоянием пайплайна: резюме, откат, сохранение промежуточных данных.

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np

from config import cfg
from logger_utils import logger

UNRAR_PATH = r"C:\Program Files\WinRAR\UnRAR.exe"
SOURCE_RAR = cfg.ROOT / "SourceFiles.rar"


def _force_remove_dir(path: Path, retries: int = 3, delay: float = 0.5) -> bool:
    """Надёжное удаление директории с retry и снятием read-only атрибутов."""
    if not path.exists():
        return True

    def _remove_readonly(func, path_str, excinfo):
        try:
            os.chmod(path_str, stat.S_IWRITE)
            func(path_str)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            if not path.exists():
                return True
        except Exception as e:
            logger.debug(f"  Attempt {attempt + 1}/{retries} failed for {path}: {e}")

        if attempt < retries - 1:
            time.sleep(delay)

    try:
        result = subprocess.run(
            ['cmd', '/c', 'rd', '/s', '/q', str(path)],
            capture_output=True,
            timeout=30
        )
        if not path.exists():
            return True
    except Exception as e:
        logger.debug(f"  cmd rd failed for {path}: {e}")

    try:
        import tempfile
        empty_dir = Path(tempfile.mkdtemp())
        subprocess.run(
            ['robocopy', str(empty_dir), str(path), '/MIR', '/R:1', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS'],
            capture_output=True,
            timeout=60
        )
        empty_dir.rmdir()
        if path.exists():
            path.rmdir()
        if not path.exists():
            return True
    except Exception as e:
        logger.debug(f"  robocopy trick failed for {path}: {e}")

    if path.exists():
        logger.warning(f"  Не удалось полностью удалить: {path}")
        return False
    return True

STAGES = [
    "stage_1_sort",
    "stage_2_process_formats",  # Каждый формат: Извлечение -> Эмбеддинги
    "stage_3_cluster_refine",   # Кластеризация + LLM уточнение (Qwen3.6-35B)
]

STAGE_LABELS = {
    "stage_1_sort": "Сортировка файлов",
    "stage_2_process_formats": "Обработка форматов",
    "stage_3_cluster_refine": "Кластеризация и уточнение (LLM)",
}

DATA_DIR = cfg.ROOT / "PipelineData"


class PipelineState:
    def __init__(self):
        self.state_file = DATA_DIR / "pipeline_state.json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Сериализация ----

    @staticmethod
    def _serialize(obj):
        if isinstance(obj, Path):
            return {"__path__": str(obj)}
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: PipelineState._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PipelineState._serialize(v) for v in obj]
        return obj

    @staticmethod
    def _deserialize(obj):
        if isinstance(obj, dict):
            if "__path__" in obj:
                return Path(obj["__path__"])
            if "__ndarray__" in obj:
                dtype = obj.get("__dtype__", "float64")
                return np.array(obj["__ndarray__"], dtype=dtype)
            return {k: PipelineState._deserialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [PipelineState._deserialize(v) for v in obj]
        return obj

    # ---- Файл состояния ----

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                content = self.state_file.read_text(encoding='utf-8').strip()
                if content:
                    return json.loads(content)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"State file corrupted or empty, resetting: {self.state_file} ({e})")
                backup_path = self.state_file.with_suffix('.json.bak')
                try:
                    self.state_file.rename(backup_path)
                    logger.info(f"Corrupted state file backed up to {backup_path}")
                except Exception:
                    self.state_file.unlink(missing_ok=True)
        return {"stages": {}, "created": datetime.now().isoformat()}

    def _save_state(self, state: dict):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def is_completed(self, stage: str) -> bool:
        return self._load_state().get("stages", {}).get(stage, {}).get("completed", False)

    def mark_completed(self, stage: str):
        state = self._load_state()
        if "stages" not in state:
            state["stages"] = {}
        state["stages"][stage] = {
            "completed": True,
            "timestamp": datetime.now().isoformat()
        }
        self._save_state(state)
        logger.info(f"Этап завершён: {STAGE_LABELS.get(stage, stage)}")

    def mark_incomplete(self, stage: str):
        state = self._load_state()
        state.get("stages", {}).pop(stage, None)
        self._save_state(state)

    def get_resume_point(self) -> Optional[str]:
        for stage in STAGES:
            if not self.is_completed(stage):
                return stage
        return None

    def print_status(self):
        state = self._load_state()
        print("\nСостояние пайплайна:")
        print("-" * 55)
        for i, stage in enumerate(STAGES, 1):
            info = state.get("stages", {}).get(stage, {})
            done = info.get("completed", False)
            ts = info.get("timestamp", "—")
            label = STAGE_LABELS[stage]
            mark = "OK" if done else "  "
            print(f"  {i}. [{mark}] {stage}  {label}  ({ts})")
        resume = self.get_resume_point()
        if resume:
            print(f"\n  Возобновление с: {resume} ({STAGE_LABELS[resume]})")
        else:
            print("\n  Все этапы завершены.")
        print("-" * 55)

    # ---- Сохранение данных этапов ----

    def save_data(self, stage: str, data):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if stage == "stage_2_process_formats":
            # Данные с эмбеддингами — metadata в JSON, векторы в npz
            self._save_embeddings_data(data)
        elif stage == "stage_3_cluster_refine":
            self._save_cluster_data(data)
        else:
            path = DATA_DIR / f"{stage}.json"
            serialized = self._serialize(data)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)

    def load_data(self, stage: str):
        if stage == "stage_2_process_formats":
            return self._load_embeddings_data()
        elif stage == "stage_3_cluster_refine":
            return self._load_cluster_data()
        else:
            path = DATA_DIR / f"{stage}.json"
            if not path.exists():
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._deserialize(data)

    # ---- Состояние подэтапов этапа 2 (группы форматов) ----

    def save_stage2_groups(self, groups: Dict[str, Dict]):
        """Сохраняет под-статусы групп этапа 2 (раздельные подэтапы извлечение/эмбеддинги).

        groups: {format_type: {"extract": str, "embed": str,
                               "extract_stats": {...}, "embed_stats": {...}}}
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "stage_2_groups.json"
        data = {"version": 2, "groups": groups}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Сохранены под-статусы групп этапа 2: {list(groups.keys())}")

    def load_stage2_groups(self) -> Dict[str, Dict]:
        """Загружает под-статусы групп этапа 2.

        Возвращает dict {format_type: {extract, embed, extract_stats, embed_stats}}.
        Понимает старый формат ({"completed": [...], "stats": {...}}) и конвертирует
        его: каждый завершённый формат → извлечение и эмбеддинги "completed".
        """
        path = DATA_DIR / "stage_2_groups.json"
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return {}

        # Новый формат
        if isinstance(data, dict) and "groups" in data:
            return data.get("groups", {})

        # Старый формат → конвертация
        result: Dict[str, Dict] = {}
        if isinstance(data, dict) and "completed" in data:
            stats = data.get("stats", {})
            for fmt in data.get("completed", []):
                s = stats.get(fmt, {})
                result[fmt] = {
                    "extract": "completed",
                    "embed": "completed",
                    "extract_stats": {"ok": s.get("ok", 0), "errors": s.get("errors", 0), "elapsed": s.get("elapsed", 0.0)},
                    "embed_stats": {"ok": 0, "errors": 0, "elapsed": 0.0},
                }
        return result

    def load_stage2_stats(self) -> Dict[str, Dict]:
        """Устарело: статистика теперь хранится внутри load_stage2_groups()."""
        groups = self.load_stage2_groups()
        return {fmt: g.get("extract_stats", {}) for fmt, g in groups.items()}

    def clear_stage2_groups(self):
        """Очищает список завершённых групп этапа 2 (при откате)."""
        path = DATA_DIR / "stage_2_groups.json"
        if path.exists():
            path.unlink()
            logger.debug("Очищен список завершённых групп этапа 2")

    def _save_embeddings_data(self, data: List[Dict]):
        metadata = []
        text_embs = []
        text_emb_indices = []
        img_embs = []
        img_emb_indices = []

        for i, item in enumerate(data):
            meta = {}
            for k, v in item.items():
                if k == 'text_embedding':
                    text_embs.append(np.asarray(v))
                    text_emb_indices.append(i)
                    meta['has_text_embedding'] = True
                elif k == 'image_embedding':
                    img_embs.append(np.asarray(v))
                    img_emb_indices.append(i)
                    meta['has_image_embedding'] = True
                elif k == 'text_cluster':
                    meta['text_cluster'] = int(v)
                elif k == 'image_cluster':
                    meta['image_cluster'] = int(v)
                elif k == 'combined_cluster':
                    meta['combined_cluster'] = int(v)
                else:
                    meta[k] = self._serialize(v)
            metadata.append(meta)

        meta_path = DATA_DIR / "stage_2_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        save_dict = {}
        if text_embs:
            save_dict['text_embeddings'] = np.vstack(text_embs)
            save_dict['text_indices'] = np.array(text_emb_indices)
        if img_embs:
            save_dict['img_embeddings'] = np.vstack(img_embs)
            save_dict['img_indices'] = np.array(img_emb_indices)

        np.savez_compressed(DATA_DIR / "stage_2_embeddings.npz", **save_dict)

    def _load_embeddings_data(self) -> Optional[List[Dict]]:
        meta_path = DATA_DIR / "stage_2_metadata.json"
        emb_path = DATA_DIR / "stage_2_embeddings.npz"

        if not meta_path.exists():
            return None

        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data = []
        for item in metadata:
            d = {}
            for k, v in item.items():
                d[k] = self._deserialize(v) if isinstance(v, (dict, list)) else v
            data.append(d)

        if emb_path.exists():
            npz = np.load(emb_path, allow_pickle=True)

            if 'text_embeddings' in npz.files and 'text_indices' in npz.files:
                text_embs = npz['text_embeddings']
                text_idx = npz['text_indices']
                for pos, idx in enumerate(text_idx):
                    idx = int(idx)
                    if idx < len(data):
                        data[idx]['text_embedding'] = text_embs[pos]

            if 'img_embeddings' in npz.files and 'img_indices' in npz.files:
                img_embs = npz['img_embeddings']
                img_idx = npz['img_indices']
                for pos, idx in enumerate(img_idx):
                    idx = int(idx)
                    if idx < len(data):
                        data[idx]['image_embedding'] = img_embs[pos]

        return data

    def _save_cluster_data(self, data: List[Dict]):
        cluster_assignments = []
        moved_files = []

        for item in data:
            entry = {
                'source': str(item.get('source', '')),
                'type': item.get('type', ''),
            }
            for key in ('text_cluster', 'image_cluster', 'combined_cluster', 'final_cluster'):
                if key in item:
                    entry[key] = int(item[key])
            cluster_assignments.append(entry)

            if 'moved_to' in item:
                moved_files.append({
                    'source': str(item['source']),
                    'moved_to': str(item['moved_to']),
                })

        save_obj = {
            'cluster_assignments': cluster_assignments,
            'moved_files': moved_files,
        }

        path = DATA_DIR / "stage_3_cluster.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=2)

    def _load_cluster_data(self) -> Optional[Dict]:
        path = DATA_DIR / "stage_3_cluster.json"
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    # ---- Откат этапов ----

    def rollback_from(self, from_stage: str, soft: bool = False):
        if from_stage not in STAGES:
            raise ValueError(f"Неизвестный этап: {from_stage}")

        idx = STAGES.index(from_stage)
        mode_label = " (мягкий)" if soft else ""
        print(f"\nОткат начиная с этапа {idx + 1}: {STAGE_LABELS[from_stage]}{mode_label}")
        print("Будут отменены этапы:", ", ".join(
            f"{i + 1}. {STAGE_LABELS[s]}" for i, s in enumerate(STAGES) if i >= idx
        ))

        for i in range(len(STAGES) - 1, idx - 1, -1):
            stage = STAGES[i]
            self._rollback_stage(stage)
            self.mark_incomplete(stage)

        print(f"Откат завершён. Следующий запуск начнётся с этапа {idx + 1}.")

    def _rollback_stage(self, stage: str):
        logger.info(f"Откат этапа: {stage} ({STAGE_LABELS[stage]})")

        if stage == "stage_1_sort":
            all_target_dirs = set(cfg.TARGETS.values())
            for rel in cfg.FORMAT_TARGETS.values():
                all_target_dirs.add(cfg.ROOT / rel)
            for target_dir in all_target_dirs:
                if target_dir.exists():
                    file_count = 0
                    for f in target_dir.rglob("*"):
                        if f.is_file():
                            dest = cfg.SOURCE_DIR / f.name
                            if not dest.exists():
                                shutil.move(str(f), str(dest))
                            else:
                                cnt = 1
                                stem = f.stem
                                suff = f.suffix
                                while dest.exists():
                                    dest = cfg.SOURCE_DIR / f"{stem}_{cnt}{suff}"
                                    cnt += 1
                                shutil.move(str(f), str(dest))
                            file_count += 1
                    remaining = sum(1 for _ in target_dir.rglob("*") if _.is_file())
                    if remaining == 0:
                        _force_remove_dir(target_dir)
                    logger.info(f"  Возвращено в SourceFiles: {file_count} файлов из {target_dir}")
            try:
                from database import DatabaseManager
                db = DatabaseManager()
                db.clear_stage1()
                db.close()
                logger.info("  БД очищена (stage 1)")
            except Exception as e:
                logger.warning(f"  Не удалось очистить БД: {e}")

        elif stage == "stage_2_process_formats":
            # Комбинированный откат: извлечение + LLM саммаризация + эмбеддинги
            # 1. Возврат файлов из FailedExtraction/ErrorFiles в Sorted/{format}
            if cfg.FAILED_EXTRACTION_DIR.exists():
                from file_processor import classify_pdf
                restored = 0
                for f in list(cfg.FAILED_EXTRACTION_DIR.iterdir()):
                    if not f.is_file():
                        continue
                    ext = f.suffix.lower()
                    if ext in cfg.PDF_FORMATS:
                        cat = classify_pdf(f)
                    elif ext in cfg.EXCEL_FORMATS:
                        cat = "excel"
                    elif ext in cfg.WORD_FORMATS:
                        cat = "word"
                    elif ext in cfg.IMAGE_FORMATS:
                        cat = "image"
                    elif ext in cfg.XML_FORMATS:
                        cat = "xml"
                    else:
                        cat = None
                    if cat:
                        target_dir = cfg.TARGETS.get(cat)
                        if target_dir:
                            dst = target_dir / f.name
                            cnt = 1
                            while dst.exists():
                                dst = target_dir / f"{f.stem}_{cnt}{f.suffix}"
                                cnt += 1
                            shutil.move(str(f), str(dst))
                            restored += 1
                if restored:
                    logger.info(f"  Возвращено из FailedExtraction: {restored} файлов")
                remaining = [f for f in cfg.FAILED_EXTRACTION_DIR.iterdir() if f.is_file()]
                if not remaining:
                    _force_remove_dir(cfg.FAILED_EXTRACTION_DIR)
                else:
                    logger.warning(f"  В FailedExtraction осталось {len(remaining)} нераспознанных файлов")
            # 2. Удаление извлечённых артефактов (PDF_Images, Office_PDF)
            if cfg.EXTRACT_ROOT.exists():
                file_count = sum(1 for _ in cfg.EXTRACT_ROOT.rglob("*") if _.is_file())
                if _force_remove_dir(cfg.EXTRACT_ROOT):
                    logger.info(f"  Удалено: {cfg.EXTRACT_ROOT} ({file_count} файлов)")
            # 3. Удаление .npy эмбеддингов
            if cfg.EMBEDDINGS_DIR.exists():
                file_count = sum(1 for _ in cfg.EMBEDDINGS_DIR.rglob("*") if _.is_file())
                if _force_remove_dir(cfg.EMBEDDINGS_DIR):
                    logger.info(f"  Удалено: {cfg.EMBEDDINGS_DIR} ({file_count} файлов)")
                cfg.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            # 4. Очистка БД: извлечение + саммаризация + эмбеддинги
            try:
                from database import DatabaseManager
                db = DatabaseManager()
                db.clear_stage2()   # text, image_path, error, stage_2_done
                db.clear_stage3()   # enriched_text, topic, doc_type, purpose, stage_3_done
                db.clear_stage4()   # text_embeddings, image_embeddings, flags
                db.close()
                logger.info("  БД очищена (извлечение + саммаризация + эмбеддинги)")
            except Exception as e:
                logger.warning(f"  Не удалось очистить БД: {e}")
            # 5. Сброс списка завершённых групп
            self.clear_stage2_groups()

        elif stage == "stage_3_cluster_refine":
            # Возврат файлов из кластеров в исходные позиции
            moved_files = self._load_move_mappings()
            if moved_files:
                restored = 0
                for mapping in moved_files:
                    src = Path(mapping['moved_to'])
                    dst = Path(mapping['source'])
                    if src.exists() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(str(src), str(dst))
                            restored += 1
                        except Exception as e:
                            logger.error(f"  Ошибка возврата {src.name}: {e}")
                if restored > 0:
                    logger.info(f"  Возвращено файлов: {restored}")

            clusters_dir = cfg.ROOT / "Clusters"
            if clusters_dir.exists():
                if _force_remove_dir(clusters_dir):
                    logger.info(f"  Удалено: {clusters_dir}")

            central_dir = cfg.ROOT / "CentralDocuments"
            if central_dir.exists():
                if _force_remove_dir(central_dir):
                    logger.info(f"  Удалено: {central_dir}")

            report = cfg.ROOT / "report.txt"
            if report.exists():
                report.unlink()
                logger.info(f"  Удалено: {report}")

        self._delete_stage_data(stage)

    def _load_move_mappings(self) -> Optional[List[Dict]]:
        path = DATA_DIR / "stage_3_cluster.json"
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('moved_files', [])
        except Exception:
            return None

    def _delete_stage_data(self, stage: str):
        if stage == "stage_2_process_formats":
            for name in ["stage_2_metadata.json", "stage_2_embeddings.npz"]:
                f = DATA_DIR / name
                if f.exists():
                    f.unlink()
            try:
                from database import DatabaseManager
                db = DatabaseManager()
                db.clear_stage4()
                db.close()
            except Exception:
                pass
        elif stage == "stage_3_cluster_refine":
            f = DATA_DIR / "stage_3_cluster.json"
            if f.exists():
                f.unlink()
        else:
            f = DATA_DIR / f"{stage}.json"
            if f.exists():
                f.unlink()

    def full_reset(self, test_mode: bool = False):
        logger.info("Начат полный сброс пайплайна")

        rar_path = cfg.SOURCE_RAR_TEST if test_mode else cfg.SOURCE_RAR
        if not rar_path.exists():
            raise FileNotFoundError(
                f"Архив не найден: {rar_path}. "
                f"Full reset отменён, чтобы не удалить исходные файлы."
            )

        try:
            from database import DatabaseManager
            db = DatabaseManager()
            db.clear_all()
            db.close()
            logger.info("  БД полностью очищена")
        except Exception as e:
            logger.warning(f"  Не удалось очистить БД: {e}")

        logger.info("  Закрытие процессов LibreOffice/Office...")
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'soffice.exe'],
                         capture_output=True, timeout=5)
            subprocess.run(['taskkill', '/F', '/IM', 'soffice.bin'],
                         capture_output=True, timeout=5)
            subprocess.run(['taskkill', '/F', '/IM', 'WINWORD.EXE'],
                         capture_output=True, timeout=5)
            subprocess.run(['taskkill', '/F', '/IM', 'EXCEL.EXE'],
                         capture_output=True, timeout=5)
            time.sleep(1)
        except Exception as e:
            logger.debug(f"  taskkill error: {e}")

        dirs_to_delete = []
        for target_dir in cfg.TARGETS.values():
            if target_dir.exists():
                dirs_to_delete.append(target_dir)
        for format_target in cfg.FORMAT_TARGETS.values():
            format_dir = cfg.ROOT / format_target
            if format_dir.exists() and format_dir not in dirs_to_delete:
                dirs_to_delete.append(format_dir)
        if cfg.EXTRACT_ROOT.exists():
            dirs_to_delete.append(cfg.EXTRACT_ROOT)
        if cfg.EMBEDDINGS_DIR.exists():
            dirs_to_delete.append(cfg.EMBEDDINGS_DIR)
        errors_dir = cfg.ERRORS_DIR
        if errors_dir.exists():
            dirs_to_delete.append(errors_dir)
        clusters_dir = cfg.ROOT / "Clusters"
        if clusters_dir.exists():
            dirs_to_delete.append(clusters_dir)
        central_dir = cfg.ROOT / "CentralDocuments"
        if central_dir.exists():
            dirs_to_delete.append(central_dir)
        if DATA_DIR.exists():
            dirs_to_delete.append(DATA_DIR)
        source_dir = cfg.SOURCE_DIR
        if source_dir.exists():
            dirs_to_delete.append(source_dir)

        for d in dirs_to_delete:
            try:
                if _force_remove_dir(d):
                    logger.info(f"  Удалено: {d}")
                else:
                    logger.warning(f"  Не удалось полностью удалить: {d}")
            except Exception as e:
                logger.error(f"  Ошибка удаления {d}: {e}")

        report = cfg.ROOT / "report.txt"
        if report.exists():
            report.unlink()
            logger.info(f"  Удалено: {report}")

        rar_label = "тестового" if test_mode else "продакшен"
        logger.info(f"  Восстановление исходных файлов из {rar_label} архива...")
        restored = self._restore_from_rar(rar_path)
        if restored:
            logger.info(f"  Восстановлено файлов из архива: {restored}")
        else:
            raise RuntimeError("Не удалось восстановить исходные файлы из архива")

        logger.info("Полный сброс завершён.")

    def _restore_from_rar(self, rar_path: Path = None) -> int:
        if rar_path is None:
            rar_path = SOURCE_RAR
        if not rar_path.exists():
            logger.warning(f"Архив не найден: {rar_path}")
            return 0
        try:
            result = subprocess.run(
                [str(UNRAR_PATH), "x", "-o+", str(rar_path), str(cfg.ROOT) + "\\"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"UnRAR ошибка: {result.stderr}")
                return 0
            if cfg.SOURCE_DIR.exists():
                return sum(1 for _ in cfg.SOURCE_DIR.rglob("*") if _.is_file())
            return 0
        except Exception as e:
            logger.error(f"Ошибка распаковки архива: {e}")
            return 0
