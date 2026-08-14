# sorting/text_processor.py
# Обработка текстовых документов: Word (.docx/.doc/.rtf/.txt), ODT, ODP
# Phase 2: разбивка на подкатегории по расширению

import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from file_processor import (
    detect_opendocument_xml_type,
    is_opendocument_xml,
    move_file_to_error,
    move_file_to_target,
)
from logger_utils import logger


# Маппинг расширения → формат для Word-файлов
WORD_EXT_MAP = {
    '.docx': 'word_docx',
    '.doc': 'word_doc',
    '.rtf': 'word_rtf',
    '.txt': 'word_txt',
    '.odt': 'word_odt',
}


def _is_txt_garbage(file_path: Path) -> bool:
    """Определяет мусорные .txt файлы.

    Мусор: текст из 1-2 строк ИЛИ размер файла <= 3 Кбайт.
    """
    try:
        size_bytes = file_path.stat().st_size
        if size_bytes <= 3 * 1024:
            return True

        content = file_path.read_text(encoding='utf-8', errors='ignore')
        stripped = content.strip()
        lines = len(stripped.splitlines())
        if lines <= 2:
            return True

        return False

    except Exception:
        return False


def process_text(file_path: Path) -> Optional[Tuple[Path, str]]:
    """Обработка текстовых файлов (Word, RTF, TXT, ODT).

    Phase 2: каждый формат → отдельная папка Sorted/Word_{ext}.

    Returns:
        (new_path, category) — если успешно
        None — если файл отправлен в ErrorFiles
    """
    ext = file_path.suffix.lower()

    try:
        # 1. OpenDocument XML → переименование в .odt/.ods и маппинг на Phase2 форматы
        if ext == '.xml' and is_opendocument_xml(file_path):
            new_ext, _cat = detect_opendocument_xml_type(file_path)
            # Маппинг legacy категорий на Phase2 format_type
            cat_map = {
                'word': 'word_odt',
                'excel': 'excel_ods',
            }
            cat = cat_map.get(_cat)
            if not cat:
                logger.warning(f"Неизвестный ODF XML тип {_cat}: {file_path.name}")
                move_file_to_error(file_path)
                return None
            new_path = file_path.with_suffix(new_ext)
            try:
                file_path.rename(new_path)
            except Exception:
                new_path = file_path
            target_dir = cfg.ROOT / cfg.FORMAT_TARGETS[cat]
            target_dir.mkdir(parents=True, exist_ok=True)
            dst = target_dir / new_path.name
            cnt = 1
            while dst.exists():
                dst = target_dir / f"{new_path.stem}_{cnt}{new_path.suffix}"
                cnt += 1
            shutil.move(str(new_path), str(dst))
            return (dst, cat)

        # 2. Word форматы — разбивка по расширению
        if ext in WORD_EXT_MAP:
            format_type = WORD_EXT_MAP[ext]

            # Частный случай: TXT-файлы вида FromDepartmentID… (мусор)
            if ext == '.txt' and _is_txt_garbage(file_path):
                logger.warning(f"Отфильтрован мусор (.txt): {file_path.name}")
                move_file_to_error(file_path)
                return None

            # ODF zip integrity для .odt
            if ext == '.odt':
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        pass
                except Exception:
                    logger.warning(f"Битый ODT zip: {file_path.name}")
                    move_file_to_error(file_path)
                    return None
            
            # Перемещаем в Sorted/Word_{ext}
            target_dir = cfg.ROOT / cfg.FORMAT_TARGETS[format_type]
            target_dir.mkdir(parents=True, exist_ok=True)
            
            dst = target_dir / file_path.name
            cnt = 1
            while dst.exists():
                dst = target_dir / f"{file_path.stem}_{cnt}{file_path.suffix}"
                cnt += 1
            
            shutil.move(str(file_path), str(dst))
            return (dst, format_type)

        # 3. ODP — относим к word_odt (презентации), с проверкой zip
        if ext == '.odp':
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    pass
            except Exception:
                logger.warning(f"Битый ODP zip: {file_path.name}")
                move_file_to_error(file_path)
                return None

            target_dir = cfg.ROOT / cfg.FORMAT_TARGETS['word_odt']
            target_dir.mkdir(parents=True, exist_ok=True)

            dst = target_dir / file_path.name
            cnt = 1
            while dst.exists():
                dst = target_dir / f"{file_path.stem}_{cnt}{file_path.suffix}"
                cnt += 1

            shutil.move(str(file_path), str(dst))
            return (dst, 'word_odt')

        return None

    except Exception as e:
        logger.error(f"Text processing error {file_path.name}: {e}")
        move_file_to_error(file_path)
        return None
