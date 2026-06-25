# sorting/excel_processor.py
# Обработка Excel файлов: .xlsx, .xls, .xlsm, .xlsb, .xltx, .xlt, .xltm, .csv, .ods
# Phase 2: разбивка на подкатегории по расширению

import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from file_processor import move_file_to_error, move_file_to_target
from logger_utils import logger


# Маппинг расширения → формат для Excel-файлов
EXCEL_EXT_MAP = {
    '.xlsx': 'excel_xlsx',
    '.xls': 'excel_xlsx',
    '.xlsm': 'excel_xlsx',
    '.xlsb': 'excel_xlsx',
    '.xltx': 'excel_xlsx',
    '.xlt': 'excel_xlsx',
    '.xltm': 'excel_xlsx',
    '.csv': 'excel_csv',
    '.ods': 'excel_ods',
}


def process_excel(file_path: Path) -> Optional[Tuple[Path, str]]:
    """Обработка Excel файлов.
    
    Phase 2: каждый формат → отдельная папка Sorted/Excel_{ext}.
    
    Returns:
        (new_path, category) — если успешно
        None — если файл отправлен в ErrorFiles
    """
    ext = file_path.suffix.lower()

    if ext not in EXCEL_EXT_MAP:
        return None

    try:
        # 1. ODF zip integrity для .ods
        if ext == '.ods':
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    pass
            except Exception:
                logger.warning(f"Битый ODS zip: {file_path.name}")
                move_file_to_error(file_path)
                return None

        # 2. Перемещаем в Sorted/Excel_{ext}
        format_type = EXCEL_EXT_MAP[ext]
        target_dir = cfg.ROOT / cfg.FORMAT_TARGETS[format_type]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        dst = target_dir / file_path.name
        cnt = 1
        while dst.exists():
            dst = target_dir / f"{file_path.stem}_{cnt}{file_path.suffix}"
            cnt += 1
        
        shutil.move(str(file_path), str(dst))
        return (dst, format_type)

    except Exception as e:
        logger.error(f"Excel processing error {file_path.name}: {e}")
        move_file_to_error(file_path)
        return None
