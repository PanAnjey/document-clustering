# sorting/pdf_processor.py
# Обработка PDF файлов на этапе 1: проверка заголовка, PKCS#7, классификация

import shutil
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from file_processor import classify_pdf, move_file_to_error, move_file_to_target
from logger_utils import logger


def process_pdf(file_path: Path) -> Optional[Tuple[Path, str]]:
    """Обработка PDF файла.
    
    Returns:
        (new_path, category) — если успешно
        None — если файл отправлен в ErrorFiles
    """
    ext = file_path.suffix.lower()
    if ext != '.pdf':
        return None

    try:
        # 1. Проверка заголовка и PKCS#7
        with open(file_path, 'rb') as fh:
            header = fh.read(14)

        pkcs7_sig = b'\x30\x80\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07'
        if len(header) >= 13 and header[:12] == pkcs7_sig:
            cms_type = header[12]
            label = "SignedData" if cms_type == 0x02 else "EnvelopedData"
            logger.warning(f"PKCS#7 {label}: {file_path.name}")
            move_file_to_error(file_path)
            return None

        if header[:4] != b'%PDF':
            logger.warning(f"Не PDF заголовок: {file_path.name}")
            move_file_to_error(file_path)
            return None

        # 2. Перемещаем в Sorted/PDF
        moved = move_file_to_target(file_path, "pdf")
        if not moved:
            return None

        # 3. Классификация PDF на подкатегории
        subcat = classify_pdf(moved)
        sorted_dir = cfg.TARGETS.get(subcat)
        if sorted_dir:
            sorted_dir.mkdir(parents=True, exist_ok=True)
            dst = sorted_dir / moved.name
            cnt = 1
            while dst.exists():
                dst = sorted_dir / f"{moved.stem}_{cnt}{moved.suffix}"
                cnt += 1
            shutil.move(str(moved), str(dst))
            moved = dst

        return (moved, subcat)

    except Exception as e:
        logger.error(f"PDF processing error {file_path.name}: {e}")
        move_file_to_error(file_path)
        return None
