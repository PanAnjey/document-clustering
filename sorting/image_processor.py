# sorting/image_processor.py
# Обработка изображений: jpg, jpeg, png, gif, bmp, tiff, tif, webp, svg, jfif
# Phase 2: разбивка на подкатегории по расширению

import shutil
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from file_processor import move_file_to_error, move_file_to_target
from logger_utils import logger


# Маппинг расширения → формат для Image-файлов
IMAGE_EXT_MAP = {
    '.jpg': 'image_jpg',
    '.jpeg': 'image_jpg',
    '.png': 'image_png',
    '.gif': 'image_gif',
    '.bmp': 'image_bmp',
    '.tiff': 'image_tiff',
    '.tif': 'image_tiff',
    '.webp': 'image_webp',
    '.svg': 'image_webp',
    '.jfif': 'image_jpg',
}


def process_image(file_path: Path) -> Optional[Tuple[Path, str]]:
    """Обработка файлов изображений.
    
    Phase 2: каждый формат → отдельная папка Sorted/Image_{ext}.
    
    Returns:
        (new_path, category) — если успешно
        None — если файл отправлен в ErrorFiles
    """
    ext = file_path.suffix.lower()

    if ext not in IMAGE_EXT_MAP:
        return None

    try:
        # 1. Перемещаем в Sorted/Image_{ext}
        format_type = IMAGE_EXT_MAP[ext]
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
        logger.error(f"Image processing error {file_path.name}: {e}")
        move_file_to_error(file_path)
        return None
