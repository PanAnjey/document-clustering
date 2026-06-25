# sorting/xml_processor.py
# Обработка XML файлов: .xml, .xsd, .xsl, .xslt, .wsdl (не ODF)
# Phase 2: разбивка на подкатегории по типу

import shutil
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from file_processor import is_opendocument_xml, move_file_to_error, move_file_to_target
from logger_utils import logger


# Маппинг расширения → формат для XML-файлов
XML_EXT_MAP = {
    '.xml': 'xml_xml',
    '.xsd': 'xml_xsd',
    '.xsl': 'xml_xsl',
    '.xslt': 'xml_xsl',
    '.wsdl': 'xml_wsdl',
}


def process_xml(file_path: Path) -> Optional[Tuple[Path, str]]:
    """Обработка XML файлов.
    
    Phase 2: каждый формат → отдельная папка Sorted/XML_{ext}.
    
    Returns:
        (new_path, category) — если успешно
        None — если файл отправлен в ErrorFiles или обработан как ODF
    """
    ext = file_path.suffix.lower()

    if ext not in XML_EXT_MAP:
        return None

    try:
        # 1. OpenDocument XML → выносится в text_processor
        if is_opendocument_xml(file_path):
            logger.debug(f"ODF XML пропущен (обработается как ODT/ODS): {file_path.name}")
            return None

        # 2. Перемещаем в Sorted/XML_{ext}
        format_type = XML_EXT_MAP[ext]
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
        logger.error(f"XML processing error {file_path.name}: {e}")
        move_file_to_error(file_path)
        return None
