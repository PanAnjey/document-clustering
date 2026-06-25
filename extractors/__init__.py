# extractors/__init__.py
# Модуль извлечения текста для различных форматов файлов

import os
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from config import cfg
from logger_utils import logger

from .txt_extractor import extract_txt, extract_csv
from .excel_extractor import extract_excel
from .pdf_extractor import extract_pdf
from .image_extractor import extract_image
from .xml_extractor import extract_xml
from .pandoc_extractor import extract_with_pandoc

# Сигнатуры файлов: (смещение, байты, расширение, категория)
MAGIC_SIGNATURES = [
    (0, b'%PDF',                           '.pdf',   'pdf'),
    (0, b'\xD0\xCF\x11\xE0\xA1\xB1',      None,     'ole2'),
    (0, b'PK\x03\x04',                     None,     'zip'),
    (0, b'\xFF\xD8\xFF',                   '.jpg',   'image'),
    (0, b'\x89PNG\r\n\x1a\n',              '.png',   'image'),
    (0, b'GIF87a',                          '.gif',   'image'),
    (0, b'GIF89a',                          '.gif',   'image'),
    (0, b'BM',                              '.bmp',   'image'),
    (0, b'II*\x00',                         '.tif',   'image'),
    (0, b'MM\x00*',                         '.tif',   'image'),
    (0, b'RIFF',                            None,     'riff'),
    (0, b'{\\rtf',                          '.rtf',   'word'),
    (0, b'<?xml',                           '.xml',   'xml'),
    (0, b'\xEF\xBB\xBF<?xml',              '.xml',   'xml'),
    (0, b'\xFF\xFE<?xml',                   '.xml',   'xml'),
    (0, b'\xFE\xFF<?xml',                   '.xml',   'xml'),
    (0, b'\x1F\x8B',                        '.gz',    None),
    (0, b'Rar!\x1a\x07',                    None,     None),
    (0, b'\x30\x80\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07', None, 'pkcs7'),
]

ZIP_CONTENT_TYPES = {
    'word':  {b'word/', b'document.xml', b'document.xml.rels'},
    'excel': {b'spreadsheet', b'worksheet', b'styles.xml', b'sheet'},
    'odt':   {b'mimetype', b'content.xml', b'META-INF/manifest.xml'},
    'ods':   {b'mimetype', b'content.xml', b'META-INF/manifest.xml'},
}

ZIP_MIMETYPES = {
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'word',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
    'application/vnd.oasis.opendocument.text': 'word',
    'application/vnd.oasis.opendocument.spreadsheet': 'excel',
}


def _detect_zip_subtype(path: Path) -> Optional[Tuple[str, str]]:
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            names_bytes = '\n'.join(names).encode('utf-8', errors='replace')

            if '[Content_Types].xml' in names:
                try:
                    ct = zf.read('[Content_Types].xml').decode('utf-8', errors='replace')
                    for mime, cat in ZIP_MIMETYPES.items():
                        if mime in ct:
                            ext = '.docx' if cat == 'word' else '.xlsx'
                            return (ext, cat)
                except Exception:
                    pass

            if any(n.startswith('word/') for n in names):
                return ('.docx', 'word')
            if any(n.startswith('xl/') for n in names):
                return ('.xlsx', 'excel')
            if 'mimetype' in names:
                try:
                    mime = zf.read('mimetype').decode('utf-8', errors='replace').strip()
                    for m, cat in ZIP_MIMETYPES.items():
                        if m in mime:
                            ext = '.docx' if cat == 'word' else '.xlsx'
                            return (ext, cat)
                except Exception:
                    pass

            if any(n.startswith('META-INF/') for n in names):
                try:
                    manifest = zf.read('META-INF/manifest.xml').decode('utf-8', errors='replace')
                    if 'text' in manifest.lower():
                        return ('.odt', 'word')
                    if 'spreadsheet' in manifest.lower():
                        return ('.ods', 'excel')
                except Exception:
                    pass

            return ('.zip', None)
    except Exception:
        return None


def _detect_ole2_subtype(path: Path) -> Optional[Tuple[str, str]]:
    try:
        size = path.stat().st_size
        if size < 512:
            return None

        with open(path, 'rb') as f:
            f.seek(512)
            sector = f.read(512)

        prop_start = struct.unpack_from('<I', sector, 0)[0] if len(sector) >= 4 else 0

        OLE_STREAM_NAMES = {
            'Workbook':          ('.xls',   'excel'),
            'Book':              ('.xls',   'excel'),
            'WordDocument':     ('.doc',   'word'),
            '1Table':            ('.doc',   'word'),
            '0Table':            ('.doc',   'word'),
            'PowerPoint Document': ('.ppt', 'word'),
            'Current User':      None,
        }

        try:
            with open(path, 'rb') as f:
                data = f.read(min(size, 8192))
            for name, result in OLE_STREAM_NAMES.items():
                if result and name.encode('utf-16-le', errors='replace') in data:
                    return result
        except Exception:
            pass

        return ('.doc', 'word')
    except Exception:
        return None


def _detect_riff_subtype(path: Path) -> Optional[Tuple[str, str]]:
    try:
        with open(path, 'rb') as f:
            f.seek(8)
            subtype = f.read(4)
        if subtype == b'WEBP':
            return ('.webp', 'image')
        if subtype == b'AVI ':
            return ('.avi', None)
        return ('.avi', None)
    except Exception:
        return None


def is_opendocument_xml(file_path: Path) -> bool:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2048)
        if b'urn:oasis:names:tc:opendocument' in header:
            return True
        return False
    except Exception:
        return False


def detect_opendocument_xml_type(file_path: Path) -> Tuple[str, str]:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        root_ns = root.tag.split('}')[0][1:] if '}' in root.tag else ''
        if 'spreadsheet' in root_ns.lower():
            return ('.ods', 'excel')
        if 'presentation' in root_ns.lower():
            return ('.odp', 'word')
    except Exception:
        pass
    return ('.odt', 'word')


def _is_likely_text(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
        if not chunk:
            return False
        null_count = chunk.count(b'\x00')
        if null_count > len(chunk) * 0.1:
            return False
        try:
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            try:
                chunk.decode('cp1251')
                return True
            except UnicodeDecodeError:
                try:
                    chunk.decode('latin-1')
                    return True
                except UnicodeDecodeError:
                    return False
    except Exception:
        return False


def detect_file_type_by_signature(file_path: Path) -> Optional[Tuple[str, str]]:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(512)

        if len(header) == 0:
            return None

        for offset, magic, ext, category in MAGIC_SIGNATURES:
            if category is None and ext is None:
                continue
            if len(header) < offset + len(magic):
                continue
            if header[offset:offset + len(magic)] == magic:
                if category == 'zip':
                    result = _detect_zip_subtype(file_path)
                    if result:
                        return result
                    return ('.zip', None)

                if category == 'ole2':
                    result = _detect_ole2_subtype(file_path)
                    if result:
                        return result
                    return ('.doc', 'word')

                if category == 'pkcs7':
                    # PKCS#7/CMS подписанный PDF - считаем как PDF
                    # PyMuPDF умеет распаковывать CMS обёртку напрямую
                    logger.info(f"PKCS#7 signed file detected: {file_path.name}")
                    return ('.pdf', 'pdf')

                if category == 'riff':
                    result = _detect_riff_subtype(file_path)
                    if result:
                        return result
                    return None

                if ext and category:
                    return (ext, category)

        if _is_likely_text(file_path):
            return ('.txt', 'word')

        return None

    except Exception as e:
        logger.error(f"Signature detection error for {file_path.name}: {e}")
        return None


def move_file_to_target(file_path: Path, target_category: str) -> Optional[Path]:
    """Перемещение файла в целевую категорию."""
    try:
        target_dir = cfg.TARGETS.get(target_category)
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / file_path.name
            if file_path.resolve() == target_path.resolve():
                return target_path

            cnt = 1
            while target_path.exists():
                target_path = target_dir / f"{file_path.stem}_{cnt}{file_path.suffix}"
                cnt += 1
            shutil.move(str(file_path), str(target_path))
            return target_path
    except Exception as e:
        logger.error(f"Move error {file_path}: {e}")
    return None


def move_file_to_error(file_path: Path) -> Optional[Path]:
    """Перемещение файла в ErrorFiles."""
    try:
        cfg.ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        target = cfg.ERRORS_DIR / file_path.name
        cnt = 1
        while target.exists():
            target = cfg.ERRORS_DIR / f"{file_path.stem}_{cnt}{file_path.suffix}"
            cnt += 1
        shutil.move(str(file_path), str(target))
        return target
    except Exception as e:
        logger.error(f"Error moving to errors {file_path}: {e}")
    return None



def _error_result(file_path: Path, file_type: str, error: str) -> Dict:
    return {
        "source": str(file_path), "type": file_type,
        "text": None, "image": None,
        "error": error, "text_quality": "none"
    }


def process_file(file_path: Path, file_type: str) -> Dict:
    """Универсальная функция для извлечения текста из файла любого типа.
    
    Приоритет: зарегистрированный пайплайн через PipelineRegistry → fallback по расширению.
    """
    from pipelines.pipeline_registry import PipelineRegistry

    # 1. Зарегистрированный пайплайн по file_type (этап 1 уже определил тип)
    if file_type and PipelineRegistry.is_registered(file_type):
        pipeline = PipelineRegistry.get(file_type)
        return pipeline.extract(file_path)

    # 2. Fallback по расширению (для форматов без регистрации)
    ext = file_path.suffix.lower()
    
    if ext == ".txt":
        return extract_txt(file_path)
    elif ext == ".csv":
        return extract_csv(file_path)
    elif ext == ".rtf":
        return _error_result(file_path, "rtf", "RTF without registered pipeline")
    elif ext == ".doc":
        return _error_result(file_path, "doc", "DOC without registered pipeline")
    elif ext == ".docx":
        return _error_result(file_path, "docx", "DOCX without registered pipeline")
    elif ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"):
        return extract_image(file_path)
    elif ext in (".xml", ".xsd", ".xsl", ".xslt", ".wsdl"):
        return extract_xml(file_path)
    elif ext == ".odt":
        return _error_result(file_path, "odt", "ODT without registered pipeline")
    elif ext == ".ods":
        return extract_xml(file_path)
    elif ext == ".odp":
        return extract_xml(file_path)
    elif ext in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        result = extract_with_pandoc(file_path, "xlsx")
        if result and result.get('text'):
            return result
        return extract_excel(file_path)
    
    return {
        "source": str(file_path),
        "type": file_type or "unknown",
        "text": None,
        "image": None,
        "error": f"Unknown format: {ext}",
        "text_quality": "none"
    }
