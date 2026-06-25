# extractors/pandoc_extractor.py
# Извлечение текста через Pandoc (основной метод для .rtf, .doc, .docx, ODF)

import subprocess
from pathlib import Path
from typing import Optional, Dict

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_with_pandoc(file_path: Path, file_type: str) -> Optional[Dict]:
    """Извлечение текста через Pandoc."""
    
    # Определение формата ввода для Pandoc
    input_format = _get_pandoc_input_format(file_type)
    if not input_format:
        return None
    
    # Пробуем CLI сначала
    result = _extract_with_pandoc_cli(file_path, input_format, file_type)
    if result:
        return result
    
    # Fallback на Python библиотеку если CLI не справился
    if cfg.PANDOC_PYTHON_FALLBACK:
        logger.debug(f"Pandoc CLI failed, trying Python library for {file_path.name}")
        result = _extract_with_pandoc_python(file_path, input_format, file_type)
        if result:
            return result
    
    logger.debug(f"Pandoc returned empty or failed for {file_path.name}")
    return None


def _extract_with_pandoc_cli(file_path: Path, input_format: str, file_type: str = "") -> Optional[Dict]:
    """Извлечение через Pandoc CLI."""
    
    try:
        pandoc_cmd = cfg.PANDOC_PATH or "pandoc"
        result = subprocess.run(
            [pandoc_cmd, str(file_path), "-f", input_format, "-t", "markdown"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=cfg.PANDOC_TIMEOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_text = result.stderr.strip()[:200] if result.stderr else ""
        
        if result.returncode == 0 and stdout_text:
            text = _clean_markdown(result.stdout)
            
            if text and len(text) > 50:
                return {
                    "source": str(file_path),
                    "type": file_type,
                    "text": text.strip(),
                    "image": None,
                    "error": None,
                    "text_quality": assess_text_quality(text)
                }
        
        if result.returncode != 0:
            logger.debug(f"Pandoc CLI failed for {file_path.name}: exit={result.returncode} {stderr_text}")
        elif not stdout_text:
            logger.debug(f"Pandoc CLI returned empty output for {file_path.name}")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error(f"Pandoc CLI timeout for {file_path.name} ({cfg.PANDOC_TIMEOUT}s)")
        return None
    except FileNotFoundError:
        logger.debug("Pandoc CLI not found in PATH")
        return None
    except Exception as e:
        logger.warning(f"Pandoc CLI error for {file_path.name}: {e}")
        return None


def _extract_with_pandoc_python(file_path: Path, input_format: str, file_type: str = "") -> Optional[Dict]:
    """Извлечение через Python библиотеку pandoc."""
    
    try:
        import pandoc
        
        doc = pandoc.Document()
        doc.input_format = input_format
        doc.output_format = 'markdown'
        
        if file_path.suffix.lower() in ('.docx', '.odt'):
            import zipfile
            import xml.etree.ElementTree as ET
            
            if file_path.suffix.lower() == '.docx':
                with zipfile.ZipFile(file_path, 'r') as zf:
                    content_xml = zf.read('word/document.xml')
                root = ET.fromstring(content_xml)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text_parts = []
                for para in root.iter('{%s}t' % ns['w']):
                    if para.text:
                        text_parts.append(para.text.strip())
                text = '\n'.join(text_parts)
            else:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    content_xml = zf.read('content.xml')
                root = ET.fromstring(content_xml)
                ns = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
                text_parts = []
                for p in root.iter('{%s}p' % ns['text']):
                    if p.text:
                        text_parts.append(p.text.strip())
                    for child in p:
                        if child.tail:
                            text_parts.append(child.tail.strip())
                text = '\n'.join(text_parts)
            
            if text and len(text) > 50:
                return {
                    "source": str(file_path),
                    "type": file_type,
                    "text": _clean_markdown(text).strip(),
                    "image": None,
                    "error": None,
                    "text_quality": assess_text_quality(text)
                }
        
        return None
        
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Pandoc Python error for {file_path.name}: {e}")
        return None


def _get_pandoc_input_format(file_type: str) -> Optional[str]:
    """Определение формата ввода для Pandoc.
    
    Примечание: Pandoc НЕ поддерживает бинарный формат .doc.
    Для .doc используется olefile (OLE2 парсер) как основной метод.
    """
    
    format_map = {
        "rtf": "rtf",
        "docx": "docx",
        "odt": "odt",
        "ods": "ods",
        "odp": "odp",
    }
    
    return format_map.get(file_type)


def _clean_markdown(md_text: str) -> str:
    """Очистка Markdown от лишних элементов."""
    
    import re
    
    lines = md_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Удаление ссылок [текст](url) → текст
        line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
        
        # Удаление изображений ![alt](url)
        line = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', line)
        
        # Удаление HTML тегов
        line = re.sub(r'<[^>]+>', '', line)
        
        # Удаление лишних пустых строк (более 2 подряд)
        if len(cleaned_lines) >= 2 and cleaned_lines[-1].strip() == '' and cleaned_lines[-2].strip() == '':
            if line.strip() == '':
                continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
