# extractors/xml_extractor.py
# Извлечение структуры XML и текста из OpenDocument файлов

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_xml(file_path: Path) -> Dict:
    """Извлечение текста из XML/OpenDocument файла."""
    
    base_result = {
        "source": str(file_path),
        "type": "xml",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }
    
    # Для XML файлов используем структуру XPath
    try:
        structure = _extract_xml_structure(file_path)
        if structure:
            base_result['text'] = structure
            base_result['text_quality'] = assess_text_quality(structure)
            return base_result
        
        # Для ODF файлов не читаем бинарный zip как текст
        if file_path.suffix.lower() in (".odt", ".ods", ".odp", ".odg"):
            base_result['error'] = "Empty XML structure"
            return base_result
        
        # Fallback на текст если структура пуста
        try:
            text = file_path.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                base_result['text'] = text[:8000]
                base_result['text_quality'] = assess_text_quality(text)
                return base_result
        except Exception:
            pass
        
        base_result['error'] = "Empty XML structure"
        return base_result
    except Exception as e:
        logger.error(f"XML extraction error for {file_path.name}: {e}")
        base_result['error'] = f"XML extraction error: {e}"
        return base_result


def _extract_xml_structure(xml_path: Path) -> Optional[str]:
    """Извлечение XPath дерева структуры XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        root_tag = root.tag if '}' not in root.tag else root.tag.split('}')[1]
        root_ns = root.tag.split('}')[0][1:] if '}' in root.tag else ''

        opendocument_ns = 'urn:oasis:names:tc:opendocument'
        if opendocument_ns in root_ns:
            return _extract_opendocument_xml_text(root, xml_path)

        paths = set()

        def clean_tag(tag):
            return tag.split('}')[-1] if '}' in tag else tag

        def traverse(element, current_path=""):
            tag = clean_tag(element.tag)
            new_path = f"{current_path}/{tag}" if current_path else tag
            paths.add(new_path)
            for attr in element.attrib:
                clean_attr = clean_tag(attr)
                paths.add(f"{new_path}/@{clean_attr}")
            for child in element:
                traverse(child, new_path)

        traverse(root)

        sorted_paths = sorted(list(paths))
        structure_text = " ".join(sorted_paths)

        if not structure_text:
            logger.warning(f"XML structure is empty: {xml_path.name}")
            return None

        return structure_text

    except ET.ParseError:
        ext = xml_path.suffix.lower()
        if ext in (".odt", ".ods", ".odp", ".odg"):
            logger.debug(f"ODF file {xml_path.name} is not valid XML (corrupted zip), skipping")
            return None
        logger.debug(f"XML parse error for {xml_path.name}, trying raw text fallback")
        try:
            text = xml_path.read_text(encoding='utf-8', errors='replace')
            content = text.strip()
            if content:
                return content[:8000]
            return None
        except Exception as e2:
            logger.debug(f"XML fallback read failed {xml_path.name}: {e2}")
            return None
    except Exception as e:
        logger.debug(f"XML structure extraction failed {xml_path.name}: {e}")
        return None


def _extract_opendocument_xml_text(root, xml_path: Path) -> Optional[str]:
    """Извлечение текста из OpenDocument XML."""
    office_ns = '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}'
    text_ns = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'

    body = root.find(f'{office_ns}body')
    if body is None:
        for child in root:
            if child.tag.endswith('}body') or child.tag == 'body':
                body = child
                break

    if body is None:
        logger.debug(f"XML without body, using LibreOffice fallback: {xml_path.name}")
        return _convert_opendocument_xml(xml_path)

    text_content = []
    for elem in body.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag in ('p', 'span', 'h', 'list-item') and elem.text and elem.text.strip():
            text_content.append(elem.text.strip())

    if text_content:
        result = "\n".join(text_content)[:8000]
        return result

    return _convert_opendocument_xml(xml_path)


def _convert_opendocument_xml(xml_path: Path) -> Optional[str]:
    """Конвертация OpenDocument XML через LibreOffice."""
    
    # Пробуем LibreOffice COM
    from extractors.office_com_extractor import convert_office_to_pdf
    
    temp_pdf_dir = cfg.EXTRA['office_pdf']
    temp_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    _file_type = "word" if xml_path.suffix.lower() in (".odt", ".docx", ".doc", ".rtf") else "excel"
    pdf_path = convert_office_to_pdf(xml_path, temp_pdf_dir, _file_type)
    
    if pdf_path:
        from extractors.pdf_extractor import extract_pdf
        pdf_res = extract_pdf(pdf_path)
        return pdf_res.get('text')
    return None
