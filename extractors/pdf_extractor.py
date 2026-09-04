# extractors/pdf_extractor.py
# Извлечение текста из PDF файлов через PyMuPDF4LLM + OCR

from pathlib import Path
from typing import Optional, Dict

import fitz  # PyMuPDF

from config import cfg
from logger_utils import logger, assess_text_quality, filter_mupdf_stderr


def _suppress_mupdf_warnings():
    try:
        fitz.TOOLS.mupdf_warnings()
    except Exception:
        pass


def extract_pdf(file_path: Path) -> Dict:
    """Извлечение текста и изображений из PDF файла."""
    result = {
        "source": str(file_path),
        "type": "pdf",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    # Фильтруем только benign-сообщение MuPDF на всё время обработки PDF;
    # остальной stderr (включая другие ошибки MuPDF) сохраняется.
    with filter_mupdf_stderr():
        return _extract_pdf_inner(file_path, result)


def _extract_pdf_inner(file_path: Path, result: Dict) -> Dict:
    text_p4llm = _extract_text_hybrid(file_path)
    if text_p4llm:
        result['text'] = text_p4llm
        result['text_quality'] = assess_text_quality(result['text'])
        if result['text_quality'] == "good":
            return result

    try:
        doc = fitz.open(str(file_path))
        try:
            if doc.is_encrypted or doc.page_count == 0:
                return result

            if not result['text']:
                page = doc.load_page(0)
                text = page.get_text("text")
                if text:
                    result['text'] = text.strip()
                    result['text_quality'] = assess_text_quality(result['text'])

            if assess_text_quality(result.get('text')) != "good":
                from utils.rotate_fix import render_pdf_first_page_upright

                img_dir = cfg.EXTRA['pdf_images']
                img_dir.mkdir(parents=True, exist_ok=True)

                img_name = f"{file_path.stem}.png"
                img_path = img_dir / img_name

                cnt = 1
                while img_path.exists():
                    img_path = img_dir / f"{file_path.stem}_{cnt}.png"
                    cnt += 1

                rendered = render_pdf_first_page_upright(file_path, img_path, zoom=2.0)
                if rendered:
                    result['image'] = rendered
        finally:
            doc.close()
            _suppress_mupdf_warnings()

    except Exception as e:
        logger.error(f"PDF processing error {file_path.name}: {e}")

    if not result['text'] and not result.get('image'):
        result['error'] = "Empty or extraction failed"

    return result


def _has_tables(pdf_path: Path) -> bool:
    """Быстрая проверка наличия таблиц на первой странице (~2ms)."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        tables = page.find_tables()
        has = tables and len(tables.tables) > 0
        doc.close()
        return has
    except Exception:
        return True


def _extract_text_hybrid(pdf_path: Path) -> Optional[str]:
    """Гибридное извлечение: PyMuPDF4LLM для PDF с таблицами, fitz для простых."""
    if not cfg.PYMUPDF4LLM_ENABLED:
        return _extract_text_fitz(pdf_path)

    has_tables = _has_tables(pdf_path)

    if has_tables:
        try:
            import pymupdf4llm
        except ImportError:
            logger.debug("PyMuPDF4LLM not available")
            return _extract_text_fitz(pdf_path)

        try:
            use_ocr = cfg.PYMUPDF4LLM_OCR_ENABLED
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                header=cfg.PYMUPDF4LLM_HEADER_FOOTER,
                footer=cfg.PYMUPDF4LLM_HEADER_FOOTER,
                use_ocr=use_ocr,
                ocr_language=cfg.TESSERACT_LANG if use_ocr else "eng",
                table_strategy=cfg.PYMUPDF4LLM_TABLE_STRATEGY,
                page_chunks=False,
            )
            if md_text and md_text.strip():
                return md_text.strip()
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM failed for {pdf_path.name}: {e}, using fitz fallback")

    else:
        logger.debug(f"No tables detected, using fast fitz extraction: {pdf_path.name}")

    return _extract_text_fitz(pdf_path)


def _extract_text_fitz(pdf_path: Path) -> Optional[str]:
    """Быстрое извлечение текста через PyMuPDF (~2ms на файл)."""
    try:
        doc = fitz.open(str(pdf_path))
        if doc.is_encrypted or doc.page_count == 0:
            doc.close()
            return None

        parts = []
        for page_num in range(doc.page_count):
            text = doc[page_num].get_text("text")
            if text and text.strip():
                parts.append(text.strip())
        doc.close()

        result = "\n\n".join(parts)
        return result.strip() if result.strip() else None
    except Exception as e:
        logger.warning(f"Fitz extraction failed for {pdf_path.name}: {e}")
        return None
