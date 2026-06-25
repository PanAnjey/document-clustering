# pipelines/pdf_tables_pipeline.py
# Пайплайн для PDF с таблицами (table_strategy="lines_strict").

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality, filter_mupdf_stderr
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("pdf_tables")
class PdfTablesPipeline(BasePipeline):
    """Пайплайн для PDF с таблицами (приоритет на структурированное извлечение)."""
    
    def extract(self, file_path: Path) -> Dict:
        """Извлечение текста и таблиц из PDF через PyMuPDF4LLM."""
        with filter_mupdf_stderr():
            return self._extract_inner(file_path)

    def _extract_inner(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }
        
        # 1. PyMuPDF4LLM с table_strategy="lines_strict" (приоритет для таблиц)
        if cfg.PYMUPDF4LLM_ENABLED:
            text = self._extract_text_pymupdf4llm_tables(file_path)
            if text and text.strip():
                result['text'] = text.strip()
                result['text_quality'] = assess_text_quality(result['text'])
                if result['text_quality'] == "good":
                    return result
        
        # 2. Fallback: fitz.get_text()
        text = self._extract_text_fitz(file_path)
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
        
        # 3. Если текст плохой — генерируем изображение первой страницы для OCR
        if result.get('text_quality') != "good":
            image_path = self._generate_first_page_image(file_path)
            if image_path:
                result['image'] = str(image_path)
        
        if not result['text'] and not result.get('image'):
            result['error'] = "Empty or extraction failed"
        
        return result
    
    def _extract_text_pymupdf4llm_tables(self, pdf_path: Path) -> Optional[str]:
        """Извлечение текста через PyMuPDF4LLM с приоритетом на таблицы."""
        try:
            import pymupdf4llm

            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                pages=[0],
                header=cfg.PYMUPDF4LLM_HEADER_FOOTER,
                footer=cfg.PYMUPDF4LLM_HEADER_FOOTER,
                use_ocr=cfg.PYMUPDF4LLM_OCR_ENABLED,
                ocr_language=cfg.TESSERACT_LANG if cfg.PYMUPDF4LLM_OCR_ENABLED else "eng",
                table_strategy="lines_strict",  # Строгая стратегия для таблиц
                page_chunks=False,
            )
            
            if md_text and md_text.strip():
                return md_text.strip()
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM tables failed for {pdf_path.name}: {e}")
        
        return None
    
    def _extract_text_fitz(self, pdf_path: Path) -> Optional[str]:
        """Быстрое извлечение текста через PyMuPDF (~2ms на файл)."""
        try:
            import fitz
            
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
    
    def _generate_first_page_image(self, pdf_path: Path) -> Optional[str]:
        """Генерация изображения первой страницы PDF."""
        try:
            import fitz
            
            doc = fitz.open(str(pdf_path))
            if doc.page_count == 0:
                doc.close()
                return None
            
            page = doc.load_page(0)
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            
            img_dir = cfg.EXTRA['pdf_images']
            img_dir.mkdir(parents=True, exist_ok=True)
            
            img_name = f"{pdf_path.stem}.png"
            img_path = img_dir / img_name
            
            cnt = 1
            while img_path.exists():
                img_path = img_dir / f"{pdf_path.stem}_{cnt}.png"
                cnt += 1
            
            pix.save(str(img_path))
            doc.close()
            
            return str(img_path)
            
        except Exception as e:
            logger.warning(f"Image generation failed for {pdf_path.name}: {e}")
            return None
