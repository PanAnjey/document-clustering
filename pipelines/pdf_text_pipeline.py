# pipelines/pdf_text_pipeline.py
# Пайплайн для PDF с текстом (>100 символов, без таблиц).

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality, filter_mupdf_stderr
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry

# Совпадает с обрезкой при записи в documents.text (stages/stage2_processing.py:404) —
# нет смысла извлекать текст сверх этого лимита.
MAX_TEXT_CHARS = 5000


@PipelineRegistry.register("pdf_text")
class PdfTextPipeline(BasePipeline):
    """Пайплайн для PDF с текстом (>100 символов, без таблиц).
    
    Этап 1 уже классифицировал PDF как текстовый — никакого OCR/изображений.
    Только fitz.get_text() для быстрого извлечения чистого текста.
    """
    
    def extract(self, file_path: Path) -> Dict:
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

        text = self._extract_text_fitz(file_path)
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])

        if not result['text']:
            result['error'] = "Empty or extraction failed"

        return result
    
    def _extract_text_fitz(self, pdf_path: Path) -> Optional[str]:
        """Быстрое извлечение текста через PyMuPDF (~2ms на файл).

        Останавливается, как только накоплено MAX_TEXT_CHARS — читать и
        склеивать весь документ бессмысленно, если он всё равно будет
        обрезан до этого лимита при записи в БД.
        """
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            if doc.is_encrypted or doc.page_count == 0:
                doc.close()
                return None

            parts = []
            total_len = 0
            for page_num in range(doc.page_count):
                text = doc[page_num].get_text("text")
                if text and text.strip():
                    text = text.strip()
                    parts.append(text)
                    total_len += len(text)
                    if total_len >= MAX_TEXT_CHARS:
                        break

            doc.close()

            result = "\n\n".join(parts)[:MAX_TEXT_CHARS]
            return result.strip() if result.strip() else None

        except Exception as e:
            logger.warning(f"Fitz extraction failed for {pdf_path.name}: {e}")
            return None
