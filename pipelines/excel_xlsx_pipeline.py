# pipelines/excel_xlsx_pipeline.py
# Пайплайн для .xlsx файлов: Pandoc CLI → COM Excel fallback

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("excel_xlsx")
class ExcelXlsxPipeline(BasePipeline):
    """Пайплайн для .xlsx файлов."""
    
    def extract(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }
        
        # 1. Pandoc CLI (основной метод)
        if cfg.PANDOC_ENABLED:
            text = self._extract_pandoc(file_path)
            if text and text.strip():
                result['text'] = text.strip()
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        
        # 2. COM Excel (fallback)
        try:
            from extractors.excel_extractor import extract_excel
            res = extract_excel(file_path)
            if res and res.get('text'):
                result['text'] = res['text']
                result['image'] = res.get('image')
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        except Exception as e:
            logger.warning(f"Excel extraction failed for {file_path.name}: {e}")
        
        result['error'] = "Pandoc+COM extraction failed"
        return result
    
    def _extract_pandoc(self, file_path: Path) -> Optional[str]:
        """Извлечение через Pandoc CLI."""
        try:
            from extractors.pandoc_extractor import extract_with_pandoc
            res = extract_with_pandoc(file_path, "xlsx")
            if res and res.get('text'):
                return res['text']
        except Exception as e:
            logger.warning(f"Pandoc failed for {file_path.name}: {e}")
        return None
