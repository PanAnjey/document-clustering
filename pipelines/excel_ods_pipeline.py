# pipelines/excel_ods_pipeline.py
# Пайплайн для .ods файлов: Pandoc XML extraction

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("excel_ods")
class ExcelOdsPipeline(BasePipeline):
    """Пайплайн для .ods файлов."""
    
    def extract(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }
        
        # 1. Pandoc XML extraction (основной метод)
        try:
            from extractors.xml_extractor import extract_xml
            res = extract_xml(file_path)
            
            if res and res.get('text'):
                result['text'] = res['text']
                result['image'] = res.get('image')
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        except Exception as e:
            logger.warning(f"XML extraction failed for {file_path.name}: {e}")
        
        # 2. Pandoc CLI (fallback)
        if cfg.PANDOC_ENABLED:
            text = self._extract_pandoc(file_path)
            if text and text.strip():
                result['text'] = text.strip()
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        
        result['error'] = "All extractors failed"
        return result
    
    def _extract_pandoc(self, file_path: Path) -> Optional[str]:
        """Извлечение через Pandoc CLI."""
        try:
            from extractors.pandoc_extractor import extract_with_pandoc
            res = extract_with_pandoc(file_path, "ods")
            if res and res.get('text'):
                return res['text']
        except Exception as e:
            logger.warning(f"Pandoc failed for {file_path.name}: {e}")
        return None
