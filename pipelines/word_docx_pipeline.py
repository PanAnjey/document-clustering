# pipelines/word_docx_pipeline.py
# Пайплайн для .docx файлов: Aspose(Java) → COM Word (fallback)

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("word_docx")
class WordDocxPipeline(BasePipeline):
    """Пайплайн для .docx файлов."""
    
    def extract(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }
        
        # 1. Aspose(Java) — основной метод
        text = self._extract_aspose(file_path)
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
            return result
        
        # 2. COM Word (fallback)
        if cfg.COM_ENABLED:
            try:
                from extractors.office_com_extractor import precheck_com, extract_word_com
                if precheck_com():
                    com_result = extract_word_com(file_path)
                    if com_result and not com_result.get("error"):
                        result['text'] = com_result.get('text')
                        result['image'] = com_result.get('image')
                        result['text_quality'] = assess_text_quality(result['text'])
                        return result
            except Exception:
                pass
        
        result['error'] = "Aspose+COM extraction failed"
        return result
    
    def _extract_aspose(self, file_path: Path) -> Optional[str]:
        """Извлечение текста через Aspose.Words for Java."""
        try:
            from extractors.aspose_extractor import extract_word_aspose
            res = extract_word_aspose(file_path)
            if res and res.get('text'):
                return res['text']
        except Exception as e:
            logger.warning(f"Aspose failed for {file_path.name}: {e}")
        return None
