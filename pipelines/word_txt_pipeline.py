# pipelines/word_txt_pipeline.py
# Пайплайн для .txt файлов: прямое чтение

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("word_txt")
class WordTxtPipeline(BasePipeline):
    """Пайплайн для .txt файлов (прямое чтение)."""
    
    def extract(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }
        
        try:
            from extractors.txt_extractor import extract_txt
            res = extract_txt(file_path)
            
            if res and res.get('text'):
                result['text'] = res['text']
                result['text_quality'] = assess_text_quality(result['text'])
            else:
                result['error'] = "Empty or extraction failed"
                
        except Exception as e:
            logger.error(f"TXT extraction error {file_path.name}: {e}")
            result['error'] = str(e)
        
        return result
