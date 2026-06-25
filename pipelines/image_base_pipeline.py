# pipelines/image_base_pipeline.py
# Базовый пайплайн для изображений: Tesseract OCR + nomic-embed-vision

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline


class ImageBasePipeline(BasePipeline):
    """Базовый пайплайн для изображений (Tesseract OCR)."""
    
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
            from extractors.image_extractor import extract_image
            res = extract_image(file_path)
            
            if res and res.get('text'):
                result['text'] = res['text']
                result['image'] = res.get('image') or str(file_path)
                result['text_quality'] = assess_text_quality(result['text'])
            elif res and res.get('image'):
                # Только изображение, текст не извлечён
                result['image'] = res['image'] or str(file_path)
                result['text_quality'] = "none"
            else:
                result['error'] = "Empty or extraction failed"
                
        except Exception as e:
            logger.error(f"Image extraction error {file_path.name}: {e}")
            result['error'] = str(e)
        
        return result
