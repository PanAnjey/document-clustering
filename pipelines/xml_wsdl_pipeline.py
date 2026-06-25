# pipelines/xml_wsdl_pipeline.py
# Пайплайн для .wsdl файлов: WSDL parsing + text extraction

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("xml_wsdl")
class XmlWsdlPipeline(BasePipeline):
    """Пайплайн для .wsdl файлов (Web Services Description Language)."""
    
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
            from extractors.xml_extractor import extract_xml
            res = extract_xml(file_path)
            
            if res and res.get('text'):
                result['text'] = res['text']
                result['image'] = res.get('image')
                result['text_quality'] = assess_text_quality(result['text'])
            else:
                result['error'] = "Empty or extraction failed"
                
        except Exception as e:
            logger.error(f"WSDL extraction error {file_path.name}: {e}")
            result['error'] = str(e)
        
        return result
