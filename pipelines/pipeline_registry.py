# pipelines/pipeline_registry.py
# Регистрация и загрузка пайплайнов по format_type.

from typing import Dict, Type
from .base_pipeline import BasePipeline


class PipelineRegistry:
    """Регистрация и загрузка пайплайнов для каждого формата."""
    
    _registry: Dict[str, Type[BasePipeline]] = {}
    
    @classmethod
    def register(cls, format_type: str):
        """Декоратор для регистрации пайплайна.
        
        Usage:
            @PipelineRegistry.register("pdf_text")
            class PdfTextPipeline(BasePipeline):
                ...
        """
        def decorator(pipeline_class: Type[BasePipeline]):
            cls._registry[format_type] = pipeline_class
            return pipeline_class
        return decorator
    
    @classmethod
    def get(cls, format_type: str) -> BasePipeline:
        """Возвращает экземпляр пайплайна для данного формата."""
        if format_type not in cls._registry:
            raise ValueError(f"Пайплайн не найден для формата: {format_type}")
        
        pipeline_class = cls._registry[format_type]
        return pipeline_class(format_type)
    
    @classmethod
    def get_all_formats(cls) -> list:
        """Возвращает список всех зарегистрированных форматов."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, format_type: str) -> bool:
        """Проверяет, зарегистрирован ли пайплайн для данного формата."""
        return format_type in cls._registry


# Инициализация регистрации PDF-пайплайнов (Phase 1: PoC)
from .pdf_text_pipeline import PdfTextPipeline
from .pdf_scan_pipeline import PdfScanPipeline
from .pdf_tables_pipeline import PdfTablesPipeline

# Phase 2: Word пайплайны
from .word_docx_pipeline import WordDocxPipeline
from .word_doc_pipeline import WordDocPipeline
from .word_rtf_pipeline import WordRtfPipeline
from .word_txt_pipeline import WordTxtPipeline
from .word_odt_pipeline import WordOdtPipeline

# Phase 2: Excel пайплайны
from .excel_xlsx_pipeline import ExcelXlsxPipeline
from .excel_csv_pipeline import ExcelCsvPipeline
from .excel_ods_pipeline import ExcelOdsPipeline

# Phase 2: Image пайплайны
from .image_jpg_pipeline import ImageJpgPipeline
from .image_png_pipeline import ImagePngPipeline
from .image_gif_pipeline import ImageGifPipeline
from .image_bmp_pipeline import ImageBmpPipeline
from .image_tiff_pipeline import ImageTiffPipeline
from .image_webp_pipeline import ImageWebpPipeline

# Phase 2: XML пайплайны
from .xml_xml_pipeline import XmlXmlPipeline
from .xml_xsd_pipeline import XmlXsdPipeline
from .xml_xsl_pipeline import XmlXslPipeline
from .xml_wsdl_pipeline import XmlWsdlPipeline
