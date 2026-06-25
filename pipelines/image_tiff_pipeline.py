# pipelines/image_tiff_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_tiff")
class ImageTiffPipeline(ImageBasePipeline):
    """Пайплайн для TIFF/TIF изображений."""
    pass
