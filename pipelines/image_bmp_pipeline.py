# pipelines/image_bmp_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_bmp")
class ImageBmpPipeline(ImageBasePipeline):
    """Пайплайн для BMP изображений."""
    pass
