# pipelines/image_webp_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_webp")
class ImageWebpPipeline(ImageBasePipeline):
    """Пайплайн для WEBP/SVG/JFIF изображений."""
    pass
