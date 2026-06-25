# pipelines/image_png_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_png")
class ImagePngPipeline(ImageBasePipeline):
    """Пайплайн для PNG изображений."""
    pass
