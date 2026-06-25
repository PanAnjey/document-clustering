# pipelines/image_gif_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_gif")
class ImageGifPipeline(ImageBasePipeline):
    """Пайплайн для GIF изображений."""
    pass
