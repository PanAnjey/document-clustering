# pipelines/image_jpg_pipeline.py
from .image_base_pipeline import ImageBasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("image_jpg")
class ImageJpgPipeline(ImageBasePipeline):
    """Пайплайн для JPG/JPEG изображений."""
    pass
