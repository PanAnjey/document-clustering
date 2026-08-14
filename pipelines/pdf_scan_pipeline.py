# pipelines/pdf_scan_pipeline.py
# Пайплайн для PDF-сканов: извлечение изображения → OSD → поворот → Tesseract OCR.

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality, filter_mupdf_stderr
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


@PipelineRegistry.register("pdf_scan")
class PdfScanPipeline(BasePipeline):
    """Пайплайн для PDF-сканов.
    
    Цепочка:
      1. Извлечь PNG первой страницы
      2. Tesseract OSD → определение угла поворота
      3. Поворот изображения, перезапись PNG
      4. Tesseract OCR
      5. Оценка качества текста — если poor/none → только image-эмбеддинг
    """
    
    def extract(self, file_path: Path) -> Dict:
        with filter_mupdf_stderr():
            return self._extract_inner(file_path)

    def _extract_inner(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }

        # 1. Извлечение изображения первой страницы
        image_path = self._generate_first_page_image(file_path)
        if not image_path:
            result['error'] = "Image extraction failed"
            return result
        result['image'] = str(image_path)

        # 2. Tesseract OSD — определение ориентации
        # self._fix_orientation(image_path)

        # 3. Tesseract OCR
        text = self._extract_text_tesseract(image_path)
        if text and text.strip():
            tq = assess_text_quality(text.strip())
            # Если качество неприемлемо — текст не сохраняем,
            # на этапе эмбеддингов будет только image-эмбеддинг
            if tq == "good":
                result['text'] = text.strip()
                result['text_quality'] = tq

        if not result['text'] and not result.get('image'):
            result['error'] = "Empty or extraction failed"

        return result

    # ---- Image generation ----

    def _generate_first_page_image(self, pdf_path: Path) -> Optional[str]:
        """Генерация PNG первой страницы PDF с коррекцией ошибочного /Rotate.

        Использует utils.rotate_fix.render_pdf_first_page_upright() —
        для сканов с /Rotate != 0 проверяет Tesseract OSD и при ошибочном
        /Rotate повторно рендерит без применения поворота.
        """
        try:
            from utils.rotate_fix import render_pdf_first_page_upright

            img_dir = cfg.EXTRA['pdf_images']
            img_dir.mkdir(parents=True, exist_ok=True)

            img_name = f"{pdf_path.stem}.png"
            img_path = img_dir / img_name

            cnt = 1
            while img_path.exists():
                img_path = img_dir / f"{pdf_path.stem}_{cnt}.png"
                cnt += 1

            result = render_pdf_first_page_upright(pdf_path, img_path, zoom=2.0)
            if result is None:
                logger.warning(f"Image generation failed for {pdf_path.name}")
            return result

        except Exception as e:
            logger.warning(f"Image generation failed for {pdf_path.name}: {e}")
            return None

    # ---- Orientation detection via Radon transform + deskew ----

    def _fix_orientation(self, image_path: str) -> None:
        """Коррекция ориентации через Radon transform.

        1. Radon sinogram → argmax(var) → определяет доминирующий угол (0..179)
        2. Если угол близок к 0 или 180 → нормально
        3. Если угол 80..100° → поворот на 90°
        4. Если угол 15..75° или 105..165° → мелкий skew
        5. Для отладки сохраняет оригинал и corrected_{angle}.png
        """
        try:
            import numpy as np
            from PIL import Image
            from skimage.transform import radon

            debug_dir = Path(image_path).parent / "_debug_orientation"
            debug_dir.mkdir(parents=True, exist_ok=True)
            orig_path = str(debug_dir / f"{Path(image_path).stem}_orig.png")

            img = Image.open(image_path).convert('RGB')
            img.save(orig_path)

            # ---- Radon transform для определения угла ----
            # Уменьшаем разрешение для скорости (~5x быстрее)
            small = img.resize((512, int(512 * img.height / img.width)), Image.LANCZOS)
            gray = np.array(small.convert('L'))
            gray = 255 - gray

            theta = np.arange(180)
            sinogram = radon(gray, theta=theta, circle=False)

            # Дисперсия проекции максимальна, когда проекция выровнена по тексту
            variances = np.var(sinogram, axis=0)
            best_theta = int(np.argmax(variances))

            # best_theta 0..179 — угол, под которым проекция даёт макс. дисперсию
            # Для текста это угол, перпендикулярный строкам.
            # Преобразуем в угол поворота изображения:
            #   best_theta = 90 → текстовые строки вертикальны → поворот на 0 (норма)
            #   best_theta = 0 или 180 → строки горизонтальны → поворот на 90°
            #   best_theta = 95 → небольшой перекос (5°)
            rot_candidates = {
                0:     {'action': 'rotate', 'angle': 90},   # строки вертикально → 90°
                90:    {'action': 'none',   'angle': 0},    # строки горизонтально → ок
                180:   {'action': 'rotate', 'angle': 90},   # строки вертикально → 90°
            }

            applied_rot = 0
            skew_angle = 0.0

            if best_theta in rot_candidates:
                action = rot_candidates[best_theta]
                if action['action'] == 'rotate':
                    img = img.rotate(-action['angle'], expand=True)
                    applied_rot = action['angle']
            else:
                # Мелкий skew: best_theta ∈ (0..90) — отклонение от горизонтали
                if best_theta < 90:
                    skew_angle = 90 - best_theta  # перекос в градусах
                else:
                    skew_angle = 90 - best_theta  # 180 - best_theta для >90

                if abs(skew_angle) >= 2 and abs(skew_angle) < 45:
                    img = img.rotate(-skew_angle, expand=True)

            # ---- Сохранение ----
            img.save(image_path)

            if applied_rot != 0 or abs(skew_angle) >= 2:
                tag = f"{applied_rot}deg" if applied_rot else f"skew{int(skew_angle)}"
                rot_name = f"{Path(image_path).stem}_{tag}.png"
                img.save(str(debug_dir / rot_name))
                logger.info(f"Radon: {Path(image_path).name} θ={best_theta} → {tag}")

        except Exception as e:
            logger.warning(f"Orientation fix failed for {Path(image_path).name}: {e}")

    # ---- Tesseract OCR ----

    def _extract_text_tesseract(self, image_path: str) -> Optional[str]:
        """Извлечение текста через Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image

            pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_PATH
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=cfg.TESSERACT_LANG)
            return text.strip() if text and text.strip() else None
        except Exception as e:
            logger.warning(f"Tesseract OCR failed for {Path(image_path).name}: {e}")
            return None
