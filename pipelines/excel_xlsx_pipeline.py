# pipelines/excel_xlsx_pipeline.py
# Пайплайн для .xlsx файлов:
#   1) Aspose.Cells for Java: .xlsx → PDF (сохраняется в cfg.EXTRA['excel_xlsx_pages'])
#   2) fitz обрезает PDF на месте до первых 1-2 страниц (если листов > max_pages)
#      — итоговый PDF содержит:
#          • 1 страницу, если исходник = 1 лист
#          • 2 страницы, если исходник = 2 листа
#          • 2 первые страницы, если исходник >= 3 листов
# Текст берётся из итогового PDF через fitz. Если Aspose.Cells недоступен —
# fallback на Pandoc CLI + COM Excel (без PDF артефактов).

from pathlib import Path
from typing import Dict, Optional

from config import cfg
from logger_utils import logger, assess_text_quality
from .base_pipeline import BasePipeline
from .pipeline_registry import PipelineRegistry


# Сколько максимум страниц оставлять в результирующем PDF.
# 1 страница → 1, 2 → 2, 3 и более → 2 первые.
MAX_PDF_PAGES = 2


@PipelineRegistry.register("excel_xlsx")
class ExcelXlsxPipeline(BasePipeline):
    """Пайплайн для .xlsx файлов: Aspose.Cells → PDF (обрезанный до max 2 страниц)."""

    def extract(self, file_path: Path) -> Dict:
        result = {
            "source": str(file_path),
            "type": self.format_type,
            "text": None,
            "image": None,
            "error": None,
            "text_quality": "none"
        }

        # ---- Основной путь: Aspose.Cells for Java ----
        try:
            from extractors.cells_aspose_extractor import (
                convert_xlsx_to_pdf,
                truncate_pdf_to_first_pages,
                extract_text_from_pdf,
            )

            out_dir = cfg.EXTRA['excel_xlsx_pages']
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = file_path.stem
            pdf_path = out_dir / f"{stem}.pdf"

            # Не переконвертируем если PDF уже есть (идемпотентность при перезапусках)
            if not pdf_path.exists():
                pdf_path = convert_xlsx_to_pdf(file_path, pdf_path)  # может вернуть None при ошибке

            if pdf_path and pdf_path.exists():
                # Шаг 2: обрезка PDF до первых 1-2 страниц (на месте)
                # (если в исходнике <= 2 листов — PDF остаётся без изменений)
                truncate_pdf_to_first_pages(pdf_path, max_pages=MAX_PDF_PAGES)

                # Текст из итогового (обрезанного) PDF
                text = extract_text_from_pdf(pdf_path, max_pages=0)
                if text:
                    result['text'] = text
                    result['text_quality'] = assess_text_quality(result['text'])

                if result['text']:
                    logger.debug(f"Excel XLSX (Aspose path) OK: {file_path.name}")
                    return result

                # Aspose.Cells создал PDF, но текст из него не извлечён
                # (image-only / 0 chars). Удаляем артефакт-сироту: иначе при
                # провале fallback источник уедет в ErrorFiles, а PDF
                # останется в Extracted/Excel_Xlsx как «лишний» файл.
                try:
                    pdf_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove orphan PDF {pdf_path.name}: {e}")

            # Если Aspose.Cells недоступен/упал — fallback ниже
            logger.info(f"Aspose.Cells unavailable for {file_path.name}, falling back to Pandoc+COM")
        except Exception as e:
            logger.warning(f"Aspose.Cells path failed for {file_path.name}: {e}, falling back to Pandoc+COM")

        # ---- Fallback: старый путь Pandoc + COM (без PDF артефактов) ----
        # Pandoc CLI
        if cfg.PANDOC_ENABLED:
            text = self._extract_pandoc(file_path)
            if text and text.strip():
                result['text'] = text.strip()
                result['text_quality'] = assess_text_quality(result['text'])
                return result

        # COM Excel
        try:
            from extractors.excel_extractor import extract_excel
            res = extract_excel(file_path)
            if res and res.get('text'):
                result['text'] = res['text']
                result['image'] = res.get('image')
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        except Exception as e:
            logger.warning(f"Excel COM extraction failed for {file_path.name}: {e}")

        result['error'] = "Aspose.Cells unavailable + Pandoc+COM extraction failed"
        return result

    def _extract_pandoc(self, file_path: Path) -> Optional[str]:
        """Извлечение через Pandoc CLI (fallback-путь)."""
        try:
            from extractors.pandoc_extractor import extract_with_pandoc
            res = extract_with_pandoc(file_path, "xlsx")
            if res and res.get('text'):
                return res['text']
        except Exception as e:
            logger.warning(f"Pandoc failed for {file_path.name}: {e}")
        return None