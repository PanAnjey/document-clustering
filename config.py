# config.py
# Версия: 1.2
# Дата: 17.04.2026

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import os  # used for os.cpu_count()

@dataclass
class Config:
    # -------------------- Общие пути --------------------
    ROOT: Path = Path(r"D:\FileOrganizer")
    SOURCE_DIR: Path = field(init=False)
    ERRORS_DIR: Path = field(init=False)
    FAILED_EXTRACTION_DIR: Path = field(init=False)
    
    # -------------------- Поддерживаемые форматы --------------------
    PDF_FORMATS: List[str] = field(default_factory=lambda: [".pdf"])
    EXCEL_FORMATS: List[str] = field(
        default_factory=lambda: [".xlsx", ".xls", ".xlsm", ".xlsb", ".xltx", ".xlt", ".xltm", ".csv", ".ods"]
    )
    WORD_FORMATS: List[str] = field(
        default_factory=lambda: [".docx", ".doc", ".rtf", ".txt", ".odt"]
    )
    IMAGE_FORMATS: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg", ".jfif"]
    )
    XML_FORMATS: List[str] = field(
        default_factory=lambda: ['.xml', '.xsd', '.xsl', '.xslt', '.wsdl']
    )
    MODELS_BASE_DIR: Path = Path(r"D:\MODELS\Transformers")
    TEXT_EMB_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    IMAGE_EMB_MODEL_NAME: str = "nomic-ai/nomic-embed-vision-v1.5"
    TEXT_EMB_MODEL_PATH: Path = MODELS_BASE_DIR / TEXT_EMB_MODEL_NAME.replace("/", "_")
    IMAGE_EMB_MODEL_PATH: Path = MODELS_BASE_DIR / IMAGE_EMB_MODEL_NAME.replace("/", "_")
    EMB_DIMENSION: int = 768

    # -------------------- GPU устройства --------------------
    # cuda:0 = RTX PRO 4000 Blackwell (24 GB) — LLM (требует больше VRAM)
    # cuda:1 = GeForce RTX 5060 Ti (16 GB) — эмбеддинги (nomic-embed ~2 GB)
    EMB_GPU_DEVICE: str = "cuda:1"

    # -------------------- LLM саммаризация --------------------
    LLM_ENABLED: bool = True

    # Список доступных моделей LM Studio (для выбора в web-интерфейсе)
    LM_STUDIO_MODELS: List[str] = field(default_factory=lambda: [
        "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive@q6_k",
        "glm-4.7-flash@q4_k_s",
        "glm-4.7-flash@q8_k_xl",
        "qwen3.6-27b",
        "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",
        "mimo-v2.5-pro",
        "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive@q4_k_m",
        "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive@q5_k_m",
        "granite-4.1-30b",
        "gigachat3-10b-a1.8b",
        "gemma-4-31b@q5_k_s",
        "gemma-4-31b@q6_k",
        "yandexgpt-5-lite-8b-instruct",
        "zai-org/glm-4.7-flash",
        "text-embedding-nomic-embed-text-v1.5",
        "qwen3-vl-8b-instruct",
    ])

    # Параметры для бэкенда "transformers" (HF Qwen3.5-4B)
    HF_MODEL_PATH: Path = MODELS_BASE_DIR / "Qwen_Qwen3.5-4B"
    LLM_GPU_DEVICE: str = "cuda:0"
    HF_BATCH_SIZE: int = 16
    HF_MAX_TOKENS: int = 512

    # Общие параметры
    LLM_MIN_TEXT_LENGTH: int = 100
    LLM_MAX_TEXT_LENGTH: int = 7000
    LLM_MAX_CONTEXT_TOKENS: int = 80000

    # -------------------- PostgreSQL (pgvector) --------------------
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB:   str = "file_organizer_db"
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "postgres"   

    # -------------------- Целевые (Sorted) — по форматам --------------------
    FORMAT_TARGETS = {
        # PDF подкатегории
        "pdf_text":   "Sorted/PDF_Text",
        "pdf_scan":   "Sorted/PDF_Scan",
        # pdf_tables — общая папка (для unmatched); подтипы раскладываются в подпапки
        "pdf_tables":         "Sorted/PDF_Tables",
        "pdf_tables_fin":     "Sorted/PDF_Tables/fin",
        "pdf_tables_tech":    "Sorted/PDF_Tables/tech",
        "pdf_tables_contr":   "Sorted/PDF_Tables/contr",
        "pdf_tables_reports": "Sorted/PDF_Tables/reports",
        "pdf_tables_other":   "Sorted/PDF_Tables/other",
        # Word форматы (будут добавлены в Phase 2)
        "word_docx":  "Sorted/Word_Docx",
        "word_doc":   "Sorted/Word_Doc",
        "word_rtf":   "Sorted/Word_Rtf",
        "word_txt":   "Sorted/Word_Txt",
        "word_odt":   "Sorted/Word_Odt",
        # Excel форматы (будут добавлены в Phase 2)
        "excel_xlsx": "Sorted/Excel_Xlsx",
        "excel_csv":  "Sorted/Excel_Csv",
        "excel_ods":  "Sorted/Excel_Ods",
        # Image форматы (будут добавлены в Phase 2)
        "image_jpg":  "Sorted/Image_Jpg",
        "image_png":  "Sorted/Image_Png",
        "image_gif":  "Sorted/Image_Gif",
        "image_bmp":  "Sorted/Image_Bmp",
        "image_tiff": "Sorted/Image_Tiff",
        "image_webp": "Sorted/Image_Webp",
        # XML форматы
        "xml_xml":    "Sorted/xml",
        "xml_xsd":    "Sorted/xsd",
        "xml_xsl":    "Sorted/xsl",
        "xml_wsdl":   "Sorted/wsdl",
        "xml_fns":    "Sorted/xml_fns",
        # Архивы и мультимедиа
        "zip":        "Sorted/zip",
        "multimedia": "Sorted/multimedia",
    }

    # Legacy TAGS (обратная совместимость)
    TAGS = {
        "pdf":   "Sorted/PDF",
        "pdf_text": "Sorted/PDF_Text",
        "pdf_scan": "Sorted/PDF_Scan",
        "pdf_tables":         "Sorted/PDF_Tables",
        "pdf_tables_fin":     "Sorted/PDF_Tables/fin",
        "pdf_tables_tech":    "Sorted/PDF_Tables/tech",
        "pdf_tables_contr":   "Sorted/PDF_Tables/contr",
        "pdf_tables_reports": "Sorted/PDF_Tables/reports",
        "pdf_tables_other":   "Sorted/PDF_Tables/other",
        "excel": "Sorted/Excel",
        "word":  "Sorted/Word",
        "image": "Sorted/Images",
        "xml":   "Sorted/XML",
        "failed_extraction": "FailedExtraction",
    }

    # -------------------- LM Studio (Qwen3.6-35B для уточнения кластеров) --------------------
    LM_STUDIO_URL: str = "http://localhost:1234"
    QWEN_36B_MODEL: str = "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive@q4_k_m"
    CLUSTER_REFINEMENT_ENABLED: bool = True
    TARGETS: dict = field(init=False)

    # -------------------- Extra (извлечённые данные) --------------------
    EXTRACT_ROOT: Path = field(init=False)
    EMBEDDINGS_DIR: Path = field(init=False)
    LOG_DIR: Path = field(init=False)
    LOG_FILE: Path = field(init=False)
    
    # Папки для временных файлов
    EXTRA: dict = field(init=False)

    # -------------------- Processing --------------------
    MAX_CONCURRENT_FILES: int = max(1, (os.cpu_count() or 4) - 4)
    LIBREOFFICE_PATH: str = r"C:\Program Files\LibreOffice\program\soffice.exe"
    LO_MAX_WORKERS: int = 3

    # -------------------- Aspose.Words for Java --------------------
    ASPOSE_WORDS_JAVA_DIR: Path = Path(r"D:\Yandex.Disk\Aspose\Aspose.Words for Java")
    ASPOSE_WORKERS: int = 8

    # -------------------- Aspose.Cells for Java (XLSX → PDF) --------------------
    # Используется на этапе 2 для формата excel_xlsx: конвертация в PDF +
    # извлечение первых 2 страниц как PNG-изображений в cfg.EXTRA['excel_xlsx_pages'].
    # jar должен лежать в lib/Aspose.Cells_for_Java/lib/, см. README в этом каталоге.
    ASPOSE_CELLS_JAVA_DIR: Path = Path(r"D:\Yandex.Disk\Aspose\Aspose.Cells for Java")

    # -------------------- COM (Microsoft Office) --------------------
    COM_ENABLED: bool = True
    COM_WORD_WORKERS: int = 2
    COM_EXCEL_WORKERS: int = 2

    EMB_BATCH_SIZE: int = 32
    MAX_IMAGE_PIXELS: int = 178_956_970

    # -------------------- PyMuPDF4LLM (извлечение текста) --------------------
    PYMUPDF4LLM_ENABLED: bool = True
    TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_LANG: str = "rus+eng"
    PYMUPDF4LLM_OCR_ENABLED: bool = True
    PYMUPDF4LLM_HEADER_FOOTER: bool = False
    PYMUPDF4LLM_TABLE_STRATEGY: str = "lines_strict"

    # -------------------- Pandoc (извлечение текста) --------------------
    PANDOC_ENABLED: bool = True
    PANDOC_PATH: str = r"C:\Program Files\Pandoc\pandoc.exe"  # Оставить пустым для поиска через PATH
    PANDOC_TIMEOUT: int = 120  # Таймаут в секундах для конвертации через Pandoc
    PANDOC_PYTHON_FALLBACK: bool = True  # Использовать pandoc-библиотеку если CLI не найден

    # -------------------- Тестовый режим --------------------
    SOURCE_RAR: Path = ROOT / "SourceFiles.rar"
    SOURCE_RAR_TEST: Path = ROOT / "SourceFiles_test.rar"

    # -------------------- Full Rollback --------------------
    ROLLBACK_REPOPULATE_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\NONFORMAL\1")

    # -------------------- Clustering --------------------
    SIMILARITY_THRESHOLD: float = 0.85
    TOP_N_CLUSTERS: int = 20
    MIN_CLUSTER_SIZE: int = 95  # Минимальный размер кластера (меньшие объединяются в "Неклассифицированные")
    DIST_MATRIX_CHUNK: int = 5000

    # -------------------- Логи --------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self):
        # Производные пути (пересчитываются при изменении ROOT через override)
        self.SOURCE_DIR = self.ROOT / "SourceFiles"
        self.ERRORS_DIR = self.ROOT / "ErrorFiles"
        self.FAILED_EXTRACTION_DIR = self.ROOT / "FailedExtraction"
        self.EXTRACT_ROOT = self.ROOT / "Extracted"
        self.EMBEDDINGS_DIR = self.ROOT / "Embeddings"
        self.LOG_DIR = self.ROOT / "Logs"
        self.LOG_FILE = self.LOG_DIR / "file_organizer.log"

        # Инициализация TARGETS
        self.TARGETS = {k: self.ROOT / v for k, v in self.TAGS.items()}
        
        # Инициализация EXTRA (пути для временных файлов)
        self.EXTRA = {
            "pdf_images":  self.EXTRACT_ROOT / "PDF_Images",
            "office_pdf":  self.EXTRACT_ROOT / "Office_PDF",
            # Excel XLSX: результат этапа 2 — конвертированный PDF и первые 1-2 страницы как PNG
            "excel_xlsx_pages": self.EXTRACT_ROOT / "Excel_Xlsx",
        }
        
        self.ensure_dirs()

    def ensure_dirs(self):
        """Создание необходимых директорий. Можно вызывать повторно после изменения путей.

        Sorted/ подкатегории НЕ создаются здесь — они создаются по требованию
        индивидуальными процессорами при перемещении файлов.
        """
        for path in [self.ERRORS_DIR, self.LOG_DIR, self.EMBEDDINGS_DIR,
                      self.FAILED_EXTRACTION_DIR]:
            if path:
                path.mkdir(parents=True, exist_ok=True)
        for path in self.TARGETS.values():
            if path and "Sorted" not in path.parts:
                path.mkdir(parents=True, exist_ok=True)
        for path in self.EXTRA.values():
            if path:
                path.mkdir(parents=True, exist_ok=True)

cfg = Config()

_override_path = Path(__file__).parent / "config_override.json"
if _override_path.exists():
    try:
        import json as _json
        _overrides = _json.loads(_override_path.read_text(encoding="utf-8"))
        for _key, _val in _overrides.items():
            if hasattr(cfg, _key):
                # Получаем тип поля из dataclass, а не текущее значение
                field_type = None
                for _field in Config.__dataclass_fields__.values():
                    if _field.name == _key:
                        field_type = _field.type
                        break
                
                if field_type == 'Path' or field_type == Path:
                    setattr(cfg, _key, Path(_val))
                elif field_type == 'bool' or field_type == bool:
                    setattr(cfg, _key, bool(_val))
                elif field_type == 'int' or field_type == int:
                    setattr(cfg, _key, int(_val))
                elif field_type == 'float' or field_type == float:
                    setattr(cfg, _key, float(_val))
                else:
                    setattr(cfg, _key, _val)
    except Exception as e:
        import logging as _logging
        _logging.getLogger("FileOrganizer").warning(f"Failed to load config_override.json: {e}")
