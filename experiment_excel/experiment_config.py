# experiment_config.py
import os
import sys
from pathlib import Path

# ── sys.path fix ─────────────────────────────────────────────────
# Скрипты лежат в подпапке experiment_excel\, а модули проекта (config,
# extractors, embeddings_engine) — в корне. Python кладёт в sys.path каталог
# скрипта, а не CWD, поэтому без этой вставки импорты падают с
# ModuleNotFoundError независимо от того, откуда запущен скрипт.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PowerShell 5.1 / pipe → stdout в cp1251/cp866: эмодзи (✅❌🗒️) в print
# вызывают UnicodeEncodeError и могут оборвать выполнение (AGENTS.md gotcha 36).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import cfg  # noqa: E402  (импорт ПОСЛЕ sys.path fix)

# ── Путь к исходным файлам ───────────────────────────────────────
SOURCE_ROOT = Path(r'D:\FileOrganizer\Sorted\Excel_Xlsx')
# Папка смешанная (проверено: 19 624 .xlsx + 12 404 .xls = 32 028 файлов).
# Оба расширения поддерживаются продакшен-путём (Aspose.Cells .NET).
SOURCE_EXTENSIONS = ('.xlsx', '.xls')

# ── База данных ───────────────────────────────────────────────
# Берём параметры из config.cfg (там PG_DB='file_organizer_db'),
# чтобы не расходиться с основным проектом. Env DATABASE_URL переопределяет.
DB_URL = os.getenv(
    'DATABASE_URL',
    f"postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}"
)

# ── Модели ─────────────────────────────────────────────────────
# Девайс НЕ дублируем: движок читает cfg.EMB_GPU_DEVICE ('cuda:1').
EMBED_BATCH_SIZE = cfg.EMB_BATCH_SIZE      # 32
EMBED_DIM = cfg.EMB_DIMENSION              # 768

# ── Экстракция текста ───────────────────────────────────────
MAX_EXTRACT_LEN = 3000                 # символов из начала документа (3000: предметные строки номенклатуры должны помещаться)
# Режим извлечения:
#   'direct' — прямой TSV-дамп через Aspose.Cells .NET (extract_text), без PDF.
#              Нет проблемы обрезания широких таблиц по краю страницы.
#   'pdf'    — legacy: Aspose → PDF (1-2 стр.) → текст через fitz.
EXTRACT_MODE = 'direct'
# Генерировать ли PDF-артефакты (для визуального аудита в pdf_viewer) даже
# в режиме 'direct'. Конвертер с FitToPagesWide=1 — широкие таблицы не режутся.
WRITE_PDF_ARTIFACT = True

# ── Кластеризация первого уровня (HDBSCAN) ───────────────────
SIMILARITY_THRESHOLD = 0.8
MIN_CLUSTER_SIZE_FACTOR = 0.01        # 1 % от количества тренировочных файлов (0.05 дало 2 кластера и 64.6% шума)

# ── Параметры второго уровня (KMeans) ────────────────────────
# Top2Vec удалён: пакет не установлен, а его API в исходном коде использовался
# неверно (model.document_ids — это не метки кластеров). Все типы, независимо
# от размера, кластеризуются KMeans по формуле из плана.
KMEANS_MAX_ITER = 300
KMEANS_MAX_K = 5
KMEANS_DOCS_PER_CLUSTER = 50          # k = max(2, min(MAX_K, N // DOCS_PER_CLUSTER))

# ── Имена скриптов (для справки) ───────────────────────────────
SCRIPT_PHASE_1 = 'phase_1_split_and_extract.py'
SCRIPT_PHASE_2 = 'phase_2_train_embeddings_hdbscan.py'
SCRIPT_PHASE_3 = 'phase_3_test_embeddings_assign.py'
SCRIPT_PHASE_4 = 'phase_4_second_level_clustering.py'
SCRIPT_PHASE_5 = 'phase_5_report_and_cleanup.py'
