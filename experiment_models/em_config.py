# em_config.py
# Конфигурация сравнительного эксперимента embedding-моделей.
#
# Цель: выбрать модель эмбеддингов для продакшена (русскоязычные деловые
# документы: XML, PDF текст, сканы через OCR). Baseline — текущая
# nomic-embed-text-v1.5 (англоязычная, посимвольная токенизация кириллицы).
#
# Модели (выбор пользователя 2026-07-24):
#   nomic_v15    — baseline, уже локальная (D:\MODELS\Transformers)
#   nomic_v2     — nomic-embed-text-v2-moe, многоязычная (~100 яз.), 768d
#   e5_large     — intfloat/multilingual-e5-large, 1024d, сильная на MIRACL
#   rubert_tiny2 — cointegrated/rubert-tiny2, 312d, маленькая/быстрая (RU)
#   sbert_ru     — ai-forever/sbert_large_nlu_ru, 1024d, русская
#
# Все модели: mean-pooling + L2-нормализация (единая методология).

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PowerShell 5.1: stdout в cp1251/cp866 — кириллица в print ломается
# (AGENTS.md gotcha 36).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import cfg  # noqa: E402  (импорт ПОСЛЕ sys.path fix)

BASE_DIR = Path(__file__).resolve().parent
EMB_DIR = BASE_DIR / 'embeddings'
LOG_DIR = BASE_DIR / 'logs'
REPORTS_DIR = BASE_DIR / 'reports'
SAMPLE_CSV = BASE_DIR / 'sample_50k.csv'

MODELS_BASE_DIR = Path(r"D:\MODELS\Transformers")

# ── Реестр моделей ───────────────────────────────────────────────
# prefix_doc — префикс для документов (как в продакшене пайплайна).
# max_len: 512 — ограничение e5/sbert; v1.5 оставляем 2048 (как в пайплайне).
MODELS = {
    'nomic_v15': {
        'hf_name': 'nomic-ai/nomic-embed-text-v1.5',
        'dim': 768,
        'prefix_doc': 'search_document: ',
        'trust_remote_code': True,
        'max_len': 2048,
        'batch': 64,
        'comment': 'baseline (текущая продакшен-модель), англоязычная',
    },
    'nomic_v2': {
        'hf_name': 'nomic-ai/nomic-embed-text-v2-moe',
        'dim': 768,
        'prefix_doc': 'search_document: ',
        'trust_remote_code': True,
        'max_len': 512,
        'batch': 64,
        'comment': 'многоязычная MoE (~100 яз.), drop-in замена v1.5',
    },
    'e5_large': {
        'hf_name': 'intfloat/multilingual-e5-large',
        'dim': 1024,
        'prefix_doc': 'passage: ',
        'trust_remote_code': False,
        'max_len': 512,
        'batch': 64,
        'comment': 'XLM-R large, сильная на MIRACL (рус.)',
    },
    'bge_m3': {
        'hf_name': 'BAAI/bge-m3',
        'dim': 1024,
        'prefix_doc': '',
        'trust_remote_code': False,
        'max_len': 512,
        'batch': 32,
        'comment': 'многоязычная BGE-M3, 1024d',
    },
    'rubert_tiny2': {
        'hf_name': 'cointegrated/rubert-tiny2',
        'dim': 312,
        'prefix_doc': '',
        'trust_remote_code': False,
        'max_len': 512,
        'batch': 256,
        'comment': 'русская, 29M параметров — скорость для продакшена',
    },
    'rubert_base_dp': {
        'hf_name': 'DeepPavlov/rubert-base-cased-sentence',
        'dim': 768,
        'prefix_doc': '',
        'trust_remote_code': False,
        'max_len': 512,
        'batch': 32,
        'comment': 'русская sentence-модель DeepPavlov на базе RuBERT',
    },
    'sbert_ru': {
        'hf_name': 'ai-forever/sbert_large_nlu_ru',
        'dim': 1024,
        'prefix_doc': '',
        'trust_remote_code': False,
        'max_len': 512,
        'batch': 64,
        'comment': 'русская sbert (nlu_ru), 427M параметров',
    },
    'qwen3e_4b': {
        'hf_name': 'Qwen/Qwen3-Embedding-4B',
        'dim': 2560,
        'prefix_doc': '',           # документы БЕЗ префикса (model card)
        'trust_remote_code': False,
        'max_len': 8192,
        'batch': 32,
        'pooling': 'last_token',    # НЕ mean! (model card: last_token_pool)
        'comment': 'Qwen3-Embedding-4B, MTEB Multi 69.45 (cluster 57.15), MRL 2560d',
    },
    'qwen3e_06b': {
        'hf_name': 'Qwen/Qwen3-Embedding-0.6B',
        'dim': 1024,
        'prefix_doc': '',
        'trust_remote_code': False,
        'max_len': 8192,
        'batch': 128,
        'pooling': 'last_token',
        'comment': 'Qwen3-Embedding-0.6B, MTEB Multi 64.33 (cluster 52.33), MRL 1024d',
    },
}

# Модели, которые нужно скачать (nomic_v15 уже локальная)
DOWNLOAD_MODELS = [k for k in MODELS if k != 'nomic_v15']


def model_local_path(key: str) -> Path:
    """Локальный путь модели в D:\\MODELS\\Transformers (соглашение проекта)."""
    return MODELS_BASE_DIR / MODELS[key]['hf_name'].replace('/', '_')


# ── PostgreSQL ───────────────────────────────────────────────────
DB_URL = (
    f"postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}"
    f"@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}"
)

# ── Документный тест: выборка из temp_xml_canonical ⋈ V3-assign ──
T_CANONICAL = 'temp_xml_canonical'
T_ASSIGN = 'temp_xml_subject_clean_assign'   # V3 (тематический очищенный)
MIN_SIMILARITY = 0.8      # продакшен-порог (0.85 дал только 25K — мало)
MIN_SUBJECT_LEN = 50      # отсев пустых subject_text
CAP_PER_THEME = 3000      # балансировка: макс. документов на тему

# ── OCR-тест ─────────────────────────────────────────────────────
OCR_N_DOCS = 200          # документов pdf_text для теста устойчивости
OCR_DPI = 200
