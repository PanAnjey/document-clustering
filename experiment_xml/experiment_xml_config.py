# experiment_xml_config.py
# Конфигурация эксперимента по классификации формализованных XML-документов
# (UniversalTransferDocument, Invoice из MySQL events_26.xml).
#
# Методология повторяет experiment_excel (эксперимент по неформализованным
# документам): nomic-embed-text-v1.5 (768d, L2-norm) → cosine к 26 центроидам
# temp_theme2_centroids_v2 → SIMILARITY_THRESHOLD=0.8, is_unknown при sim<0.8.

import os
import sys
from pathlib import Path

# ── sys.path fix (как в experiment_excel) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PowerShell 5.1 / pipe → stdout в cp1251/cp866: эмодзи и кириллица
# в print могут вызвать UnicodeEncodeError (AGENTS.md gotcha 36).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import cfg  # noqa: E402  (импорт ПОСЛЕ sys.path fix)

# ── MySQL (источник XML) ─────────────────────────────────────────
# Параметры совпадают с mysql-rw MCP (opencode.json).
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3308'))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'mysql')
MYSQL_DB = 'events_26'
MYSQL_TABLE = 'xml'

# Типы документов для обработки. Все — формат ON_NSCHFDOPPR (КНД 1115131)
# или корректировочный ON_NKORSCHFDOPPR (КНД 1115133) — один парсер.
#   Исходные: UTD + Invoice (223,849)
#   Расширение (п.2 правок): акты приемки/возврата/выполненных работ (+171,604)
#   Корректировочные: УКД + КСФ (+6,784)
# Не берутся (иные форматы, мелкие объёмы): ActOfsetting (OAKTNAVZ),
# ProformaInvoice (ON_SCHET), ReconciliationAct (ON_AKTSVEROTP).
DOC_TYPES = (
    'UniversalTransferDocument',           # 167,247
    'Invoice',                             # 56,604
    'ReturnInventoryAcceptanceCertificate',      # 55,021
    'StorageInventoryAcceptanceCertificate',     # 40,029
    'PerformedWorkAcceptanceCertificate',        # 29,796
    'PerformedWorkCostCertificate',              # 29,740
    'XmlAcceptanceCertificate',                  # 17,018
    'UniversalCorrectionDocument',               # 4,315
    'InvoiceCorrection',                         # 2,469
)

# Признак маркированных товаров (ГИС МТ) в имени файла:
# ИдФайл = R_T_A_O_GGGGMMDD_N1_N2_N3_N4_N5_N6_N7, N3 == '1' → УПД с маркировкой.
# ВНИМАНИЕ: префикс ON_NSCHFDOPPR сам содержит '_', поэтому при split('_')
# позиции смещены на +1: N1(UUID)=parts[5], N2=parts[6], N3=parts[7].
# Проверено на данных: parts[7]=='1' → 1,208 файлов, все UTD (как и должно быть).
# Такие документы выделяются в отдельный кластер (assigned_theme = MARKED_THEME).
MARKED_N3_INDEX = 7          # позиция N3 при split('_') имени файла (0-based)
MARKED_THEME = 90            # спец. метка кластера «ГИС МТ (маркированные товары)»

# ── PostgreSQL (результаты) ──────────────────────────────────────
DB_URL = os.getenv(
    'DATABASE_URL',
    f"postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}"
)

# Таблицы эксперимента (префикс temp_xml_* — по аналогии с temp_pdftext_*)
T_CANONICAL = 'temp_xml_canonical'    # канонизированные документы (V1 + V2)
T_EMBEDDINGS = 'temp_xml_embeddings'  # эмбеддинги bytea (float32, 768): оба варианта
T_ASSIGN = 'temp_xml_assign'          # присвоение тем по V1 (полный канон)
T_SUBJECT_ASSIGN = 'temp_xml_subject_assign'  # присвоение тем по V2 (тематический)

# ── Канонизация ──────────────────────────────────────────────────
MAX_TABLE_ROWS = 5          # первых строк табличной части (СведТов)
MAX_INFO_BLOCK_CHARS = 500  # лимит символов на блок ИнфПолФХЖ* (каждый)
MAX_OSNLEN = 300            # лимит символов на блок ОснПер

# ── Эмбеддинги / присвоение тем ──────────────────────────────────
EMBED_BATCH_SIZE = cfg.EMB_BATCH_SIZE   # 32
EMBED_DIM = cfg.EMB_DIMENSION           # 768
SIMILARITY_THRESHOLD = 0.8              # как в experiment_excel
CENTROIDS_TABLE = 'temp_theme2_centroids_v2'  # 26 тем предыдущего эксперимента
