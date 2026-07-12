-- Миграция 001: начальная схема исследовательской системы.
--
-- Создаёт в базе `file_organizer_db`:
--
-- 1. public.experiment_registry — каталог всех прогонов (заметки, параметры,
--    метрики, git commit, статусы).
-- 2. public.golden_state — текущее состояние золотого DAG (snapshot того,
--    какой вариант считается эталоном на момент коммита).
-- 3. public.format_taxonomy — справочник свойств подформатов: какому
--    подформату нужна саммаризация, какие topic'ы триггерят finance-gate.
-- 4. Схема gold_stage1 с таблицей documents — snapshot выхода Stage 1
--    (file_path, format_type, id), который клонируется в схемы экспериментов.
--
-- НЕ трогает существующие таблицы в public (documents, documents_<fmt>,
-- *_embeddings_<fmt>): они остаются как есть, используются действующим
-- (legacy) пайплайном, пока не будут демонтированы в одном из поздних блоков.
--
-- Применение:
--     python -m airflow.experiments.migrations.apply --snapshot

-- ===== public.experiment_registry =====
CREATE TABLE IF NOT EXISTS public.experiment_registry (
    run_id           TEXT PRIMARY KEY,
    dag_run_id       TEXT,
    parent_run_id   TEXT,
    started_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMP,
    status          TEXT NOT NULL DEFAULT 'running',  -- running|success|failed|rolled_back
    note            TEXT,
    params_jsonb    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics_jsonb   JSONB NOT NULL DEFAULT '{}'::jsonb,
    git_commit      TEXT,
    dag_name        TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_registry_status
    ON public.experiment_registry (status);
CREATE INDEX IF NOT EXISTS idx_exp_registry_started
    ON public.experiment_registry (started_at DESC);

-- ===== public.golden_state =====
-- Одно-строчная таблица (current). При коммите в golden_pipeline DAG
-- обновляется через UPSERT.
CREATE TABLE IF NOT EXISTS public.golden_state (
    id            SMALLINT PRIMARY KEY DEFAULT 1,
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    git_commit    TEXT,
    variants_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {stage.subformat: variant_name}
    CHECK (id = 1)
);
INSERT INTO public.golden_state (id) VALUES (1)
ON CONFLICT (id) DO NOTHING;

-- ===== public.format_taxonomy =====
-- Справочник свойств подформатов. Заполняется вручную через SQL
-- (или мини-UI позже). Airflow читает на build_stage2_plan.
CREATE TABLE IF NOT EXISTS public.format_taxonomy (
    format_type        TEXT PRIMARY KEY,
    family             TEXT NOT NULL,                  -- pdf|word|excel|image|xml|...
    summary_strategy   TEXT NOT NULL DEFAULT 'off',    -- off|optional|mandatory
    finance_topics     TEXT[] NOT NULL DEFAULT '{}',    -- topics, триггерящие final_class='finance'
    notes              TEXT
);
-- Начальная вставка для известных подформатов из текущего FORMAT_TARGETS.
-- Запускается один раз миграцией; дальнейшие правки — через SQL.
INSERT INTO public.format_taxonomy (format_type, family, summary_strategy, notes) VALUES
    ('pdf_text',        'pdf',   'off',      'Текстовые PDF, без саммаризации'),
    ('pdf_scan',        'pdf',   'off',      'Сканы, OCR'),
    ('pdf_tables',      'pdf',   'optional', 'PDF с таблицами, саммаризация опциональна'),
    ('pdf_tables_fin',   'pdf',   'mandatory','Финансовые PDF — саммаризация обязательна, finance-gate'),
    ('pdf_tables_tech',  'pdf',   'optional', 'Технические PDF'),
    ('pdf_tables_contr', 'pdf',   'optional', 'Договоры PDF'),
    ('pdf_tables_reports','pdf',  'optional', 'Отчёты PDF'),
    ('pdf_tables_other', 'pdf',   'optional', 'Прочие PDF с таблицами'),
    ('word_docx',  'word',  'off', 'Word docx'),
    ('word_doc',   'word',  'off', 'Word doc (legacy)'),
    ('word_rtf',   'word',  'off', 'RTF'),
    ('word_txt',   'word',  'off', 'Plain text'),
    ('word_odt',   'word',  'off', 'ODT'),
    ('excel_xlsx', 'excel', 'off', 'Excel xlsx'),
    ('excel_csv', 'excel',  'off', 'CSV'),
    ('excel_ods', 'excel',  'off', 'ODS'),
    ('image_jpg',  'image', 'off', 'JPG'),
    ('image_png',  'image', 'off', 'PNG'),
    ('image_gif',  'image', 'off', 'GIF'),
    ('image_bmp',  'image', 'off', 'BMP'),
    ('image_tiff', 'image', 'off', 'TIFF'),
    ('image_webp', 'image', 'off', 'WebP'),
    ('xml_xml',    'xml',   'off', 'XML'),
    ('xml_xsd',    'xml',   'off', 'XSD'),
    ('xml_xsl',    'xml',   'off', 'XSL'),
    ('xml_wsdl',   'xml',   'off', 'WSDL'),
    ('zip',        'archive','off', 'Архивы (вне Stage 2)'),
    ('multimedia', 'multimedia','off','Мультимедиа (вне Stage 2)')
ON CONFLICT (format_type) DO UPDATE SET
    family           = EXCLUDED.family,
    summary_strategy = EXCLUDED.summary_strategy,
    notes            = EXCLUDED.notes
WHERE public.format_taxonomy.family IS DISTINCT FROM EXCLUDED.family
   OR public.format_taxonomy.summary_strategy IS DISTINCT FROM EXCLUDED.summary_strategy;

-- Начальный список finance_topics. Расширяется вручную через SQL.
UPDATE public.format_taxonomy
SET finance_topics = ARRAY['финансовый отчёт','бухгалтерский баланс','отчёт о финансовых результатах','отчёт о движении денежных средств','прибыль','убыток','дебет','кредит']
WHERE format_type IN ('pdf_tables_fin');

-- ===== Схема gold_stage1 =====
-- Золотой выход Stage 1: список всех 182,724 файлов с их format_type.
-- Источник для всех экспериментов по умолчанию (если re_sort=false).
CREATE SCHEMA IF NOT EXISTS gold_stage1;

CREATE TABLE IF NOT EXISTS gold_stage1.documents (
    id           SERIAL PRIMARY KEY,
    file_path    TEXT UNIQUE NOT NULL,
    format_type  TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gold_stage1_format
    ON gold_stage1.documents (format_type);