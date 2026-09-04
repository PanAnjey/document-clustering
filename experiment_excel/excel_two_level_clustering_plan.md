**План эксперимента — «Двухуровневая кластеризация Excel-файлов»**

---

## 0. Общие настройки (файл `experiment_config.py`)

```python
# experiment_config.py
import os, sys
from pathlib import Path

# ── sys.path fix: модули проекта (config, extractors, embeddings_engine)
# лежат в корне, скрипты — в подпапке experiment_excel\
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import cfg

# ── Путь к исходным файлам ───────────────────────────────────────
SOURCE_ROOT = Path(r'D:\FileOrganizer\Sorted\Excel_Xlsx')
SOURCE_EXTENSIONS = ('.xlsx', '.xls')   # папка смешанная: 19.6k xlsx + 12.4k xls

# ── База данных (из config.cfg → file_organizer_db; env DATABASE_URL переопределяет)
DB_URL = os.getenv('DATABASE_URL',
                   f"postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}")

# ── Модели ─────────────────────────────────────────────────────
EMBED_BATCH_SIZE = cfg.EMB_BATCH_SIZE   # 32; девайс — cfg.EMB_GPU_DEVICE ('cuda:1')
EMBED_DIM = cfg.EMB_DIMENSION           # 768

# ── Экстракция текста ───────────────────────────────────────
MAX_EXTRACT_LEN = 1000                 # символов из начала документа

# ── Кластеризация первого уровня (HDBSCAN) ───────────────────
SIMILARITY_THRESHOLD = 0.8
MIN_CLUSTER_SIZE_FACTOR = 0.05        # 5 % от количества тренировочных файлов

# ── Параметры второго уровня (KMeans) ────────────────────────
KMEANS_MAX_ITER = 300
KMEANS_MAX_K = 5
KMEANS_DOCS_PER_CLUSTER = 50          # k = max(2, min(MAX_K, N // DOCS_PER_CLUSTER))
```

**Вспомогательный модуль `embed_helper.py`** — батчевый эмбеддинг текстов:
грузит **только текстовую** модель nomic-embed-text-v1.5 (image-модель ~2 ГБ
VRAM не нужна), не пишет `.npy` на диск. Возвращает L2-нормализованные
`float32 (N, 768)`.

---

## 1️⃣ Phase 1 — Разделение, извлечение, запись во временные таблицы

| Шаг | Действие | SQL-таблицы | Ошибки |
|-----|----------|------------|--------|
| 1.1 | Список всех файлов `SOURCE_EXTENSIONS` (`.xlsx` + `.xls`) в `SOURCE_ROOT`. | – | Если файлов нет – abort. |
| 1.2 | Перемешивание (seed = 42) и разделение пополам. | – | – |
| 1.3 | Извлечение **продакшен-путём** (`extractors/cells_aspose_extractor.py`, как `pipelines/excel_xlsx_pipeline.py`): Aspose.Cells → PDF → первые 1-2 стр. (`truncate_pdf_to_first_pages`) → текст через fitz. Fallback: COM Excel. Берутся первые `MAX_EXTRACT_LEN` символов. | `temp_excel_train`, `temp_excel_test` | Любая неудача (исключение **или** пустой текст/`error` без исключения) → `temp_excel_failed`. |
| 1.4 | Bulk-INSERT (по 500 записей), прогресс каждые 200 файлов. | – | – |

PDF-артефакты сохраняются в `cfg.EXTRA['excel_xlsx_pages']` — тот же каталог,
что и у этапа 2 продакшена (идемпотентно: готовые PDF переиспользуются).

**Таблицы**

```sql
CREATE TABLE temp_excel_train (id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, txt TEXT NOT NULL);
CREATE TABLE temp_excel_test  (id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, txt TEXT NOT NULL);
CREATE TABLE temp_excel_failed(id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, reason TEXT NOT NULL);
```

---

## 2️⃣ Phase 2 — Эмбеддинг тренировочного набора + HDBSCAN (формы)

| Шаг | Действие | Таблицы | Возможные проблемы |
|-----|----------|---------|--------------------|
| 2.1 | Считать `id, txt` из `temp_excel_train`. | – | – |
| 2.2 | Эмбеддинг через `embed_helper.encode_texts` (только text-модель). | – | OOM → уменьшить `EMBED_BATCH_SIZE`. |
| 2.3 | HDBSCAN: `metric='euclidean'` (эмбеддинги L2-нормализованы → euclid² = 2·(1−cos); `metric='cosine'` не поддерживается hdbscan 0.8.44 + sklearn 1.9), `min_cluster_size = max(2, int(MIN_CLUSTER_SIZE_FACTOR * N_train))`. | – | Если только шум (`-1`) → уменьшить `MIN_CLUSTER_SIZE_FACTOR`. |
| 2.4 | Сохранить метки: `temp_hdbscan(train_id, label)`. | `temp_hdbscan` | – |
| 2.5 | Сохранить эмбеддинги train-набора: `temp_train_embeddings(train_id, embedding)` — Phase 4 переиспользует их без GPU. | `temp_train_embeddings` | – |
| 2.6 | Вычислить центроиды для реальных кластеров (без `-1`) → `temp_hdbscan_centroids(label, embedding)`. | `temp_hdbscan_centroids` | – |
| 2.7 | Silhouette-score с сэмплированием (`sample_size=5000`) — без O(N²) на полном наборе. | – | – |

**Таблицы**

```sql
CREATE TABLE temp_hdbscan (train_id INTEGER PRIMARY KEY, label INTEGER NOT NULL);
CREATE TABLE temp_hdbscan_centroids (label INTEGER PRIMARY KEY, embedding BYTEA NOT NULL);
CREATE TABLE temp_train_embeddings (train_id INTEGER PRIMARY KEY, embedding BYTEA NOT NULL);
```

---

## 3️⃣ Phase 3 — Эмбеддинг тест-набора + назначение ближайшего типа

| Шаг | Действие | Таблицы | Ошибки |
|-----|----------|---------|--------|
| 3.1 | Считать `id, txt` из `temp_excel_test`. | – | – |
| 3.2 | Эмбеддинг через `embed_helper.encode_texts`. | – | OOM → уменьшить `EMBED_BATCH_SIZE`. |
| 3.3 | Считать все центроиды из `temp_hdbscan_centroids`. | – | Если пусто → abort (Phase 2 не успел). |
| 3.4 | Косинус-схожесть с центроидами, выбрать максимум. | `temp_test_assign(test_id, assigned_label, similarity)` | – |
| 3.5 | Покрытие: `coverage = (similarity ≥ SIMILARITY_THRESHOLD).mean()` + avg/min/max similarity. | – | Низкое среднее сходство → пересмотреть параметры Phase 2. |

**Таблица**

```sql
CREATE TABLE temp_test_assign (
    test_id INTEGER PRIMARY KEY,
    assigned_label INTEGER NOT NULL,
    similarity REAL NOT NULL
);
```

---

## 4️⃣ Phase 4 — Внутренняя (тематическая) кластеризация внутри каждого типа

**Метод — KMeans для всех типов** (Top2Vec удалён: пакет не установлен, а его
API в исходной версии использовался неверно — `model.document_ids` не является
метками кластеров, а `document_vectors` выровнены по внутреннему порядку
Top2Vec, а не по порядку входных документов):

```
k = max(2, min(KMEANS_MAX_K, N // KMEANS_DOCS_PER_CLUSTER)),  k ≤ N
```

### Таблица результатов

```sql
CREATE TABLE temp_second_level (
    train_id   INTEGER NOT NULL,
    type_label INTEGER NOT NULL,
    sub_label  INTEGER NOT NULL,
    embedding  BYTEA NOT NULL
);
```

### Пошаговый алгоритм
1. Список уникальных `label` из `temp_hdbscan` **`WHERE label != -1`** — шум
   исключается (это не тип, а гетерогенные отказники; Phase 2 его тоже не
   включает в центроиды).
2. Для каждого `label` взять эмбеддинги **из `temp_train_embeddings`** (JOIN с
   `temp_hdbscan`) — GPU на этой фазе не нужен.
3. KMeans → `sub_label`; запись в `temp_second_level`.

---

## 5️⃣ Phase 5 — Отчёт и очистка

| Шаг | Действие |
|-----|----------|
| 5.1 | **Метрики**: train/test/failed counts, количество типов (без `-1`), число шумовых документов, покрытие тест-набора при **пороге из конфига**, avg/min/max similarity, распределение под-кластеров по типам. Устойчива к пропущенным фазам (проверка `to_regclass`). |
| 5.2 | **Экспорт** в JSON: `D:\FileOrganizer\Reports\excel_experiment_<timestamp>.json` (UTF-8). |
| 5.3 | **Удалить** все временные таблицы: `temp_excel_*`, `temp_hdbscan`, `temp_hdbscan_centroids`, `temp_train_embeddings`, `temp_test_assign`, `temp_second_level`. |

---

## 6️⃣ Как запускать (по-очереди)

```bash
python phase_1_split_and_extract.py
python phase_2_train_embeddings_hdbscan.py
python phase_3_test_embeddings_assign.py
python phase_4_second_level_clustering.py
python phase_5_report_and_cleanup.py
```

Работает из любой директории (sys.path чинится в `experiment_config.py`).

* При ошибке **OOM** – уменьшить `EMBED_BATCH_SIZE` в `config_override.json` и перезапустить проблемный скрипт.
* При **отсутствии кластеров** в HDBSCAN – уменьшить `MIN_CLUSTER_SIZE_FACTOR` и запустить Phase 2 снова.
* Каждая фаза пересоздаёт свои таблицы (`DROP TABLE IF EXISTS`) — перезапуск безопасен.

---

## 7️⃣ Выводы для пользователя

* После **Phase 5** в `D:\FileOrganizer\Reports` появится `excel_experiment_YYYYMMDD_HHMMSS.json` с метриками.
* База данных чистая (все временные таблицы удалены).
* PDF-артефакты в `D:\FileOrganizer\Extracted\Excel_Xlsx` остаются и переиспользуются этапом 2 продакшен-пайплайна.
