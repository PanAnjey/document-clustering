# Сводный benchmark embedding-моделей

Дата: 2026-08-18

## Набор

- Файл: `experiment_models/mixed_sample_2000_noscan.csv`
- Документов: 1 999 (один повреждённый Excel-файл исключён)
- Сканов и изображений: 0
- Ограничение входа: 512 токенов для всех моделей
- Средний текст: 3 107 символов / 394 слова

| Группа | Документов |
|---|---:|
| PDF с текстом | 400 |
| PDF с таблицами | 400 |
| Word | 399 |
| Excel | 400 |
| XML | 400 |

## Скорость

| Модель | Размерность | CPU, док./с | GPU, док./с | Ускорение GPU |
|---|---:|---:|---:|---:|
| `rubert_tiny2` | 312 | 145.5 | 619.5 | 4.3x |
| `rubert_base_dp` | 768 | 12.4 | 387.0 | 31.2x |
| `nomic_v15` | 768 | 7.7 | 254.7 | 33.1x |
| `nomic_v2` | 768 | 7.5 | 158.2 | 21.1x |
| `e5_large` | 1024 | 3.5 | 132.3 | 37.8x |
| `bge_m3` | 1024 | 3.6 | 131.5 | 36.5x |
| `sbert_ru` | 1024 | 3.5 | 147.8 | 42.2x |
| `Qwen3-Embedding-0.6B` | 1024 | 2.1 | 26.0 | 12.4x |
| `Qwen3-Embedding-4B` | 2560 | 0.4 | 11.5 | 28.8x |

CPU Qwen 4B прогнан с batch size 16; остальные модели использовали штатный batch size.
Все прогоны завершились без нулевых векторов после повторного запуска Qwen 4B.

Эмбеддинги сохранены в `experiment_models/embeddings/` с суффиксами
`mixed_cpu512`, `mixed_cpu512_b16` и `mixed_gpu512`.

## Библиотеки

| Модель | Базовые библиотеки | Особенности |
|---|---|---|
| `nomic_v15` | `torch`, `transformers`, `tokenizers` | `trust_remote_code=True`, собственный код Nomic |
| `nomic_v2` | `torch`, `transformers`, `tokenizers` | `trust_remote_code=True`; `megablocks` желательно для скорости, но не обязательно |
| `e5_large` | `torch`, `transformers`, `tokenizers` | Стандартный `AutoModel`; префикс `passage:` |
| `bge_m3` | `torch`, `transformers`, `tokenizers` | Стандартный `AutoModel`, mean-pooling |
| `rubert_tiny2` | `torch`, `transformers`, `tokenizers` | Стандартный `AutoModel`, mean-pooling |
| `rubert_base_dp` | `torch`, `transformers`, `tokenizers` | Стандартный `AutoModel`, mean-pooling |
| `sbert_ru` | `torch`, `transformers`, `tokenizers` | Стандартный `AutoModel`, mean-pooling |
| `Qwen3-Embedding-0.6B` | `torch`, `transformers`, `tokenizers` | Last-token pooling; размерность 1024 |
| `Qwen3-Embedding-4B` | `torch`, `transformers`, `tokenizers` | Last-token pooling; размерность 2560 |

Общие дополнительные компоненты:

- `safetensors` — загрузка весов большинства моделей;
- `huggingface_hub` — скачивание моделей, не нужен при полностью локальной установке;
- `numpy` — преобразование и сохранение векторов;
- `pandas` — только подготовка выборки и benchmark;
- CUDA-драйвер и CUDA-сборка PyTorch — только для GPU;
- `sentence-transformers` — **не требуется** в текущей реализации.

## Размер моделей

Размер на диске измерен по локальным весам в `D:\MODELS\Transformers`.
Количество параметров указано приблизительно.

| Модель | Параметров | Веса на диске | Оценка RAM/VRAM для инференса |
|---|---:|---:|---:|
| `rubert_tiny2` | 29M | 0.11 GB | минимальная |
| `rubert_base_dp` | 178M | 1.33 GB | низкая |
| `nomic_v15` | 137M | 0.51 GB | низкая |
| `nomic_v2` | 475M | 1.79 GB | средняя |
| `e5_large` | 560M | 2.11 GB | средняя |
| `bge_m3` | 568M | 4.27 GB | средняя |
| `sbert_ru` | 427M | 1.59 GB | средняя |
| `Qwen3-Embedding-0.6B` | 0.6B | 1.12 GB | высокая для CPU |
| `Qwen3-Embedding-4B` | 4B | 7.51 GB | высокая |

## Языковой smoke-тест

Дополнительный M1-тест для `bge_m3` (14 русско-английских пар, 5 различителей,
32 уникальных термина): средняя близость
синонимов `0.833`, Top-1 `78.6%`, различители `3/5`. Как и остальные модели,
BGE-M3 не различил пару `proforma invoice` / `invoice` с нужной точностью.

### Синонимы EN-RU

| English | Русский |
|---|---|
| proforma invoice | счет на оплату |
| invoice | счет-фактура |
| certificate of completed work | акт выполненных работ |
| acceptance certificate | акт приемки |
| consignment note | товарная накладная |
| universal transfer document | универсальный передаточный документ |
| lease agreement | договор аренды |
| agency commission | агентское вознаграждение |
| telecommunication services | услуги связи |
| equipment supply | поставка оборудования |
| flight tickets | авиабилеты |
| business travel expenses | командировочные расходы |
| electricity reimbursement | возмещение электроэнергии |
| base station modernization | модернизация базовой станции |

### Предметные различители

| Термин | Правильный вариант | Неправильный вариант |
|---|---|---|
| proforma invoice | счет на оплату | счет-фактура |
| invoice | счет-фактура | счет на оплату |
| reconciliation act | акт сверки | акт выполненных работ |
| acceptance certificate | акт приемки | акт сверки |
| storage certificate | складской акт | акт выполненных работ |

### Как читать показатели

- **Cosine similarity (косинусная близость)** — мера близости двух векторов: от -1 до 1, где 1 означает одинаковое направление. Для эмбеддингов документов обычно сравниваются значения от 0 до 1. Чем выше значение между английским и русским термином, тем лучше модель считает их семантически связанными.
- **Средняя близость синонимов** — среднее значение cosine similarity по всем 14 парам EN↔RU. Например, `0.892` у `e5_large` означает, что в среднем английские термины хорошо совпадали с соответствующими русскими вариантами. Это не процент правильных ответов.
- **Top-1** — доля случаев, когда правильный русский перевод оказался самым близким среди всех 16 русских кандидатов. В наборе 14 пар поэтому один правильный ответ составляет примерно 7,1 процентного пункта. `86%` означает 12 правильных попаданий из 14.
- **Предметные различители** — пять проверок, где модель должна выбрать правильный вариант из двух похожих. Например, `proforma invoice` должен быть ближе к `счет на оплату`, чем к `счет-фактура`. Результат `4/5` означает четыре правильных сравнения.
- **Почему метрики могут расходиться** — высокая средняя близость не гарантирует высокий Top-1: модель может считать правильный перевод близким, но поставить рядом ещё более похожий неправильный термин. Поэтому смотрим на показатели вместе.
- **Ограничение теста** — M1 проверяет небольшой набор терминов и кросс-языковую лексику. Он не заменяет оценку кластеризации на реальных документах.

## Символьные n-граммы

Дополнительно протестированы два варианта без нейросети на том же смешанном наборе:

- **raw char n-grams** — `HashingVectorizer(analyzer='char_wb', ngram_range=(3,5), n_features=4096, norm='l2')`;
- **TF-IDF char n-grams + SVD** — `TfidfVectorizer(char_wb 3-5, min_df=2, max_features=120000, sublinear_tf=True)` + `TruncatedSVD(512)` + L2-нормализация.

### Скорость построения векторов

| Представление | Размерность | Время | Док./с | RAM-оценка векторов |
|---|---:|---:|---:|---:|
| raw char n-grams | 4096 | 2.1 с | 946 | 31.2 MB |
| TF-IDF char + SVD | 512 | 32.6 с | 61.4 | 3.9 MB |
| `rubert_tiny2` CPU | 312 | 13.7 с | 145.5 | 2.4 MB |
| `rubert_base_dp` CPU | 768 | 160.6 с | 12.4 | 5.9 MB |

### HDBSCAN на смешанном наборе

Параметры для всех представлений: `min_cluster_size=95`, `cluster_selection_epsilon=sqrt(0.6)`, euclidean на нормализованных векторах. Для этого набора нет истинных тематических меток, поэтому NMI/ARI считаются против **группы формата** (`pdf_text`, `pdf_tables`, `word`, `excel`, `xml`), а не против тем документов.

| Представление | Dim | Кластеров | Шум | NMI к формату | ARI к формату |
|---|---:|---:|---:|---:|---:|
| raw char n-grams | 4096 | 2 | 15.7% | 0.297 | 0.145 |
| TF-IDF char + SVD | 512 | 2 | 68.2% | 0.248 | 0.100 |
| `nomic_v15` | 768 | 2 | 9.8% | 0.318 | 0.140 |
| `nomic_v2` | 768 | 2 | 76.3% | 0.376 | 0.173 |
| `e5_large` | 1024 | 2 | 62.5% | 0.252 | 0.138 |
| `rubert_tiny2` | 312 | 2 | 21.5% | 0.283 | 0.149 |
| `rubert_base_dp` | 768 | 2 | 43.0% | 0.254 | 0.143 |
| `sbert_ru` | 1024 | 2 | 28.3% | 0.319 | 0.203 |
| `bge_m3` | 1024 | 2 | 54.7% | 0.351 | 0.230 |
| `Qwen3-Embedding-0.6B` | 1024 | 2 | 48.0% | 0.275 | 0.171 |
| `Qwen3-Embedding-4B` | 2560 | 2 | 65.3% | 0.274 | 0.151 |

### Вывод по символьным векторам

- Raw n-grams очень быстрые и дают умеренную согласованность с форматной структурой (`NMI=0.297`), сравнимую с частью нейросетевых моделей.
- TF-IDF + SVD компактнее raw-варианта, но на этом наборе дала сильно больше шума (68.2%) и хуже согласованность с форматами.
- На текущем смешанном корпусе HDBSCAN видит преимущественно макроструктуру из двух крупных групп, а не темы; поэтому этот тест отвечает на вопрос «можно ли кластеризовать такие векторы», но не является окончательной оценкой тематической кластеризации.
- Для production символьные n-граммы лучше рассматривать как CPU-бейзлайн, дедупликатор/ближайшие-дубликаты или дополнительный канал к embedding-модели, а не как полную замену семантической модели.

Данные сохранены в `experiment_models/reports/m9_char_ngrams.json`, векторы — `char_raw_4096__mixed_cpu__shard0.npy` и `char_tfidf_svd_512__mixed_cpu__shard0.npy`.

## Библиотеки по вариантам

### Python

| Вариант | Обязательные библиотеки | Комментарий |
|---|---|---|
| raw char n-grams | `scikit-learn`, `scipy`, `numpy` | Только `HashingVectorizer`; без модели и обучения |
| TF-IDF char + SVD | `scikit-learn`, `scipy`, `numpy`, `joblib` | Fitted `TfidfVectorizer` + `TruncatedSVD` сохраняются как артефакт |
| `rubert_tiny2`, `rubert_base_dp` | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | Стандартный AutoModel + mean-pooling |
| `e5_large`, `sbert_ru` | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | e5 требует префикс `passage:` |
| `bge_m3` | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | AutoModel, mean-pooling |
| `nomic_v15` | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | `trust_remote_code=True` (кастомный код Nomic) |
| `nomic_v2` | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | `trust_remote_code=True`; `megablocks` желательно |
| Qwen3-Embedding 0.6B/4B | `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy` | Last-token pooling вместо mean-pooling |
| Все (вспомогательное) | `huggingface_hub` | Только для скачивания; локально не нужен |
| Все (GPU) | CUDA-сборка PyTorch, драйвер NVIDIA | Только для GPU-инференса |
| Кластеризация | `hdbscan`, `scikit-learn`, `numpy`, `pandas` | Общее для всех вариантов |

`sentence-transformers` не требуется ни одному варианту.

### Jmix / Java

Рекомендуемый стек для Jmix-приложения: **ONNX Runtime Java** (инференс) + **DJL HuggingFace Tokenizers** (токенизация) + собственная реализация pooling и L2-нормализации.

| Вариант | Java-библиотеки | Комментарий для Jmix |
|---|---|---|
| raw char n-grams | Только JDK (+ опционально `it.unimi.dsi:fastutil` для разреженных векторов) | Не нужен ни ONNX, ни DJL: n-граммы и MurmurHash реализуются чистой Java |
| TF-IDF char + SVD | JDK + собственный код; артефакты (IDF-веса + SVD-матрица ~245 MB) выгружаются из Python | Самый неудобный для Java вариант: нужно перенести токенизацию, IDF и матрицу SVD |
| `rubert_tiny2`, `rubert_base_dp` | `com.microsoft.onnxruntime:onnxruntime`, `ai.djl.huggingface:tokenizers` | Экспорт в ONNX; mean-pooling + L2 в Java. Низкая сложность |
| `e5_large` | `com.microsoft.onnxruntime:onnxruntime`, `ai.djl.huggingface:tokenizers` | То же + префикс `passage:`. Низкая сложность |
| `sbert_ru`, `bge_m3` | `com.microsoft.onnxruntime:onnxruntime`, `ai.djl.huggingface:tokenizers` | То же. bge_m3 тяжелее по диску (4,3 GB) |
| `nomic_v15` | `ai.djl.pytorch:pytorch-engine` + native PyTorch, либо ONNX-экспорт | Средняя: кастомный код придётся отрезать при экспорте |
| `nomic_v2` | `ai.djl.pytorch:pytorch-engine` или Python-сервис (REST/gRPC) | Высокая: MoE + кастомный код; в Jmix практичнее сервис |
| Qwen 0.6B/4B | `ai.djl.pytorch:pytorch-engine` или Python-сервис | Высокая: causal-архитектура + last-token pooling; на CPU в Jmix практически неприменимо |
| Кластеризация | HDBSCAN в Java нет. Варианты: `smile`, `ELKI` или Python-сервис | Кластеризацию разумно держать на стороне Python |

**Maven-координаты для Java-стека:**
- `com.microsoft.onnxruntime:onnxruntime` — инференс ONNX;
- `ai.djl:api` + `ai.djl.huggingface:tokenizers` — токенизация HuggingFace;
- `ai.djl.pytorch:pytorch-engine` (+ `pytorch-native-cpu` или `pytorch-native-cuda`) — только если модель остаётся PyTorch;
- для GPU в Java: `onnxruntime_gpu` или CUDA-native DJL engine.

**Итог по интеграции:** минимальные зависимости у raw char n-grams (чистая Java, ноль нативных библиотек). Самые простые нейросетевые кандидаты для ONNX — `rubert_base_dp`, `rubert_tiny2`, `e5_large`, `sbert_ru`. Nomic v2 и Qwen в Jmix интегрировать нецелесообразно — им место в отдельном Python-сервисе.