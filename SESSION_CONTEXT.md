# SESSION_CONTEXT.md

## Cluster description
Кластеризация документов (110k+ PDF) через извлечение предметного фрагмента → LLM-саммаризация (Qwen3.5-4B) → nomic-embed → HDBSCAN.

## Important info for the next session
- **Config override file**: `config_override.json` существует и переопределяет `config.py`. Редактировать ОБА файла.
- **`report.txt`**: `D:\FileOrganizer\report.txt` — всегда отражает последний запуск.
- **LLM backend**: transformers (Qwen3.5-4B на cuda:0), НЕ llama-server
- **PyTorch**: 2.12.0+cu132 (`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`)
- **Aspose.Words for Java**: `D:\Yandex.Disk\Aspose\Aspose.Words for Java` — основной метод для Word-форматов
- **Архитектура оркестратора**: Airflow **нативно в WSL2** (BashOperator + /mnt/ interop). НЕ VM+SSH (legacy), НЕ web_server (legacy). См. секцию «Архитектура Airflow/WSL2» ниже.

---

## Сессия 2026-07-11: PDF-классификация и предметный фрагмент

### 1. classify_pdf() — переписана (`file_processor.py:415-473`)
- **Только первая страница** (конвенция проекта, gotcha 35 в AGENTS.md)
- Считаются **читаемые буквы** `[а-яА-ЯёЁa-zA-Z]`, а не все символы — mojibake (шрифты без ToUnicode) больше не проходит порог
- `find_tables()` вызывается один раз, только при наличии текста
- Ранний выход: вердикт известен сразу после проверки
- **~97 мс/файл** (было пропорционально числу страниц)

### 2. Субклассификация pdf_tables по содержанию (`file_processor.py:318-391`)
`classify_pdf()` для PDF с таблицей определяет подтип через словарь `_PDF_TABLES_SUBRULES`:
- `pdf_tables_fin` — финансовая первичка (~88%)
- `pdf_tables_tech` — техдокументация (~6%)
- `pdf_tables_contr` — договорные (~1%)
- `pdf_tables_reports` — отчёты, МТС-биллинг (~0.4%)
- `pdf_tables_other` — письма, ЕГРЮЛ, анкеты (~4%)
- `pdf_tables` (без подтипа) — fallback для LLM (<1%)

**Порядок правил**: `fin → tech → contr → reports → other`. fin — **строгий**: только явные названия форм первички. Универсальные термины (электроэнерг, вознаграждени, поставщик, стоимость услуг, акт № без даты) убраны — давали ложные срабатывания.

**Нормализация**: `_collapse_single_letter_runs()` — схлопывает серии одиночных букв (`С М Е Т А` → `СМЕТА`), не трогает многобуквенные слова.

### 3. Новые типы в config.py и database.py
- `FORMAT_TARGETS` и `TAGS`: `pdf_tables_fin/tech/contr/reports/other` → подпапки `Sorted/PDF_Tables/{sub}/`
- `database.py`: все 5 списков форматов (init_db, clear_stage4, get_embedding_table_name, create_backup_tables, get_format_stats) обновлены

### 4. extract_subject_fin() — `subject_extractor.py`
Извлекает «чистый» предметный фрагмент из фин.документа (~200-500 символов):
- **Способ 1**: `find_tables()` — точный ФНС-заголовок «Наименование товара (работ, услуг)» для счёт-фактуры/УПД, затем широкий паттерн «наименование»
- **Способ 2**: текстовый fallback — поиск строк с предметными ключевыми словами, фильтрация мусора (реквизиты, суммы прописью, подписи)
- Берёт **первые 2-5 строк** номенклатуры
- `_clean_subject()` — удаляет цифры, единицы измерения, ИНН/КПП, формы организаций
- **Классификация предмета** (аренда/связь/энерго/...) — задача LLM на этапе 3, regex-словарь удалён

### 5. Миграционный скрипт — `migrate_pdf_tables.py`
Переклассификация уже отсортированных файлов без отката этапа 1:
```bash
python migrate_pdf_tables.py --dry-run   # отчёт без перемещений
python migrate_pdf_tables.py             # миграция + обновление БД
```

### 6. PDF Viewer — `pdf_viewer.py`
Простой просмотрщик PDF для визуальной проверки классификации:
```bash
python pdf_viewer.py "D:\FileOrganizer\Sorted\PDF_Tables\fin"
```
- Выбор папки, листание файлов/страниц
- Масштаб: Fit / +/− / колесо мыши / пресеты
- Копирование имени файла в буфер (кнопка / Ctrl+C / двойной клик)

### 7. AGENTS.md обновлён
- Gotcha 35: конвенция «анализ только по первой странице»
- Gotcha 36: субклассификация pdf_tables по содержанию

### Текущее состояние Sorted/PDF_Tables (после прогона этапа 1)
| Папка | Файлов |
|---|---|
| PDF_Scan | 14 400 |
| PDF_Tables (всего) | 93 221 |
| PDF_Text | 18 621 |

### Что осталось сделать
- Интегрировать `extract_subject_fin()` в pipeline-обработчик `pdf_tables_fin` этапа 2
- Настроить LLM-промпт (этап 3) для классификации предмета из фрагмента
- Pipeline-обработчики для `pdf_tables_tech/contr/reports/other`

---

## Сессия 2026-07-16: Excel_Xlsx через Aspose.Cells + автозапуск Airflow

### 1. Excel_Xlsx пайплайн: Aspose.Cells → PDF → страницы 1-2 как PNG

Изменения для формата `excel_xlsx` на этапе 2:
- **Шаг 1**: `.xlsx` → `.pdf` через **Aspose.Cells for Java** (subprocess JVM), полный PDF
- **Шаг 2**: fitz обрезает PDF на месте (`insert_pdf` первые N страниц) до:
    - 1 страницы, если исходник = 1 лист
    - 2 страниц, если исходник = 2 листа
    - 2 первых страниц, если исходник >= 3 листов
- **Артефакт** сохраняется в `D:\FileOrganizer\Extracted\Excel_Xlsx\`:
  - `{stem}.pdf` — итоговый PDF (обрезанный до max 2 страниц)
- `result['text']` берётся из обрезанного PDF через fitz (чище, чем Pandoc, для сложных листов)
- `result['image']` = None (image-эмбеддинги для excel_xlsx не строятся)

**Fallback** (если Aspose.Cells недоступен — jar не скачан): старый путь Pandoc CLI + COM Excel, без PDF артефактов. Пайплайн не падает, откатывается автоматически.

### 2. Архитектура Aspose.Cells for Java

| Файл | Назначение |
|---|---|
| `D:\Yandex.Disk\Aspose\Aspose.Cells for Java\aspose_cells.py` | Python-обёртка `AsposeCells.convert_excel_to_pdf(input, output)` |
| `...\helpers\CellsToPdfHelper.java` | Java-хелпер (источник), вызывается обёрткой через `subprocess.run(['java', '-cp', ...])` |
| `...\compile_helpers.bat` | Компиляция `.java` → `.class` (`javac -cp lib\aspose-cells-*.jar`) |
| `...\README.md` | Инструкция по установке jar и компиляции |
| `extractors\cells_aspose_extractor.py` | Слой интеграции в проект: `_get_aspose_cells()` lazy singleton, `convert_xlsx_to_pdf()`, `render_pdf_pages_as_png(max_pages=2)`, `extract_text_from_pdf()` |
| `pipelines\excel_xlsx_pipeline.py` | Модифицированный пайплайн: Aspose путь + Pandoc/COM fallback |
| `config.py` | Добавлены `ASPOSE_CELLS_JAVA_DIR` и `cfg.EXTRA['excel_xlsx_pages']` = `D:\FileOrganizer\Extracted\Excel_Xlsx` |

### 3. Установка Aspose.Cells (выполнено)

⚠ **КРИТИЧНО**: Aspose.Cells for Java 20.3 **не поддерживает формат `.xls`** (Excel 97-2003 binary, 38.7% датасета = 12402 из 32024 файлов). Для `.xlsx` — работает отлично.

Решение — переход на **Aspose.Cells for .NET v26.7** (2026-07-09):
- Скачан `Aspose.Cells for .NET v26.7.0 (09 Jul 2026) + License Key & CRACK`
- Лицензия `Aspose.Total.NET.lic` (SubscriptionExpiry 2059 — покрывает все продукты Aspose.Total .NET)
- Поддерживает .xlsx + .xls + .xlsm + .xlsb
- Структура проекта `CellsToPdfServer\` (C#, .NET 10):

| Файл | Назначение |
|---|---|
| `CellsToPdfServer\CellsToPdfServer.csproj` | Консольный проект, зависимость Aspose.Cells.dll + System.Drawing.Common |
| `CellsToPdfServer\Program.cs` | .NET-сервер: чтение JSON из stdin, запись JSON в stdout, `Workbook.Save(out, SaveFormat.PDF)` |
| `lib\net10.0-windows7.0\Aspose.Cells.dll` | Aspose.Cells v26.7 .NET (18 MB) |
| `Aspose.Total.NET.lic` | Лицензия до 2059 |

Сборка: `dotnet build -c Release` — фреймворк-зависимая.

**Python-обёртка**: `AsposeCellsServer`自适应 режим:
- Если `dotnet.exe` + `CellsToPdfServer.dll` + `Aspose.Total.NET.lic` доступны → **.NET mode** (Рекомендуется)
- Иначе → **JVM fallback** (Aspose.Cells Java 20.3, только для `.xlsx`)

**Производительность** (тест .NET сервера):
- .xlsx: 484ms cold → 4ms warm
- .xls: 108ms cold → 2ms warm  
- В сумме поддержка .xls **увеличилась c 19622 (Aspose 20.3) до 32024 (100%) файлов**

### 4. Aspose.Cells for Java 20.3 (legacy)

| Что | Состояние |
|---|---|
| `D:\Yandex.Disk\Aspose\Aspose.Cells for Java\lib\aspose-cells-20.3.jar` | Установлен (fallback для standalone) |
| `...\lib\bcprov-jdk15on-160.jar` | BouncyCastle dependency для Cells 20.3 |
| `...\helpers\CellsToPdfServer.class` | Persistent JVM-сервер (subprocess stdin/stdout JSON) |
| `...\helpers\CellsToPdfHelper.class` | Разовый JVM-запуск (fallback) |
| Лицензия | `Aspose.Total.lic` (SubscriptionExpiry 2020-04-03) — НЕ покрывает Aspose.Cells 22.x+ |

⚠ Aspose.Cells Java 20.3 поддерживает **только `.xlsx`** (OOXML). Для `.xls` кидает `Invalid encoding: null`. Используется только как fallback когда .NET недоступен.

### 4. Автозапуск Airflow при логоне Windows (Task Scheduler)

Реализован через Windows Task Scheduler, триггер `At user logon` (`ONLOGON`):

| Файл | Назначение |
|---|---|
| `airflow_wsl\start_airflow_windows.cmd` | Обёртка на одну команду: `wsl.exe -d Ubuntu -u root -- /opt/airflow/start_airflow.sh` |
| `airflow_wsl\airflow_autostart.xml` | Описатель задачи Task Scheduler 2.0 (LogonTrigger, InteractiveToken, LeastPrivilege, ExecutionTimeLimit PT5M) |
| `airflow_wsl\register_task.ps1` | PowerShell-регистратор из XML |
| `airflow_wsl\README.md` | Обновлён: секция «Автозапуск при логоне Windows (Task Scheduler)» с 3 вариантами регистрации |

**Регистрация задачи** (1 раз, из elevated PowerShell):
```powershell
D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\airflow_wsl\register_task.ps1
```

(Не-elevated PowerShell, в т.ч. opencode-сессия, не имеет прав на запись в Task Scheduler — даже в пользовательской папке `\Users\<user>` получаем "Отказано в доступе". Файлы готовы, регистрация — за пользователем из админ-сессии.)

После регистрации:
- При логоне → Task Scheduler → `wsl -d Ubuntu` → systemd поднимает PostgreSQL → `start_airflow.sh` поднимает scheduler+webserver через nohup (~2-3с)
- UI: http://localhost:8080 (admin/admin)
- Логи: `\\wsl$\Ubuntu\opt\airflow\logs\{scheduler,webserver}.out`

### 5. Запуск Airflow прямо сейчас (без Task Scheduler)

```powershell
wsl -d Ubuntu -u root -- /opt/airflow/start_airflow.sh
# Проверка:
(Invoke-WebRequest http://localhost:8080/health -UseBasicParsing).StatusCode  # 200
```

---

## Сессия 2026-07-17: CellsToPdfServer — FitToPagesWide + extract_text; Excel-эксперимент на direct text

### 1. CellsToPdfServer (Program.cs) — два изменения протокола
- **FitToPagesWide=1 / FitToPagesTall=0** для каждого листа перед `wb.Save(PDF)`:
  широкие таблицы (40+ колонок, напр. `dimension=AV31` + `orientation="portrait"`)
  раньше разбивались по горизонтали и правые колонки терялись при обрезке PDF
  до 1-2 страниц. Теперь все колонки вписываются в ширину одной страницы.
- **Новая команда `extract_text`**: `{"cmd":"extract_text","input":...}` →
  `{"status":"ok","text":"..."}` — прямой TSV-дамп всех листов
  (`TxtSaveOptions(SaveFormat.Tsv){ExportAllSheets=true}`), лимит 200k символов.
  Без PDF вообще → нет класса проблем «ориентация/обрезание/порядок страниц».
- Python: `AsposeCellsServer.extract_text()` (только .NET mode; общий `_send_request`),
  `extractors/cells_aspose_extractor.py::extract_text_direct()`.

### 2. Excel-эксперимент (experiment_excel) — direct text
- `experiment_config.py`: `EXTRACT_MODE='direct'` (TSV для эмбеддингов),
  `WRITE_PDF_ARTIFACT=True` (PDF только для визуального аудита в pdf_viewer).
- `phase_1`: `normalize_tsv()` — схлопывание серий табов (40+ пустых ячеек),
  экономия бюджета MAX_EXTRACT_LEN=1000.
- Старые 32k PDF (без fit-to-width) удалены, перегенерированы новыми.
- Прогон: phase 1 — 2969s (13.4 f/s, direct+PDF), failed 4; phase 2 —
  13 кластеров, шум 43.3%, silhouette 0.294 (MIN_CLUSTER_SIZE_FACTOR=0.01).
- `show_cluster.py` — визуальный аудит кластеров через hardlink-папки
  `D:\FileOrganizer\Reports\ClusterViews\cluster_N\` + pdf_viewer.

### 3. pdf_viewer.py — новое управление
- Колесо мыши = листание файлов (раньше — масштаб).
- Масштаб = только кнопки/пресеты.
- Drag зажатой ЛКМ = панорамирование при выходе за рамки (scan_mark/scan_dragto),
  курсор `fleur` в ручном масштабе.

### 4. Замеченное
- Нестандартные xlsx: `xl/SharedStrings.xml` с заглавной S → openpyxl падает
  (ищет lowercase), Aspose читает нормально.
- TSV-дамп оборачивает поля с кавычками в `"..."` с удвоением (CSV-конвенция) —
  при сравнении probe-строк учитывать.
- hdbscan 0.8.44 + sklearn 1.9: `metric='cosine'` НЕ поддерживается (ValueError).
  Рабочий путь: L2-нормализованные эмбеддинги + `metric='euclidean'`.
  **Тот же баг в `cluster_engine.py:66` основного пайплайна — требует фикса.**

---

## Сессия 2026-07-20: Контрактная экстракция (ОСНОВАНИЕ + КОНТРАГЕНТ)

### Цель
Привязка фин.первички к договорным документам: помимо ПРЕДМЕТ/ТИП LLM
извлекает ОСНОВАНИЕ (договор № + дата) и КОНТРАГЕНТ (кортеж: название | ИНН | КПП | регион).

### Файлы последнего прогона (experiment_excel/)
| Файл | Роль |
|---|---|
| `phase_5v_validate_contract.py` | 4-полевой промпт + валидация на 1000. Содержит: SYSTEM_PROMPT/_USER_TMPL, `_DOC_FILTER_SQL` (фин.первичка), `inn_valid_10/12`, `is_self`, `parse_kp`, `parse_raw` |
| `phase_6a_sample_extract.py` | Выборка + переизвлечение «первая+последняя страница» → `temp_contract_files` |
| `phase_6b_contract_fields.py` | Полный LLM-прогон (шард/GPU, возобновляемый) → `temp_contract_fields` |

### Данные в БД (file_organizer_db)
- `temp_contract_files` — 13 773 док. (10 000 tables фин.первичка + 3 773 text фин.первичка), тексты «первая+последняя стр.» ≤3000 симв.
- `temp_contract_fields` — результаты: predmet, tip, osnovanie, kp_name, kp_inn, kp_kpp, kp_region, kp_self, inn_in_text, raw
- Пулы фин.первички (фильтр `_DOC_FILTER_SQL`): PDF_Tables 73 526, PDF_Text 3 773

### Результаты прогона (13 773 док., ~3 ч на 2 GPU)
| Метрика | tables | text | Всего |
|---|---|---|---|
| ОСНОВАНИЕ | 88.6% | 60.2% | 80.8% |
| КОНТРАГЕНТ (без self) | 87.3% | 70.0% | 82.5% |
| ИНН | 83.6% | 55.8% | 76.0% |
| — валиден чексуммой 10/12 | 97.3% | 75.5% | 92.9% |
| — есть в тексте | 95.6% | 77.2% | 91.9% |
| КПП | 70.7% | 52.2% | 65.6% |
| Регион | 88.2% | 80.4% | 86.0% |
| Self отсеяно | 1053 | 912 | 1965 |

### Ключевые уроки (для следующих экспериментов)
1. **ИНН ИП = 12 цифр** (2 контрольные), юрлица = 10. Валидировать обе.
2. **Self-фильтр обязателен** (ВымпелКом/вымпел/билайн варианты): 11–24% двусторонних документов.
3. **Анти-галлюцинация ИНН**: сверка подстрокой с исходным текстом (~4–8% выдуманных/с пробелами).
4. **Парсер полей**: `[ \t]*`, не `\s*` — иначе пустое поле заливает следующим.
5. **Правило страниц**: «первая + последняя» универсально (реквизиты в шапке и/или хвосте), классификация заранее не нужна.
6. **ОСНОВАНИЕ форматы**: «Договор 50130002010548 от 01.07.2013», «№Д24112 от 13.05.2024», «BEE.001.5105 от 10.12.2020» — LLM нормально читает все.
7. **Филиалы ловятся КПП+регионом**: «ФИЛИАЛ В ЧУВАШСКОЙ РЕСПУБЛИКЕ ПАО РОСТЕЛЕКОМ | КПП 213043001 | Чебоксары».
8. Скорость 4-полевого извлечения: ~0.78 д/с/GPU (длиннее вывод, max_new_tokens=140).

### Следующий шаг (не начат)
**Линковка к договорам**: contracts_registry (ТИП ∈ Договор/Приложение/Доп.соглашение),
нормализация номеров (гомоглифы, разделители, ведущие нули) + pg_trgm нечёткий мэтч,
скоринг = 0.55·sim(номер) + 0.30·score_cp(ИНН/КПП/название/регион) + 0.15·match(дата),
трёхзонное решение (auto ≥0.85 / review 0.55–0.85 / unmatched).

---

## Сессия 2026-07-21: Валидация контрагента по метаданным ЭДО (Диадок)

### Задача
Проверка извлечённых LLM данных контрагента (kp_name/kp_inn/kp_kpp из
`temp_contract_fields`) по эталону — метаданным оператора ЭДО:
MySQL `events_26.doc_in` ⨝ `diadoc_api.contragens_info`
(`doc_in.CounteragentBoxId = contragens_info.BoxId`).
EntityId = подстрока слева от первой точки в имени файла.

### Реализация — `experiment_excel/phase_7_edo_validate.py`
- EntityId из basename; батч-запросы MySQL по 1000 id (pymysql, localhost:3308).
- Снимок ЭДО-данных в PG **`temp_contract_edo`** (doc_id PK, entity_id, found,
  edo_inn/kpp/fullname/shortname) — пригодится для линковки.
- Сравнение: ИНН/КПП — точное digits-only; имя — нормализация (оргформы, кавычки)
  + rapidfuzz token_sort_ratio ≥ 85 (проверяются и ShortName, и FullName).
- Отчёт: `experiment_excel/work/contract_edo_validate_report.txt`;
  расхождения: `work/contract_edo_divergences.csv` (3110 строк).
- Excel для визуального анализа: `phase_7b_edo_xlsx.py` →
  `work/contract_edo_compare.xlsx` (13 773 строки; лист «Данные» с цветовой
  индикацией статусов + автофильтр, лист «Сводка» с метриками).

### Результаты (13 773 док.; найдено в doc_in 13 074 = 94.9%)
Основной контур — **без self (n=11 139)**:
| Поле | match | mismatch | llm_missing |
|---|---|---|---|
| ИНН | 65.0% | 11.3% | 23.6% |
| КПП | 45.8% | 18.6% | 27.6% |
| ИМЯ | 85.0% | 11.1% | 3.9% |

Полное совпадение ИНН+КПП+ИМЯ: 42.6%. tables заметно лучше text
(ИНН 77.3% vs 27.6%; ИМЯ 87.5% vs 77.6%).

### Глубокий анализ расхождений (без self)
- **ИНН mismatch (1263)**: 51.5% — невалидная чексумма (галлюцинация),
  48.5% — валидный, но ДРУГОЙ ИНН (извлечён ИНН не той стороны:
  агент/покупатель/грузополучатель). 763 случая — имя верное, ИНН чужой.
- **КПП mismatch (2077)**: 1086 (52%) при совпавшем ИНН → КПП филиала
  (ЭДО-бокс = головной КПП) — НЕ ошибка LLM. КПП — слабый индикатор ошибки.
- **Self (1935)**: ИНН mismatch 64.5% — подтверждает, что LLM взяла нашу сторону.

### Выводы для линковки
1. ЭДО-метаданные — эталон контрагента при наличии (94.9% покрытие);
   LLM-поля — fallback для 5.1% не найденных в doc_in.
2. ИМЯ у LLM надёжно (85%); ИНН — нет (галлюцинации цифр, чужой ИНН).
   В скоринге линковки ИНН из ЭДО > ИНН из LLM.
3. КПП нельзя требовать точного совпадения (филиалы), только как мягкий сигнал.
