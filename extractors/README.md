# Модуль извлечения текста (extractors)

## Структура

```
extractors/
├── __init__.py          # Универсальный интерфейс + утилиты
├── txt_extractor.py     # TXT, CSV файлы
├── rtf_extractor.py     # RTF файлы (striprtf)
├── olefile_extractor.py # .doc файлы (olefile)
├── docx_extractor.py    # .docx файлы (python-docx)
├── excel_extractor.py   # Excel файлы (COM)
├── pdf_extractor.py     # PDF файлы (PyMuPDF4LLM + OCR)
├── image_extractor.py   # Изображения → PDF → текст
├── xml_extractor.py     # XML/OpenDocument структура
└── office_com_extractor.py  # COM Word/Excel конвертация
```

## Функции

### Универсальный интерфейс

- `process_file(file_path: Path, file_type: str) -> Dict` — автоматическое определение типа файла и вызов соответствующего экстрактора

### Экстракторы по форматам

| Модуль | Форматы | Метод |
|--------|---------|-------|
| txt_extractor | .txt, .csv | Прямое чтение (без COM) |
| rtf_extractor | .rtf | striprtf |
| olefile_extractor | .doc | Python-парсинг OLE2 (~1 мс) |
| docx_extractor | .docx | python-docx |
| excel_extractor | .xls/.xlsx | COM Excel |
| pdf_extractor | .pdf | PyMuPDF4LLM + OCR |
| image_extractor | .jpg/.png/.gif | Конвертация в PDF → текст |
| xml_extractor | .xml, .odt/.ods | XPath структура / OpenDocument XML |
| office_com_extractor | .doc/.docx/.xls/.xlsx | COM Word/Excel (fallback) |

### Утилиты из `__init__.py`

- `move_file_to_target(file_path, category)` — перемещение в целевую категорию
- `move_file_to_error(file_path)` — перемещение в ErrorFiles
- `is_opendocument_xml(file_path)` — проверка OpenDocument XML
- `detect_opendocument_xml_type(file_path)` — определение типа ODF (.odt/.ods/.odp)
- `detect_file_type_by_signature(file_path)` — определение формата по сигнатуре

## Fallback цепочки

### .doc файлы:
1. olefile (~1 мс) → если пусто
2. COM Word (~7 сек) → если пусто  
3. LibreOffice → PDF → PyMuPDF4LLM

### .docx файлы:
1. python-docx → если ошибка/пусто
2. COM Word (fallback)

### ODF (.odt/.ods):
1. LibreOffice → PDF → PyMuPDF4LLM

## Обновление существующего кода

**main.py:**
```python
# Было:
import file_processor
result = file_processor.process_single_file(fp, cat)

# Стало:
from extractors import process_file
result = process_file(fp, cat)
```

**stage_1_main.py:**
```python
# Было:
import file_processor
futures = [loop.run_in_executor(executor, file_processor.process_single_file, path, cat)...]

# Стало:
from extractors import process_file
futures = [loop.run_in_executor(executor, process_file, path, cat)...]
```

## Преимущества рефакторинга

1. **Изоляция логики** — каждый формат в отдельном модуле
2. **Тестируемость** — можно тестировать экстракторы по отдельности
3. **Расширяемость** — легко добавить новый формат (создать новый файл)
4. **Читаемость** — меньше кода в `file_processor.py` (~100 строк вместо ~1000)
5. **Переиспользование** — COM экстракторы доступны из разных модулей
