# Pandoc Integration - Summary

## Изменения в архитектуре извлечения текста

### Новая стратегия экстракции

| Формат | Основной метод | Fallback 1 | Fallback 2 |
|--------|---------------|------------|------------|
| .txt | Прямое чтение | — | — |
| .csv | Прямое чтение | — | — |
| **.rtf** | **Pandoc CLI** | striprtf | — |
| **.doc** | **Pandoc CLI** | olefile | COM Word → LibreOffice |
| **.docx** | **Pandoc CLI** | python-docx | COM Word |
| **.odt/.ods/.odp** | **Pandoc CLI** | Pandoc Python lib | LibreOffice COM |
| Excel (.xls/.xlsx) | COM | — | — |
| PDF | PyMuPDF4LLM + OCR | fitz fallback | — |
| Изображения | Конвертация в PDF → текст | — | — |

### Созданные модули

```
extractors/
├── __init__.py              # Универсальный интерфейс process_file()
├── pandoc_extractor.py      # Pandoc CLI + Python library fallback
├── txt_extractor.py         # TXT, CSV (прямое чтение)
├── rtf_extractor.py         # RTF (Pandoc → striprtf)
├── olefile_extractor.py     # .doc (Pandoc → olefile → COM)
├── docx_extractor.py        # .docx (Pandoc → python-docx → COM)
├── excel_extractor.py       # Excel (COM)
├── pdf_extractor.py         # PDF (PyMuPDF4LLM + OCR)
├── image_extractor.py       # Изображения → PDF
└── xml_extractor.py         # XML/OpenDocument (Pandoc → XPath)
```

### Конфигурация (config.py)

Добавлены параметры:
```python
PANDOC_ENABLED: bool = True           # Включить Pandoc как основной метод
PANDOC_TIMEOUT: int = 120             # Таймаут CLI в секундах
PANDOC_PYTHON_FALLBACK: bool = True   # Использовать pandoc-библиотеку если CLI не найден
```

### Обновлённые функции

#### `extractors/__init__.py::process_file()`

Упрощённая логика с автоматическим выбором экстрактора:
```python
def process_file(file_path: Path, file_type: str) -> Dict:
    ext = file_path.suffix.lower()
    
    if ext == ".txt": return extract_txt(file_path)
    elif ext == ".csv": return extract_csv(file_path)
    elif ext == ".rtf": return extract_rtf(file_path)  # Pandoc → striprtf
    elif ext == ".doc": return extract_doc_olefile(file_path)  # Pandoc → olefile
    elif ext == ".docx": return extract_docx(file_path)  # Pandoc → python-docx
    elif ext in (".xls", ".xlsx"): return extract_excel(file_path)
    elif ext == ".pdf" or file_type == "pdf": return extract_pdf(file_path)
    elif ext in (".jpg", ".png", ...): return extract_image(file_path)
    elif ext == ".xml" or ext in (".odt", ".ods", ".odp"): return extract_xml(file_path)
```

#### `extractors/pandoc_extractor.py::extract_with_pandoc()`

Двухуровневая стратегия:
1. **Pandoc CLI** — быстрый и надёжный метод
2. **Python библиотека** (`pandoc` package) — если CLI не найден в PATH

```python
def extract_with_pandoc(file_path: Path, file_type: str) -> Optional[Dict]:
    # 1. Пробуем CLI
    result = _extract_with_pandoc_cli(file_path, input_format)
    if result: return result
    
    # 2. Fallback на Python библиотеку
    if cfg.PANDOC_PYTHON_FALLBACK:
        result = _extract_with_pandoc_python(file_path, input_format)
        if result: return result
    
    return None
```

### Исправленные ошибки

1. **`NoneType` вместо `Dict`** — все экстракторы теперь возвращают корректный словарь с полями `text`, `error`, `text_quality`
2. **Отсутствие Pandoc CLI** — добавлен fallback на Python библиотеку
3. **Unicode encoding errors** — тестовые скрипты используют ASCII-safe вывод

### Тестирование

```bash
# Проверка синтаксиса всех модулей
python -m py_compile extractors/*.py  # ✅ OK

# Проверка импортов
python -c "import main; import stage_1_main"  # ✅ OK

# Тест извлечения
python -c "from extractors import process_file; result = process_file(Path('file.docx'), 'docx'); print(type(result))"  # ✅ dict
```

### Преимущества новой архитектуры

1. **Единый инструмент** — Pandoc конвертирует .rtf/.doc/.docx/.odt в Markdown без COM/LibreOffice
2. **Лучшее качество текста** — Markdown output чище для LLM обработки
3. **Меньше зависимостей** — одна библиотека вместо olefile + python-docx + striprtf + LibreOffice
4. **Гибкость** — можно отключить Pandoc через `PANDOC_ENABLED=False`
5. **Надёжность** — многоуровневый fallback на случай сбоев

### Установка Pandoc (опционально)

```bash
# Windows: скачать установщик с https://pandoc.org/installing.html
# Или через Chocolatey:
choco install pandoc

# Python библиотека как альтернатива:
pip install pandoc
```

### Примечания

- Pandoc CLI **не обязателен** — система работает и без него (fallback на python-docx/olefile/striprtf)
- Для ODF (.odt/.ods/.odp) используется собственный парсинг XML в `xml_extractor.py` как fallback
- Excel и PDF используют COM/PyMuPDF4LLM без изменений
