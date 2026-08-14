# file_processor.py
# Stage 1 helpers: детекция типов файлов по сигнатурам и перемещение в папки

import io
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from config import cfg
from logger_utils import logger

MAGIC_SIGNATURES = [
    (0, b'%PDF',                           '.pdf',   'pdf'),
    (0, b'\xD0\xCF\x11\xE0\xA1\xB1',      None,     'ole2'),
    (0, b'PK\x03\x04',                     None,     'zip'),
    (0, b'\xFF\xD8\xFF',                   '.jpg',   'image'),
    (0, b'\x89PNG\r\n\x1a\n',              '.png',   'image'),
    (0, b'GIF87a',                          '.gif',   'image'),
    (0, b'GIF89a',                          '.gif',   'image'),
    (0, b'BM',                              '.bmp',   'image'),
    (0, b'II*\x00',                         '.tif',   'image'),
    (0, b'MM\x00*',                         '.tif',   'image'),
    (0, b'RIFF',                            None,     'riff'),
    (0, b'{\\rtf',                          '.rtf',   'word'),
    (0, b'<?xml',                           '.xml',   'xml'),
    (0, b'\xEF\xBB\xBF<?xml',              '.xml',   'xml'),
    (0, b'\xFF\xFE<?xml',                   '.xml',   'xml'),
    (0, b'\xFE\xFF<?xml',                   '.xml',   'xml'),
    (0, b'\x1F\x8B',                        '.gz',    None),
    (0, b'Rar!\x1a\x07',                    None,     None),
    (0, b'\x30\x80\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07', None, 'pkcs7'),
]

ZIP_CONTENT_TYPES = {
    'word':  {b'word/', b'document.xml', b'document.xml.rels'},
    'excel': {b'spreadsheet', b'worksheet', b'styles.xml', b'sheet'},
    'odt':   {b'mimetype', b'content.xml', b'META-INF/manifest.xml'},
    'ods':   {b'mimetype', b'content.xml', b'META-INF/manifest.xml'},
}

ZIP_MIMETYPES = {
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'word',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
    'application/vnd.oasis.opendocument.text': 'word',
    'application/vnd.oasis.opendocument.spreadsheet': 'excel',
}


def _detect_zip_subtype(path: Path) -> Optional[Tuple[str, str]]:
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            names_bytes = '\n'.join(names).encode('utf-8', errors='replace')

            if '[Content_Types].xml' in names:
                try:
                    ct = zf.read('[Content_Types].xml').decode('utf-8', errors='replace')
                    for mime, cat in ZIP_MIMETYPES.items():
                        if mime in ct:
                            ext = '.docx' if cat == 'word' else '.xlsx'
                            return (ext, cat)
                except Exception:
                    pass

            if any(n.startswith('word/') for n in names):
                return ('.docx', 'word')
            if any(n.startswith('xl/') for n in names):
                return ('.xlsx', 'excel')
            if 'mimetype' in names:
                try:
                    mime = zf.read('mimetype').decode('utf-8', errors='replace').strip()
                    for m, cat in ZIP_MIMETYPES.items():
                        if m in mime:
                            ext = '.docx' if cat == 'word' else '.xlsx'
                            return (ext, cat)
                except Exception:
                    pass

            if any(n.startswith('META-INF/') for n in names):
                try:
                    manifest = zf.read('META-INF/manifest.xml').decode('utf-8', errors='replace')
                    if 'text' in manifest.lower():
                        return ('.odt', 'word')
                    if 'spreadsheet' in manifest.lower():
                        return ('.ods', 'excel')
                except Exception:
                    pass

            return ('.zip', None)
    except Exception:
        return None


def _detect_ole2_subtype(path: Path) -> Optional[Tuple[str, str]]:
    """Определяет подтип OLE2: doc, xls, ppt.

    Fast path — поиск по байтам в первых 64 КБ файла (быстро, ~100x быстрее olefile).
    Slow path — olefile.listdir() для редких файлов, где имена потоков за пределами 64 КБ.
    """
    try:
        size = path.stat().st_size
        if size < 512:
            return None

        # Fast path: байтовый поиск UTF-16 LE имён потоков в первых 64 КБ
        try:
            with open(path, 'rb') as f:
                data = f.read(min(size, 65536))

            if 'Workbook'.encode('utf-16-le') in data:
                return ('.xls', 'excel')
            if 'WordDocument'.encode('utf-16-le') in data:
                return ('.doc', 'word')
            if 'Book'.encode('utf-16-le') in data:
                return ('.xls', 'excel')
            if 'PowerPoint Document'.encode('utf-16-le') in data:
                return ('.ppt', 'word')
            if '1Table'.encode('utf-16-le') in data or '0Table'.encode('utf-16-le') in data:
                return ('.doc', 'word')
            if '_xlwd.MSExcelWorkspace'.encode('utf-16-le') in data:
                return ('.xls', 'excel')
        except Exception:
            pass

        # Slow path: olefile для файлов, где потоки за пределами 64 КБ
        try:
            import olefile
            ole = olefile.OleFileIO(str(path))
            stream_names = ole.listdir()
            ole.close()

            name_set = {entry[-1] for entry in stream_names if entry}
            for name, result in (('Workbook', ('.xls', 'excel')),
                                 ('Book', ('.xls', 'excel')),
                                 ('_xlwd.MSExcelWorkspace', ('.xls', 'excel')),
                                 ('WordDocument', ('.doc', 'word')),
                                 ('1Table', ('.doc', 'word')),
                                 ('0Table', ('.doc', 'word')),
                                 ('PowerPoint Document', ('.ppt', 'word'))):
                if name in name_set:
                    return result
        except Exception:
            pass

        return None
    except Exception:
        return None


def _detect_riff_subtype(path: Path) -> Optional[Tuple[str, str]]:
    try:
        with open(path, 'rb') as f:
            f.seek(8)
            subtype = f.read(4)
        if subtype == b'WEBP':
            return ('.webp', 'image')
        if subtype == b'AVI ':
            return ('.avi', None)
        return ('.avi', None)
    except Exception:
        return None


def is_opendocument_xml(file_path: Path) -> bool:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        root_ns = root.tag.split('}')[0][1:] if '}' in root.tag else ''
        if 'urn:oasis:names:tc:opendocument' in root_ns:
            return True
        return False
    except Exception:
        return False


def detect_opendocument_xml_type(file_path: Path) -> Tuple[str, str]:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        root_ns = root.tag.split('}')[0][1:] if '}' in root.tag else ''
        if 'spreadsheet' in root_ns.lower():
            return ('.ods', 'excel')
        if 'presentation' in root_ns.lower():
            return ('.odp', 'word')
    except Exception:
        pass
    return ('.odt', 'word')


def _is_likely_text(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
        if not chunk:
            return False
        null_count = chunk.count(b'\x00')
        if null_count > len(chunk) * 0.1:
            return False
        try:
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            try:
                chunk.decode('cp1251')
                return True
            except UnicodeDecodeError:
                try:
                    chunk.decode('latin-1')
                    return True
                except UnicodeDecodeError:
                    return False
    except Exception:
        return False


def detect_file_type_by_signature(file_path: Path) -> Optional[Tuple[str, str]]:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(512)

        if len(header) == 0:
            return None

        for offset, magic, ext, category in MAGIC_SIGNATURES:
            if category is None and ext is None:
                continue
            if len(header) < offset + len(magic):
                continue
            if header[offset:offset + len(magic)] == magic:
                if category == 'zip':
                    result = _detect_zip_subtype(file_path)
                    if result:
                        return result
                    return ('.zip', None)

                if category == 'ole2':
                    result = _detect_ole2_subtype(file_path)
                    if result:
                        return result
                    # Подтип не определён — сохраняем исходное расширение
                    orig = file_path.suffix.lower()
                    cat_map = {'.xls': 'excel', '.doc': 'word', '.ppt': 'word'}
                    return (orig, cat_map.get(orig))

                if category == 'riff':
                    result = _detect_riff_subtype(file_path)
                    if result:
                        return result
                    return None

                if ext and category:
                    return (ext, category)

        if _is_likely_text(file_path):
            return ('.txt', 'word')

        return None

    except Exception as e:
        logger.error(f"Signature detection error for {file_path.name}: {e}")
        return None


def move_file_to_target(file_path: Path, target_category: str) -> Optional[Path]:
    try:
        target_dir = cfg.TARGETS.get(target_category)
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / file_path.name
            if file_path.resolve() == target_path.resolve():
                return target_path

            cnt = 1
            while target_path.exists():
                target_path = target_dir / f"{file_path.stem}_{cnt}{file_path.suffix}"
                cnt += 1
            shutil.move(str(file_path), str(target_path))
            return target_path
    except Exception as e:
        logger.error(f"Move error {file_path}: {e}")
    return None


def move_file_to_error(file_path: Path) -> Optional[Path]:
    try:
        cfg.ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        target = cfg.ERRORS_DIR / file_path.name
        cnt = 1
        while target.exists():
            target = cfg.ERRORS_DIR / f"{file_path.stem}_{cnt}{file_path.suffix}"
            cnt += 1
        shutil.move(str(file_path), str(target))
        return target
    except Exception as e:
        logger.error(f"Error moving to errors {file_path}: {e}")
    return None


def _collapse_single_letter_runs(s: str) -> str:
    """Схлопывает серии одиночных букв, разделённых пробелами: «С М Е Т А» -> «СМЕТА».

    НЕ трогает многобуквенные слова: «Информация для учета» остаётся как есть,
    чтобы не сломать regex-матчи на многословных терминах.
    """
    return re.sub(
        r'(?<![\w])([а-яА-ЯёЁa-zA-Z](?:\s+[а-яА-ЯёЁa-zA-Z])+)(?![\w])',
        lambda m: re.sub(r'\s+', '', m.group(1)),
        s, flags=re.UNICODE
    )


# Словарь-классификатор содержания pdf_tables (по первой странице).
# Покрытие ~99% на выборке 1000 случайных файлов из 93k PDF_Tables.
# Порядок проверки: fin -> tech -> contr -> reports -> other.
# fin стоит ПЕРВЫм (наиболее массовая категория), но СТРОГИМ:
# только явные названия форм первички. Универсальные термины
# (электроэнерг / вознаграждени / детализац / расшифровк / задолженность
# по ежемес / счёт № без «на оплату» / акт № без числа / поставщик /
# покупатель / стоимость услуг) УБРАНЫ — они давали ложные срабатывания
# на техдокументации, договорах, МТС-биллинге, анкетах.
_PDF_TABLES_SUBRULES = [
    # fin — финансовая первичка (счета на оплату, УПД, акты, накладные, КС-2/3)
    ("fin", re.compile(
        r'счет\s*на\s*оплату|счёт\s*на\s*оплату|'
        r'счет-фактура|счёт-фактура|универсальный\s+передаточный|'
        r'акт\s+сверки|акт\s+взаимозачета|акт\s+взаимореализ|акт\s+зачета|'
        # акты работ/услуг — конкретные формы.
        # «акт № N от <дата>» — типовой фин.акт (с «от» и датой; не «акт № 3.2»).
        r'акт\s+№\s*\d+\s+от\s+\d|'
        # «акт приёма материальных ценностей» / «акт об оказании услуг»
        r'акт\s+(о\s+приемке|о\s+приёмке|об\s+оказании\s+услуг|выполненных|выполненых|оказанных|'
        r'приема\s+материальн|приемки\s+материальн|приёма\s+материальн|приёмки\s+материальн|'
        r'выдачи|на\s+внутрисклад)|'
        # «сдачи-приемки выполненных» — фин.акт без слова «акт» непосредственно перед
        r'сдачи\s*-\s*приемки\s+выполненн|сдачи\s*-\s*приёмки\s+выполненн|'
        r'приложение\s+к\s+акту|'
        r'товарная\s+накладная|торг-12|транспортная\s+накладная|'
        r'\bкс\s*-\s*2\b|\bкс\s*-\s*3\b|справка\s+о\s+стоимости|'
        r'ведомость\s+показаний|показаний\s+приборов\s+учета|'
        # электросчёт — только конкретные формы (не голое «электроэнерг»)
        r'акт\s+электропотребления|акт\s+потребления|акт\s+снятия\s+показан|'
        r'электросчетчик|'
        r'расч[её]т\s+электроэнерг|возмещение\s+затрат\s+по\s+э|'
        r'расчетно-платежн|расчётно-платёжн|ведомость\s+электропотребл|информация\s+для\s+учета|'
        # мощность / энергия
        r'профиль\s+мощности|ставка\s+за\s+мощность|ставка\s+за\s+энергию|'
        r'кнд\s+1121005|'
        # расчёты вознаграждений/аренды — только в форме «расчёт ...»
        r'расч[её]т\s+вознаграждени|'
        r'расч[её]т\s+(переменной\s+части\s+)?аренд|'
        # платёжные документы
        r'платежное\s+поручение|сбер\s*бизнес|сбербизнес|поступ\.?\s*в\s+банк\s+плат|'
        # счётчики
        r'этаж\s+№\s*счетчик|счетчик\s+на\s+\d|'
        # Ростелеком: «СЧЁТ № N-NNNNNN от <дата>» + «Лицевой счет абонента»
        r'лицевой\s+сч[её]т\s+абонента|'
        r'сч[её]т\s+№\s*\d+-\d+\s+от\s+\d{1,2}[\.\s]|'
        r'Rent_.*invoice',
        re.I
    )),
    # tech — техдокументация (РД/РП/ПД, сметы, монтаж, АТП, ПЗ, журналы приёмки)
    ("tech", re.compile(
        r'рабочая\s+документация|рабочий\s+проект|проектная\s+документация|пояснительная\s+записка|'
        r'строительство\s+базовой\s+станции|базовая\s+станция|сеть\s+сотовой\s+радиотелефонной|'
        r'Взам\.?\s*инв|Инв\.?\s*N\s*подл|Стадия\s+Лист|Формат\s+А[134]|'
        r'СМЕТА|смета|локальный\s+сметный|типовая.*смет|'
        r'акт\s+технической\s+приемки|АКТ\s+ДОПУСК|УСТАНОВИЛА:|техническая\s+готовность|'
        r'демонтаж|монтаж\s+оборудования|произведен\s+монтаж|пусконаладоч|'
        r'основные\s+технические\s+решения|\bОТР\b|разграничени|АРБП|инцидент|'
        r'заключение\s+о\s+соответствии|\bПЗ\b|АМО|перенос\s+РРС|'
        # журналы приёмки/замечаний — характерные признаки (ПЗ 15932 и т.п.)
        r'при[её]мк[аи]\s+заказчик|приоритет.{0,5}раздел|формулировка\s+замечания|'
        r'значение\s+параметра.{0,10}наличие\s+замечания',
        re.I
    )),
    # contr — договорные (договоры, допсоглашения, оферты)
    ("contr", re.compile(
        r'договор\s*(№|аренды|подряда|поставки|субаренды|на\s+|энергоснабжения|об\s+оказании)|'
        r'договор\s+об\s+оказании\s+услуг\s+связи|'
        r'дополнительное\s+соглашение|\bоферта\b|предмет\s+договора|'
        r'^\s*соглашение\s*№|соглашение\s+№|'
        r'заявка\s+\d+|бонусн|bonus|framework\s+agreement|предложение\s+о\s+заключении|'
        r'оператор\s+оказывает\s+услуги\s+связи|действует\s+на\s+основании.*услуги\s+связи',
        re.I
    )),
    # reports — отчёты (о работах, по ТО, предрейсовые, SLA, МТС-биллинг)
    ("reports", re.compile(
        r'отчет\s+о\s+выполненн|отчёт\s+о\s+выполненн|отчет\s+комитенту|отчёт\s+комитенту|'
        r'отчет\s+о\s+реализован|отчёт\s+о\s+реализован|отчет\s+по\s+ТО|отчёт\s+по\s+ТО|'
        r'отчет\s+о\s+работе|отчёт\s+о\s+работе|отчет\s+о\s+соблюдении|отчёт\s+о\s+соблюдении|'
        r'отчет\s+по\s+объемам|отчёт\s+по\s+объемам|'
        r'журнал\s+мо|предрейсов|предрейсовый|'
        r'ID\s+заявки|заявка\s+по\s+SLA|'
        # МТС-биллинг (отчёты по SMS-запросам, выписки по абонентской задолженности)
        r'программа\s+номер\s+доступа|задолженность\s+по\s+ежемесячн|менеджер\s*:',
        re.I
    )),
    # other — письма, уведомления, выписки из ЕГРЮЛ, сертификаты, анкеты
    ("other", re.compile(
        r'заявление|уведомление|уведомля|\bписьмо\b|'
        r'смена\s+оператора|смена\s+реквизитов|'
        r'выписк[аи]\s+из\s+(егрюл|единого\s+государственного)|лист\s+записи|'
        r'dual\s+purpose|military|настоящим\s+уведомляет|'
        r'сертификат\s+страхования|сертификат|напоминание|ранее\s+отправленный\s+вам|'
        r'о\s+подписании\s+договора|digital\s+approval|'
        r'\bанкета\b|анкета\s+для\s+клиент|анкета\s+контрагент',
        re.I
    )),
]


def _classify_pdf_tables_content(page_text: str) -> Optional[str]:
    """Возвращает подтип pdf_tables по содержанию первой страницы.

    Args:
        page_text: текст первой страницы PDF.

    Returns:
        Один из: "fin", "tech", "contr", "reports", "other" или None,
        если документ не распознан словарём (попадает в общий "pdf_tables").
    """
    t = _collapse_single_letter_runs(page_text)
    for sub, rx in _PDF_TABLES_SUBRULES:
        if rx.search(t):
            return sub
    return None


def classify_pdf(file_path: Path, text_threshold: int = 100) -> str:
    """Определяет категорию PDF по ПЕРВОЙ странице.

    Конвенция проекта: для анализа/классификации во всех форматах
    используется только первая страница документа.

    Считаются читаемые буквы (кириллица/латиница), а не все символы —
    защита от mojibake (шрифты без ToUnicode дают мусорные глифы,
    которые проходили порог по длине строки, но текст нечитаем).

    Для ``pdf_tables`` дополнительно определяется подтип по содержанию
    (словарь-классификатор): ``fin``, ``tech``, ``contr``, ``reports``,
    ``other``. Подтип прибавляется к коду через подчёркивание
    (``pdf_tables_fin`` и т.д.). Не опознанные словарём документы
    возвращаются как ``pdf_tables`` (общая папка, для LLM-дообработки).

    Returns:
        "pdf_scan"             — букв на первой странице < threshold
                                 (скан / CAD-экспорт в кривых / битая кодировка)
        "pdf_text"             — есть читаемый текст, таблиц нет
        "pdf_tables_fin"       — финансовая первичка (счета, УПД, акты, ...)
        "pdf_tables_tech"      — техдокументация (РД/РП/ПД, сметы, КС-2/3, ...)
        "pdf_tables_contr"     — договорные (договоры, доп.соглашения, оферты)
        "pdf_tables_reports"   — отчёты (о работах, по ТО, SLA, ...)
        "pdf_tables_other"     — письма, уведомления, выписки из ЕГРЮЛ
        "pdf_tables"           — есть таблица и текст, подтип не распознан
                                 (fallback для LLM-классификации на этапе 2)
    """
    import fitz
    from logger_utils import filter_mupdf_stderr

    doc = fitz.open(str(file_path))
    try:
        # Фильтруем только benign-сообщение MuPDF, прочий stderr сохраняется.
        with filter_mupdf_stderr():
            if doc.page_count == 0:
                return "pdf_scan"

            page = doc[0]
            page_text = page.get_text("text")
            letters = len(re.findall(r"[а-яА-ЯёЁa-zA-Z]", page_text))
            if letters < text_threshold:
                return "pdf_scan"

            tables = page.find_tables()
            has_tables = bool(tables and len(tables.tables) > 0)

            if not has_tables:
                return "pdf_text"

            sub = _classify_pdf_tables_content(page_text)
            return f"pdf_tables_{sub}" if sub else "pdf_tables"
    except Exception as e:
        logger.warning(f"Не удалось классифицировать PDF {file_path.name}: {e}")
        return "pdf_scan"
    finally:
        doc.close()


def is_scanned_pdf(file_path: Path, threshold: int = 100) -> bool:
    """Legacy wrapper — для обратной совместимости."""
    cat = classify_pdf(file_path, threshold)
    return cat == "pdf_scan"
