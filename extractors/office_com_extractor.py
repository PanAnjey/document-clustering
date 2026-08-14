# extractors/office_com_extractor.py
# COM automation для Word/Excel файлов

import multiprocessing
import threading
from pathlib import Path
from typing import Optional, Dict

try:
    import win32com.client
    import pythoncom
    _COM_AVAILABLE = True
except ImportError:
    _COM_AVAILABLE = False

from config import cfg
from logger_utils import logger, assess_text_quality

_check_done = multiprocessing.Event()
_check_lock = multiprocessing.Lock()
_word_available = None
_excel_available = None
_word_semaphore = multiprocessing.Semaphore(cfg.COM_WORD_WORKERS)
_excel_semaphore = multiprocessing.Semaphore(cfg.COM_EXCEL_WORKERS)
_zombie_killed: set = set()


def precheck_com() -> bool:
    global _word_available, _excel_available
    if _check_done.is_set():
        return _word_available or _excel_available
    with _check_lock:
        if _check_done.is_set():
            return _word_available or _excel_available
        if not _COM_AVAILABLE:
            _word_available = False
            _excel_available = False
            _check_done.set()
            return False
        try:
            pythoncom.CoInitialize()
            try:
                if _word_available is None:
                    try:
                        word = win32com.client.DispatchEx("Word.Application")
                        word.Quit()
                        _word_available = True
                        logger.info("Microsoft Word COM: available")
                    except Exception as e:
                        _word_available = False
                        logger.warning(f"Microsoft Word COM: unavailable ({e})")
                if _excel_available is None:
                    try:
                        excel = win32com.client.DispatchEx("Excel.Application")
                        excel.Quit()
                        _excel_available = True
                        logger.info("Microsoft Excel COM: available")
                    except Exception as e:
                        _excel_available = False
                        logger.warning(f"Microsoft Excel COM: unavailable ({e})")
            finally:
                pythoncom.CoUninitialize()
        except Exception as e:
            _word_available = False
            _excel_available = False
            logger.warning(f"COM precheck failed: {e}")
        _check_done.set()
        return _word_available or _excel_available


def _kill_zombie_office(app_type: str = None):
    """Kill orphaned Office processes."""
    import subprocess
    targets = []
    if app_type == 'word' or app_type is None:
        targets.append('WINWORD.EXE')
    if app_type == 'excel' or app_type is None:
        targets.append('EXCEL.EXE')
    for proc in targets:
        if proc in _zombie_killed:
            continue
        _zombie_killed.add(proc)
        try:
            subprocess.run(
                ['taskkill', '/f', '/im', proc],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except Exception:
            pass


def convert_office_to_pdf(input_path: Path, output_dir: Path, file_type: str) -> Optional[Path]:
    """Convert Office file to PDF via COM."""
    if not _COM_AVAILABLE:
        return None

    precheck_com()

    if file_type in ("word", "rtf", "txt", "odt"):
        if not _word_available:
            return None
        with _word_semaphore:
            return _convert_word(input_path, output_dir)
    elif file_type in ("excel", "csv", "ods"):
        if not _excel_available:
            return None
        with _excel_semaphore:
            return _convert_excel(input_path, output_dir)
    return None


def extract_word_com(file_path: Path) -> Dict:
    """Извлечение текста из Word файла через COM."""
    global _word_available

    result = {
        "source": str(file_path),
        "type": "word",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    if not _COM_AVAILABLE:
        result['error'] = "COM not available"
        return result

    precheck_com()
    if _word_available is False and _check_done.is_set():
        with _word_semaphore:
            text = _extract_word_text(file_path)
            if text:
                _word_available = True
                result['text'] = text
                result['text_quality'] = assess_text_quality(result['text'])
                return result
        result['error'] = "Word COM unavailable"
        return result

    with _word_semaphore:
        text = _extract_word_text(file_path)
        if text:
            result['text'] = text
            result['text_quality'] = assess_text_quality(result['text'])
        else:
            result['error'] = "Empty or extraction failed"

    return result


def _convert_word(input_path: Path, output_dir: Path) -> Optional[Path]:
    """Конвертация Word файла в PDF через COM."""
    try:
        file_size = input_path.stat().st_size
        if file_size < 100:
            logger.warning(f"File too small ({file_size} bytes), skipping: {input_path.name}")
            return None
    except Exception:
        pass
    
    pythoncom.CoInitialize()
    word = None
    doc = None
    timeout_occurred = False
    
    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        logger.error(f"Timeout converting file: {input_path.name}")
    
    timer = threading.Timer(30.0, timeout_handler)
    
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        word.ScreenUpdating = False
        word.AutomationSecurity = 3
        
        try:
            word.EnableEvents = False
        except Exception:
            pass
        
        try:
            word.AutoRecover.Enabled = False
        except Exception:
            pass

        timer.start()
        
        doc = word.Documents.Open(
            str(input_path), 
            ReadOnly=True, 
            AddToRecentFiles=False,
            ConfirmConversions=False,
            Revert=False,
            PasswordDocument="",
            WritePasswordDocument="",
            OpenAndRepair=False,
        )
        
        timer.cancel()
        
        if timeout_occurred:
            logger.error(f"Timeout occurred, skipping file: {input_path.name}")
            return None
        
        pdf_path = output_dir / f"{input_path.stem}.pdf"
        cnt = 1
        while pdf_path.exists():
            pdf_path = output_dir / f"{input_path.stem}_{cnt}.pdf"
            cnt += 1

        doc.ExportAsFixedFormat(
            OutputFileName=str(pdf_path),
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=False,
            KeepIRM=True,
            CreateBookmarks=0,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False
        )
        
        doc.Close(SaveChanges=False)
        doc = None

        if pdf_path.exists():
            return pdf_path
        return None
    except Exception as e:
        logger.error(f"Word COM error for {input_path.name}: {e}")
        return None
    finally:
        timer.cancel()
        try:
            if doc:
                doc.Close(SaveChanges=False)
        except Exception:
            pass
        try:
            if word:
                word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


def _convert_excel(input_path: Path, output_dir: Path) -> Optional[Path]:
    """Конвертация Excel файла в PDF через COM."""
    pythoncom.CoInitialize()
    excel = None
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.ScreenUpdating = False
        excel.AutomationSecurity = 3
        excel.AskToUpdateLinks = False
        
        try:
            excel.EnableEvents = False
        except Exception:
            pass
        
        try:
            excel.AutoRecover.Enabled = False
        except Exception:
            pass

        pdf_path = output_dir / f"{input_path.stem}.pdf"
        cnt = 1
        while pdf_path.exists():
            pdf_path = output_dir / f"{input_path.stem}_{cnt}.pdf"
            cnt += 1

        try:
            excel.Interactive = False
        except Exception:
            pass

        wb = excel.Workbooks.Open(
            str(input_path), 
            ReadOnly=True,
            UpdateLinks=False,
            AddToMru=False
        )

        excel.DisplayAlerts = False

        wb.ExportAsFixedFormat(
            Type=0,
            Filename=str(pdf_path),
            Quality=0,
            IncludeDocProperties=False,
            IgnorePrintAreas=False,
            OpenAfterPublish=False
        )
        
        wb.Close(SaveChanges=False)

        if pdf_path.exists():
            return pdf_path
        return None
    except Exception as e:
        logger.error(f"Excel COM error for {input_path.name}: {e}")
        return None
    finally:
        try:
            if excel:
                excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


def _extract_word_text(input_path: Path) -> Optional[str]:
    """Извлечение текста из Word файла через COM."""
    try:
        file_size = input_path.stat().st_size
        if file_size < 100:
            logger.warning(f"File too small ({file_size} bytes), skipping: {input_path.name}")
            return None
    except Exception:
        pass
    
    pythoncom.CoInitialize()
    word = None
    doc = None
    timeout_occurred = False
    
    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        logger.error(f"Timeout opening file: {input_path.name}")
    
    timer = threading.Timer(30.0, timeout_handler)
    
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        word.ScreenUpdating = False
        word.AutomationSecurity = 3
        
        try:
            word.EnableEvents = False
        except Exception:
            pass
        try:
            word.AutoRecover.Enabled = False
        except Exception:
            pass

        timer.start()
        
        doc = word.Documents.Open(
            str(input_path),
            ReadOnly=True,
            AddToRecentFiles=False,
            ConfirmConversions=False,
            Revert=False,
            PasswordDocument="",
            WritePasswordDocument="",
            OpenAndRepair=False,
        )
        
        timer.cancel()
        
        if timeout_occurred:
            logger.error(f"Timeout occurred, skipping file: {input_path.name}")
            return None

        text_parts = []
        total_pages = doc.ComputeStatistics(2)

        for page_num in range(1, total_pages + 1):
            try:
                word.Selection.GoTo(What=1, Which=1, Count=page_num)
                doc.Bookmarks(r"\Page").Select()
                rng = word.Selection.Range
            except Exception:
                continue

            try:
                if rng.Tables.Count > 0:
                    for table in rng.Tables:
                        try:
                            if table.Rows.Count == 1 and table.Rows(1).Cells.Count == 2:
                                continue
                            for row_idx in range(1, table.Rows.Count + 1):
                                try:
                                    row = table.Rows(row_idx)
                                    cells = []
                                    for cell_idx in range(1, row.Cells.Count + 1):
                                        try:
                                            cell = row.Cells(cell_idx)
                                            cell_text = cell.Range.Text.strip().rstrip('\r\a')
                                            cells.append(cell_text)
                                        except Exception:
                                            pass
                                    if cells:
                                        text_parts.append(" | ".join(cells))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                else:
                    raw = rng.Text
                    if raw:
                        raw = raw.strip().rstrip('\r\a')
                        text_parts.append(raw)
            except Exception:
                pass

        doc.Close(SaveChanges=False)
        doc = None

        full_text = "\n".join(text_parts)
        return full_text if full_text.strip() else None

    except Exception as e:
        logger.error(f"Word COM text extraction error for {input_path.name}: {e}")
        return None
    finally:
        timer.cancel()
        try:
            if doc:
                doc.Close(SaveChanges=False)
        except Exception:
            pass
        try:
            if word:
                word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


def _extract_excel_text(input_path: Path) -> Optional[str]:
    """Извлечение текста из Excel файла через COM."""
    pythoncom.CoInitialize()
    excel = None
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.ScreenUpdating = False
        excel.AutomationSecurity = 3
        excel.AskToUpdateLinks = False
        
        try:
            excel.EnableEvents = False
        except Exception:
            pass
        try:
            excel.AutoRecover.Enabled = False
        except Exception:
            pass
        try:
            excel.Interactive = False
        except Exception:
            pass
        try:
            excel.Calculation = -4135  # xlCalculationManual
        except Exception:
            pass

        wb = excel.Workbooks.Open(
            str(input_path),
            ReadOnly=True,
            UpdateLinks=False,
            AddToMru=False,
        )

        MAX_ROWS = 5000
        MAX_COLS = 500
        best_text = None
        best_len = 0

        for ws in wb.Worksheets:
            try:
                ws.Activate()
                _ = ws.UsedRange.Rows.Count
                used = ws.UsedRange

                nrows = min(used.Rows.Count, MAX_ROWS)
                ncols = min(used.Columns.Count, MAX_COLS)
                if nrows == 1 and ncols == 1 and not used.Cells(1, 1).Text:
                    continue

                rows_text = []
                for row in range(1, nrows + 1):
                    cells = []
                    for col in range(1, ncols + 1):
                        val = used.Cells(row, col).Text
                        if val:
                            cells.append(str(val).strip())
                    if cells:
                        rows_text.append(" | ".join(cells))
                        if sum(len(r) for r in rows_text) > 50000:
                            break
                    else:
                        empty_streak = sum(1 for r in rows_text[-5:] if not r.strip())
                        if empty_streak >= 5 and len(rows_text) > 10:
                            break

                if rows_text:
                    full = "\n".join(rows_text)
                    if len(full) > best_len:
                        best_text = full
                        best_len = len(full)
                        if best_len >= 50000:
                            break
            except Exception:
                continue

        wb.Close(SaveChanges=False)

        return best_text

    except Exception as e:
        logger.error(f"Excel COM text extraction error for {input_path.name}: {e}")
        return None
    finally:
        try:
            if excel:
                excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()
