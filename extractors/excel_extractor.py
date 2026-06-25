# extractors/excel_extractor.py
# Извлечение текста из Excel файлов через COM

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


def extract_excel(file_path: Path) -> Dict:
    """Извлечение текста из Excel файла через COM."""
    result = {
        "source": str(file_path),
        "type": "excel",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    if not _COM_AVAILABLE:
        result['error'] = "COM not available"
        return result

    try:
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
                str(file_path),
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

            if best_text and best_text.strip():
                result['text'] = best_text.strip()
                result['text_quality'] = assess_text_quality(result['text'])
            else:
                result['error'] = "Empty or extraction failed"

        finally:
            try:
                if excel:
                    excel.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()

    except Exception as e:
        result['error'] = f"Excel COM extraction error: {e}"

    return result
