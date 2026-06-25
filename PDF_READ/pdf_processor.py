import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
import time
from datetime import timedelta

# --- 1. Конфигурация ---
CONFIG = {
    "PDF_DIR": r"D:\FileOrganizer\Sorted\PDF",
    
    # Настройки pymupdf4llm
    "PYMUPDF4LLM_ENABLED": True,
    "TESSERACT_PATH": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "TESSERACT_LANG": "rus+eng",
    "PYMUPDF4LLM_OCR_ENABLED": True,
    "PYMUPDF4LLM_HEADER_FOOTER": False,
    "PYMUPDF4LLM_TABLE_STRATEGY": "lines_strict",

    # PostgreSQL
    "PG_HOST": "localhost",
    "PG_PORT": 5432,
    "PG_DB": "file_organizer_db",
    "PG_USER": "postgres",
    "PG_PASSWORD": "postgres",

    # Настройки пакетной вставки (Оптимизировано под 64 ГБ RAM)
    "BATCH_SIZE": 1000,
}

# --- 2. Обработка одного файла ---
def process_pdf_file(file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает текст из PDF и возвращает словарь с результатами.
    Работает в отдельном процессе.
    """
    result = {
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1].lower(),
        "text": "",
        "error": None
    }

    try:
        if config["PYMUPDF4LLM_ENABLED"]:
            ocr_kwargs = {
                "tesseract_path": config["TESSERACT_PATH"],
                "lang": config["TESSERACT_LANG"],
                "header_footer": config["PYMUPDF4LLM_HEADER_FOOTER"],
                "table_strategy": config["PYMUPDF4LLM_TABLE_STRATEGY"]
            }

            if config["PYMUPDF4LLM_OCR_ENABLED"]:
                ocr_kwargs["ocr"] = True
            
            from pymupdf4llm import to_markdown
            text_content = to_markdown(file_path, **ocr_kwargs)
            
            # Ограничение длины текста до 5000 символов
            result["text"] = text_content[:5000] if text_content else ""
        else:
            result["text"] = ""

    except Exception as e:
        # Запись ошибки в поле error
        result["error"] = str(e)
    
    return result

# --- 3. Пакетная вставка в БД ---
def save_batch_to_database(batch: List[Dict[str, Any]], cursor):
    """
    Выполняет пакетную вставку данных в таблицу documents_2.
    """
    if not batch:
        return

    # Формируем список кортежей для execute_batch
    insert_data = [
        (r["file_path"], r["file_type"], r["text"], r["error"])
        for r in batch
    ]

    sql = """
        INSERT INTO "public"."documents_2" 
        ("file_path", "file_type", "text", "error") 
        VALUES (%s, %s, %s, %s)
    """
    
    try:
        execute_batch(cursor, sql, insert_data)
    except Exception as e:
        print(f"[DB ERROR] Ошибка пакетной вставки: {e}")

# --- 4. Основной процесс ---
def main():
    # Импортируем to_markdown здесь для основного процесса (опционально)
    try:
        from pymupdf4llm import to_markdown
    except ImportError:
        print("Ошибка: Библиотека pymupdf4llm не найдена. Установите её.")
        return

    # Получаем список PDF файлов
    pdf_pattern = os.path.join(CONFIG["PDF_DIR"], "*.pdf")
    files_to_process = glob.glob(pdf_pattern)

    if not files_to_process:
        print("Файлы PDF не найдены в указанной директории.")
        return

    total_files = len(files_to_process)
    start_time = time.time()  # Время начала обработки
    
    print(f"Найдено {total_files} файлов для обработки...")
    print(f"RAM доступно: 64 ГБ. Используем BATCH_SIZE = {CONFIG['BATCH_SIZE']}")

    # Подключение к БД ОДИН РАЗ на весь процесс
    conn = psycopg2.connect(
        host=CONFIG["PG_HOST"],
        port=CONFIG["PG_PORT"],
        dbname=CONFIG["PG_DB"],
        user=CONFIG["PG_USER"],
        password=CONFIG["PG_PASSWORD"]
    )
    cursor = conn.cursor()

    # Буфер для накопления результатов перед вставкой
    batch = []
    processed_count = 0
    
    # Запуск пула процессов
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(process_pdf_file, f, CONFIG): f for f in files_to_process}

        # Сбор результатов с прогресс-баром и ETA
        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Обработка файлов"):
            file_path = future_to_file[future]
            try:
                res = future.result()
                batch.append(res)
                processed_count += 1

                # Если накопилось достаточно записей - отправляем в БД и делаем commit
                if len(batch) >= CONFIG["BATCH_SIZE"]:
                    save_batch_to_database(batch, cursor)
                    conn.commit()
                    
                    elapsed = time.time() - start_time
                    avg_speed = processed_count / elapsed if elapsed > 0 else 0
                    remaining_files = total_files - processed_count
                    eta_seconds = remaining_files / avg_speed if avg_speed > 0 else 0
                    
                    print(f"[DB] Пакет сохранён ({len(batch)} записей). "
                          f"Обработано: {processed_count}/{total_files} | "
                          f"Скорость: {avg_speed:.1f} файл/сек | "
                          f"ETA: {timedelta(seconds=int(eta_seconds))}")
                    
                    batch = []  # Очищаем буфер

            except Exception as e:
                # Ошибка самого процесса (редко, но бывает)
                print(f"Ошибка воркера для файла {file_path}: {e}")

    # Финальная вставка оставшихся записей (если их меньше чем BATCH_SIZE)
    if batch:
        save_batch_to_database(batch, cursor)
        conn.commit()
        print(f"[DB] Оставшиеся {len(batch)} записей сохранены.")

    # Закрываем соединение и курсор
    cursor.close()
    conn.close()

    # Итоговая статистика
    success_count = sum(1 for r in batch + [r for f in files_to_process if (lambda x: x["error"] is None)(x) for x in []])
    error_count = total_files - success_count
    
    elapsed_total = time.time() - start_time
    avg_speed_total = total_files / elapsed_total if elapsed_total > 0 else 0

    print(f"\n{'='*60}")
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"Всего файлов: {total_files}")
    print(f"Успешно: {success_count}")
    print(f"С ошибками: {error_count}")
    print(f"Общее время: {timedelta(seconds=int(elapsed_total))}")
    print(f"Средняя скорость: {avg_speed_total:.2f} файла/сек")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
