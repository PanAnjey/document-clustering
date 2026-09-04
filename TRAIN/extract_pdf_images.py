#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для извлечения изображений с первой страницы PDF файлов.

Исходная директория: D:\FileOrganizer\Sorted\PDF_Scan
Выходная директория: D:\FileOrganizer\TRAIN\TEST

Особенности:
- Параллельная многопроцессорная обработка
- Динамическое распределение между процессорами (все ядра минус 3)
- Очистка целевой директории перед обработкой
- Минимальный вывод в консоль (только прогресс и итоги)
"""

import os
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_first_page_to_png(args):
    """
    Извлекает изображение с первой страницы PDF и сохраняет как PNG.
    
    Args:
        args: tuple (pdf_path, output_dir)
        
    Returns:
        tuple: (success, filename) - результат обработки
    """
    pdf_path, output_dir = args
    
    try:
        # Импортируем библиотеки внутри функции для ленивой загрузки
        import fitz  # PyMuPDF
        
        # Открываем PDF документ
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            return False, os.path.basename(pdf_path)
        
        # Берем первую страницу
        page = doc[0]
        
        # Рендерим страницу в изображение с высоким разрешением (2x масштаб)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Сохраняем как PNG
        output_path = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}.png")
        pix.save(output_path)
        
        return True, os.path.basename(pdf_path)
        
    except Exception as e:
        return False, os.path.basename(pdf_path)
    finally:
        try:
            doc.close()
        except:
            pass


def clean_output_directory(output_dir):
    """
    Удаляет все файлы из выходной директории.
    
    Args:
        output_dir (str): Путь к директории
        
    Returns:
        int: Количество удаленных файлов
    """
    removed_count = 0
    
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_count += 1
            except Exception as e:
                pass
    
    return removed_count


def main():
    """Основная функция скрипта"""
    
    # Пути к директориям
    input_dir = r"D:\FileOrganizer\Sorted\PDF_Scan"
    output_dir = r"D:\FileOrganizer\TRAIN\TEST"
    
    print("=" * 80)
    print("🚀 Скрипт извлечения изображений с PDF файлов")
    print("=" * 80)
    print(f"\n📁 Входная директория: {input_dir}")
    print(f"📂 Выходная директория: {output_dir}")
    
    # Проверяем входную директорию
    if not os.path.exists(input_dir):
        print(f"\n❌ Директория не найдена: {input_dir}")
        return
    
    # Получаем количество доступных ядер
    total_cores = cpu_count()
    cores_to_use = max(1, total_cores - 3)  # Используем все ядра минус 3
    
    print(f"\n💻 Доступные процессоры: {total_cores}")
    print(f"🔧 Ядра для обработки: {cores_to_use} (все минус 3)")
    
    # Очищаем выходную директорию перед обработкой
    removed_count = clean_output_directory(output_dir)
    if removed_count > 0:
        print(f"\n🗑️  Удалено {removed_count} файлов из целевой директории")
    
    # Создаем выходную директорию если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех PDF файлов в входной директории
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n⚠️  В директории {input_dir} нет PDF файлов")
        return
    
    print(f"\n📄 Найдено {len(pdf_files)} PDF файлов для обработки\n")
    
    # Подготавливаем аргументы для каждого файла
    file_args = [(os.path.join(input_dir, pdf_file), output_dir) for pdf_file in sorted(pdf_files)]
    
    # Обрабатываем файлы параллельно с использованием ProcessPoolExecutor
    success_count = 0
    error_count = 0
    
    print(f"🔄 Запуск обработки на {cores_to_use} потоках...\n")
    
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        # Отправляем задачи в пул и собираем результаты
        futures = {executor.submit(extract_first_page_to_png, args): pdf_files[i] 
                   for i, args in enumerate(file_args)}
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            success, filename = future.result()
            
            if success:
                success_count += 1
            else:
                error_count += 1
            
            completed += 1
            # Показываем прогресс каждые 5 файлов или в конце
            if completed % 5 == 0 or completed == total:
                progress = (completed / total) * 100
                print(f"📊 Прогресс: {completed}/{total} ({progress:.1f}%)")
    
    # Вывод итогов
    print("\n" + "=" * 80)
    print("📊 ИТОГИ ОБРАБОТКИ")
    print("=" * 80)
    print(f"Всего файлов: {len(pdf_files)}")
    print(f"Успешно: {success_count}")
    print(f"Ошибки: {error_count}")
    print(f"Ядра использовано: {cores_to_use} из {total_cores}")
    print(f"\n📂 Изображения сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
