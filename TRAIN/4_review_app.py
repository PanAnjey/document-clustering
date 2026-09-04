"""Приложение для проверки результатов инференса и сбора данных для дообучения (retrain).

- Загружает результаты из INFERENCE_RESULTS_FILE.
- Показывает превью изображений с предсказанной ориентацией.
- Позволяет вручную исправить метки для ошибок.
- Сохраняет исправленные изображения в RETRAIN_DIR/{0,90,180,270}.
- Генерирует review_report.csv со статистикой ошибок.

Используется после 3_test_inference.py для сбора данных на переобучение модели.
"""
import argparse
import csv
import json
from pathlib import Path

import config as C
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox


class ReviewApp:
    """Приложение для ручной проверки и исправления предсказаний."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"Review App - Vision Transformer Orientation")
        
        # Загрузка результатов
        if not C.INFERENCE_RESULTS_FILE.exists():
            messagebox.showerror("Ошибка", f"Файл результатов не найден: {C.INFERENCE_RESULTS_FILE}")
            self.root.destroy()
            return
        
        with open(C.INFERENCE_RESULTS_FILE, "r", encoding="utf-8") as fh:
            self.results = json.load(fh)
        
        # Фильтрация для отображения (только ошибки или low confidence)
        self.filtered_results = [
            r for r in self.results.get("results", [])
            if r.get("low_confidence") or r.get("error") or not r.get("corrected")
        ]
        
        self.current_idx = 0
        self.retrain_dir = C.RETRAIN_DIR
        
        # Создание UI
        self._create_ui()
        
        print(f"Загружено {len(self.filtered_results)} элементов для проверки")

    def _create_ui(self):
        """Создаёт интерфейс приложения."""
        
        # Верхняя панель со статистикой
        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(stats_frame, text=f"Всего для проверки: {len(self.filtered_results)}").pack(side="left")
        ttk.Label(stats_frame, text=f"Текущий: {self.current_idx + 1}/{len(self.filtered_results)}").pack(side="right")
        
        # Основная область с изображением
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Левая панель - изображение
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True)
        
        self.image_label = ttk.Label(left_panel)
        self.image_label.pack(expand=True)
        
        # Правая панель - информация и управление
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        
        # Информация о предсказании
        info_frame = ttk.LabelFrame(right_panel, text="Предсказание")
        info_frame.pack(fill="x", pady=5)
        
        self.predicted_label = ttk.Label(info_frame, text="")
        self.predicted_label.pack(anchor="w", padx=5, pady=2)
        
        self.confidence_label = ttk.Label(info_frame, text="")
        self.confidence_label.pack(anchor="w", padx=5, pady=2)
        
        # Кнопки управления
        btn_frame = ttk.Frame(right_panel)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="< Пред.", command=self.prev_item).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="След. >", command=self.next_item).pack(side="right", expand=True, fill="x", padx=2)
        
        # Кнопки исправления
        correct_frame = ttk.LabelFrame(right_panel, text="Исправить")
        correct_frame.pack(fill="x", pady=5)
        
        for angle in C.CLASS_NAMES[:4]:  # 0, 90, 180, 270
            btn = ttk.Button(correct_frame, text=f"Правильно: {angle}°", 
                           command=lambda a=angle: self.correct(a))
            btn.pack(fill="x", padx=5, pady=2)
        
        # Кнопка "Не-скан"
        ttk.Button(correct_frame, text="Не-скан (other)", 
                  command=lambda: self.correct("other")).pack(fill="x", padx=5, pady=2)
        
        # Статус бар
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", side="bottom")
        
        self.status_label = ttk.Label(status_bar, text="Готово")
        self.status_label.pack(side="left")

    def _load_image(self):
        """Загружает и отображает текущее изображение."""
        if not self.filtered_results:
            return
        
        result = self.filtered_results[self.current_idx]
        
        # Определяем путь к изображению
        file_name = result.get("file")
        if not file_name:
            return
        
        # Пытаемся найти файл в разных директориях
        possible_paths = [
            C.INFERENCE_INPUT_DIR / file_name,
            Path(C.CORRECTED_DIR) / file_name,
            Path(C.NOT_CORRECTED_DIR) / file_name,
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if not img_path:
            self.status_label.config(text=f"Файл не найден: {file_name}")
            return
        
        # Загрузка изображения
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                
                # Масштабирование для превью
                max_size = (C.PREVIEW_MAX_WIDTH, C.PREVIEW_MAX_HEIGHT)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                self.photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo)
        except Exception as e:
            self.status_label.config(text=f"Ошибка загрузки: {e}")

    def _update_info(self):
        """Обновляет информацию о текущем элементе."""
        if not self.filtered_results:
            return
        
        result = self.filtered_results[self.current_idx]
        
        pred_orient = result.get("predicted_orientation", "N/A")
        confidence = result.get("confidence", 0)
        
        self.predicted_label.config(text=f"Предсказано: {pred_orient}°")
        self.confidence_label.config(
            text=f"Уверенность: {confidence:.2%}",
            foreground="green" if confidence > C.INFERENCE_CONFIDENCE_THRESHOLD else "red"
        )

    def prev_item(self):
        """Переход к предыдущему элементу."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_image()
            self._update_info()
            self.status_label.config(text=f"Элемент {self.current_idx + 1}/{len(self.filtered_results)}")

    def next_item(self):
        """Переход к следующему элементу."""
        if self.current_idx < len(self.filtered_results) - 1:
            self.current_idx += 1
            self._load_image()
            self._update_info()
            self.status_label.config(text=f"Элемент {self.current_idx + 1}/{len(self.filtered_results)}")

    def correct(self, angle):
        """Сохраняет изображение в RETRAIN_DIR с правильной меткой."""
        if not self.filtered_results:
            return
        
        result = self.filtered_results[self.current_idx]
        file_name = result.get("file")
        
        # Определяем путь к изображению
        possible_paths = [
            C.INFERENCE_INPUT_DIR / file_name,
            Path(C.CORRECTED_DIR) / file_name,
            Path(C.NOT_CORRECTED_DIR) / file_name,
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if not img_path:
            messagebox.showwarning("Внимание", f"Файл не найден: {file_name}")
            return
        
        # Создаём целевую директорию
        target_dir = self.retrain_dir / str(angle)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем файл
        try:
            import shutil
            shutil.copy(img_path, target_dir / file_name)
            
            # Обновляем статус
            self.status_label.config(text=f"Сохранено в {target_dir}")
            
            # Переходим к следующему элементу
            if self.current_idx < len(self.filtered_results) - 1:
                self.next_item()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")

    def run(self):
        """Запускает приложение."""
        self._load_image()
        self._update_info()
        
        # Привязка клавиш
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())
        
        print("Приложение запущено. Нажмите F5 для обновления.")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Review app для Vision Transformer")
    parser.add_argument("--skip-low-conf", action="store_true",
                        help="Показывать только low confidence предсказания")
    args = parser.parse_args()

    if args.skip_low_conf:
        # Фильтруем только low confidence
        with open(C.INFERENCE_RESULTS_FILE, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        
        C.RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
        for angle in C.CLASS_NAMES[:4]:
            (C.RETRAIN_DIR / str(angle)).mkdir(parents=True, exist_ok=True)
    
    app = ReviewApp()
    app.run()


if __name__ == "__main__":
    main()
