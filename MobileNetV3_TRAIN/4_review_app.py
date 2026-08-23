"""Приложение проверки результатов тестового прогона.

Одновременно показывает 2 изображения:
  слева  — исходное из INFERENCE_INPUT_DIR (PDF_Images)
  справа — оно же, восстановленное в правильной ориентации, из CORRECTED_DIR

Между изображениями — информация от модели: ориентация, степень уверенности и угол
поворота, на который было повернуто правое изображение.
Над изображениями — полный путь к соответствующей директории.
Под изображениями — имя файла.

Отображаются ТОЛЬКО пары, которые были скорректированы (orientation != 0).
Навигация: кнопки «< Назад» и «Вперёд >», колесо мыши (вверх — назад, вниз — вперёд),
клавиши ←/→ и A/D, а также переход к следующей паре после выбора радиобатона.

Под исходным изображением — 4 радиобатона (0, 90, 180, 270), означающие ИСТИННУЮ
ориентацию исходного изображения по мнению оператора (т.е. ошибку модели).
Выбранные значения сохраняются в REVIEW_REPORT_FILE (CSV). Исходное изображение
перемещается в RETRAIN_DIR/{angle} (с заменой при коллизии), а парное исправленное
изображение удаляется из CORRECTED_DIR. После выбора — автоматический переход к
следующей паре (с сохранением текущей порядковой позиции, а не возврат к началу).

Справа — панель списка файлов: графически выделяет текущую отображаемую пару,
позволяет произвольную навигацию кликом по любой записи, обновляется после
перемещения файлов (перемещённая пара исчезает из списка).

Все параметры берутся из config.py. Командная строка НЕ используется.
"""
import bisect
import csv
import json
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # отключить DecompressionBombError для больших сканов

from PIL import ImageTk

import config as C

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


class ReviewApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Проверка ориентации — MobileNetV3")
        root.geometry("1900x900")

        # Данные
        self.input_dir = C.INFERENCE_INPUT_DIR
        self.corrected_dir = C.CORRECTED_DIR
        self.retrain_dir = C.RETRAIN_DIR
        self.report_file = C.REVIEW_REPORT_FILE
        for ang in C.RETRAIN_CLASSES:
            (self.retrain_dir / str(ang)).mkdir(parents=True, exist_ok=True)
        self.retrain_dir.mkdir(parents=True, exist_ok=True)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

        self.entries = self._load_entries()
        self.index = 0

        self.radio_var = tk.IntVar(value=-1)   # -1 = ничего не выбрано
        self.tk_left = None
        self.tk_right = None
        self._radio_guard = False
        self._tree_guard = False
        self._copy_target = None

        self._build_ui()
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("d", lambda e: self.next())
        self.root.bind("a", lambda e: self.prev())
        self.root.bind("<MouseWheel>", self._on_wheel)

        try:
            self._show_current()
        except Exception as e:
            import traceback
            print("[ERROR] при показе первой пары:", e)
            traceback.print_exc()

        # форсировать показ окна поверх
        root.update_idletasks()
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.after(500, lambda: root.attributes("-topmost", False))
        root.mainloop()

    # ---------- данные ----------
    def _load_entries(self):
        if not C.INFERENCE_RESULTS_FILE.exists():
            print(f"[ERROR] Файл результатов не найден: {C.INFERENCE_RESULTS_FILE}. "
                  f"Сначала запустите 3_test_inference.py", file=sys.stderr)
            sys.exit(1)
        with open(C.INFERENCE_RESULTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        res = [r for r in data.get("results", []) if r.get("corrected") and not r.get("error")]
        print(f"Загружено скорректированных пар: {len(res)}")
        return res

    def _valid_entries(self):
        """Индексы записей, у которых файлы ещё существуют."""
        valid = []
        for i, e in enumerate(self.entries):
            if (self.input_dir / e["file"]).exists() and (self.corrected_dir / e["file"]).exists():
                valid.append(i)
        return valid

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        self.counter_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.counter_var, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="< Назад", command=self.prev).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Вперёд >", command=self.next).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status_var, foreground="#444").pack(side=tk.LEFT, padx=10)
        ttk.Label(top, text="Колесо мыши/←→/A,D — навигация", foreground="#888",
                  font=("Segoe UI", 8)).pack(side=tk.RIGHT)

        # Панель с тремя колонками: путь слева | инфо | путь справа
        paths = ttk.Frame(self.root, padding=(6, 2))
        paths.pack(side=tk.TOP, fill=tk.X)
        self.dir_left_var = tk.StringVar(value=str(self.input_dir))
        self.dir_right_var = tk.StringVar(value=str(self.corrected_dir))
        ttk.Label(paths, textvariable=self.dir_left_var, foreground="#0066cc",
                  font=("Consolas", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")
        ttk.Label(paths, text="МОДЕЛЬ", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=8)
        ttk.Label(paths, textvariable=self.dir_right_var, foreground="#0066cc",
                  font=("Consolas", 9)).pack(side=tk.RIGHT, fill=tk.X, expand=True, anchor="e")

        # Контент: изображения слева + список файлов справа
        content = ttk.Frame(self.root)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        mid = ttk.Frame(content)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_columnconfigure(2, weight=1)
        mid.grid_rowconfigure(0, weight=1)

        self.lbl_left = ttk.Label(mid, text="", anchor="center", background="#eee")
        self.lbl_left.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        info = ttk.Frame(mid, padding=8)
        info.grid(row=0, column=1, sticky="ns")
        self.orient_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="-")
        self.ang_var = tk.StringVar(value="-")
        self.deskew_var = tk.StringVar(value="-")
        self.crop_var = tk.StringVar(value="-")
        ttk.Label(info, text="ОРИЕНТАЦИЯ", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.orient_var, font=("Segoe UI", 18, "bold"),
                  foreground="#222").pack(anchor="w", pady=(0, 6))
        ttk.Label(info, text="УВЕРЕННОСТЬ", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.conf_var, font=("Segoe UI", 14),
                  foreground="#222").pack(anchor="w", pady=(0, 6))
        ttk.Label(info, text="УГОЛ ПОВОРОТА", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.ang_var, font=("Segoe UI", 14, "bold"),
                  foreground="#0a7").pack(anchor="w", pady=(0, 6))
        ttk.Label(info, text="ПЕРЕКОС (deskew)", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.deskew_var, font=("Segoe UI", 12),
                  foreground="#a33").pack(anchor="w", pady=(0, 6))
        ttk.Label(info, text="ОБРЕЗКА КРАЁВ", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.crop_var, font=("Segoe UI", 12),
                  foreground="#555").pack(anchor="w")

        self.lbl_right = ttk.Label(mid, text="", anchor="center", background="#eee")
        self.lbl_right.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

        # Имена файлов (Entry в readonly — можно выделять и копировать Ctrl+C)
        names = ttk.Frame(self.root, padding=(6, 2))
        names.pack(side=tk.TOP, fill=tk.X)
        self.name_left_var = tk.StringVar(value="-")
        self.name_right_var = tk.StringVar(value="-")
        self.ent_left = ttk.Entry(names, textvariable=self.name_left_var, font=("Consolas", 9),
                                  state="readonly", takefocus=False)
        self.ent_left.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")
        self.ent_right = ttk.Entry(names, textvariable=self.name_right_var, font=("Consolas", 9),
                                   state="readonly", takefocus=False)
        self.ent_right.pack(side=tk.RIGHT, fill=tk.X, expand=True, anchor="e")

        # Контекстное меню «Копировать» по правой кнопке мыши
        self.ctx_menu = tk.Menu(self.root, tearoff=0)
        self.ctx_menu.add_command(label="Копировать имя файла", command=self._copy_active_name)
        for ent in (self.ent_left, self.ent_right):
            ent.bind("<Button-3>", self._on_entry_right_click)
            ent.bind("<Control-c>", lambda e: "break")

        # Панель списка файлов (справа, внутри content)
        self._build_list_panel(content)

        # Радиобатоны под исходным изображением
        self._build_radio_buttons()

    # ---------- панель списка файлов ----------
    def _build_list_panel(self, parent):
        panel = ttk.Frame(parent, padding=(6, 0))
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(panel, text="Список файлов", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 2))

        tree_frame = ttk.Frame(panel)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.file_tree = ttk.Treeview(tree_frame,
                                     columns=("ord", "orient", "conf"),
                                     show="tree headings",
                                     selectmode="browse", height=26)
        self.file_tree.heading("#0", text="Имя файла")
        self.file_tree.heading("ord", text="#")
        self.file_tree.heading("orient", text="Ор")
        self.file_tree.heading("conf", text="Увер")
        self.file_tree.column("#0", width=360, anchor="w")
        self.file_tree.column("ord", width=36, anchor="center", stretch=False)
        self.file_tree.column("orient", width=40, anchor="center", stretch=False)
        self.file_tree.column("conf", width=56, anchor="center", stretch=False)
        self.file_tree.tag_configure("current", background="#ffe08a")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=vsb.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.bind("<<TreeviewSelect>>", self._on_list_select)
        self._refresh_list()

    def _refresh_list(self):
        """Перестроить список, оставив только валидные (существующие) пары.
        Подсветить текущую запись и прокрутить к ней."""
        if not hasattr(self, "file_tree"):
            return
        self._tree_guard = True
        try:
            for iid in self.file_tree.get_children():
                self.file_tree.delete(iid)
            valid = self._valid_entries()
            for n, i in enumerate(valid, 1):
                e = self.entries[i]
                self.file_tree.insert("", "end", iid=str(i),
                                      text=e["file"],
                                      values=(n, e.get("orientation", ""),
                                              f"{e.get('confidence', 0):.2f}"))
            if str(self.index) in self.file_tree.get_children():
                self.file_tree.selection_set(str(self.index))
                self.file_tree.item(str(self.index), tags=("current",))
                self.file_tree.see(str(self.index))
        except tk.TclError:
            pass
        finally:
            self.root.after(0, self._release_tree_guard)

    def _highlight_current(self):
        """Лёгкое обновление подсветки текущей записи без перестроения."""
        if not hasattr(self, "file_tree"):
            return
        self._tree_guard = True
        try:
            for iid in self.file_tree.get_children():
                self.file_tree.item(iid, tags=())
            if str(self.index) in self.file_tree.get_children():
                self.file_tree.selection_set(str(self.index))
                self.file_tree.item(str(self.index), tags=("current",))
                self.file_tree.see(str(self.index))
        except tk.TclError:
            pass
        finally:
            # держим guard до следующей прокачки очереди событий, чтобы
            # TreeviewSelect (поставился в очередь от selection_set) игнорировался
            self.root.after(0, self._release_tree_guard)

    def _release_tree_guard(self):
        self._tree_guard = False

    def _on_list_select(self, event):
        if self._tree_guard:
            return
        sel = self.file_tree.selection()
        if not sel:
            return
        self.index = int(sel[0])
        self._show_current(show_refresh=False)

    # ---------- радиобатоны ----------
    def _build_radio_buttons(self):
        rb = ttk.LabelFrame(self.root, text="Истинная ориентация исходника (отметить ошибку модели)",
                            padding=8)
        rb.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        rb_inner = ttk.Frame(rb)
        rb_inner.pack(anchor="w")
        for ang in C.CLASS_NAMES:
            ttk.Radiobutton(rb_inner, text=str(ang), value=ang, variable=self.radio_var,
                            command=self._on_radio).pack(side=tk.LEFT, padx=20)
        ttk.Label(rb, text="Назначение: 0/90/180/270 = истинный угол -> перенос исходника в "
                            "RETRAIN/{0,90,180,270}, исправленное удаляется -> переход к следующей паре.",
                  foreground="#666", wraplength=950).pack(anchor="w", pady=(4, 0))
        ttk.Label(rb, text="Подсказка: ← / →, A / D или колесо мыши для навигации.",
                  foreground="#999").pack(anchor="w")

    # ---------- копирование имени ----------
    def _on_entry_right_click(self, event):
        ent = event.widget
        ent.focus_set()
        ent.selection_range(0, "end")
        self._copy_target = ent
        self.ctx_menu.tk_popup(event.x_root, event.y_root)

    def _copy_active_name(self):
        ent = self._copy_target
        if ent is None:
            return
        try:
            val = ent.get()
        except Exception:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(val)
        self.status_var.set(f"Имя скопировано в буфер: {val}")

    # ---------- изображения ----------
    def _load_preview(self, path: Path):
        try:
            im = Image.open(path)
            im.load()
        except Exception as e:
            print(f"[warn] не открыть {path}: {e}")
            return None
        try:
            im.thumbnail((C.PREVIEW_MAX_WIDTH, C.PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        except Exception as e:
            print(f"[warn] не удалось превью {path}: {e}")
            return None
        return ImageTk.PhotoImage(im)

    # ---------- навигация ----------
    def _show_current(self, show_refresh=True):
        self._radio_guard = True
        self.radio_var.set(-1)
        self._radio_guard = False

        valid = self._valid_entries()
        if not valid:
            self._clear_display()
            self.counter_var.set("Нет пар для проверки")
            self.status_var.set("Все пары обработаны или отсутствуют.")
            if show_refresh:
                self._refresh_list()
            return

        if self.index not in valid:
            # текущая запись была удалена — перейти к ближайшей следующей
            pos = bisect.bisect_right(valid, self.index)
            if pos < len(valid):
                self.index = valid[pos]
            else:
                self.index = valid[-1]

        pos = valid.index(self.index) + 1
        entry = self.entries[self.index]
        fname = entry["file"]
        left_path = self.input_dir / fname
        right_path = self.corrected_dir / fname

        self.dir_left_var.set(str(self.input_dir))
        self.dir_right_var.set(str(self.corrected_dir))
        self.name_left_var.set(fname)
        self.name_right_var.set(fname)

        self.orient_var.set(f"{entry.get('orientation', '?')}°")
        self.conf_var.set(f"{entry.get('confidence', 0):.3f}")
        self.ang_var.set(f"{entry.get('correction_angle', '?')}°")
        dk = entry.get("deskew_angle", 0.0)
        self.deskew_var.set(f"{dk:+.2f}°" if dk else "нет")
        cr = entry.get("borders_cropped", False)
        self.crop_var.set("да" if cr else "нет")

        self.counter_var.set(f"{pos} / {len(valid)}")
        self.status_var.set("")

        limg = self._load_preview(left_path)
        if limg is not None:
            self.tk_left = limg
            self.lbl_left.config(image=limg, text="")
        else:
            self.lbl_left.config(image="", text="нет изображения")
        rimg = self._load_preview(right_path)
        if rimg is not None:
            self.tk_right = rimg
            self.lbl_right.config(image=rimg, text="")
        else:
            self.lbl_right.config(image="", text="нет исправленного")

        if show_refresh:
            self._refresh_list()
        else:
            self._highlight_current()

    def _clear_display(self):
        self.lbl_left.config(image="", text="")
        self.lbl_right.config(image="", text="")
        self.name_left_var.set("-")
        self.name_right_var.set("-")
        self.orient_var.set("-")
        self.conf_var.set("-")
        self.ang_var.set("-")
        self.deskew_var.set("-")
        self.crop_var.set("-")

    def _on_wheel(self, event):
        if event.delta > 0:
            self.prev()
        elif event.delta < 0:
            self.next()

    def next(self):
        valid = self._valid_entries()
        if not valid:
            self._clear_display()
            self._refresh_list()
            return
        if self.index in valid:
            pos = valid.index(self.index)
            if pos + 1 < len(valid):
                self.index = valid[pos + 1]
                self._show_current()
            else:
                self.status_var.set("Достигнут конец списка.")
        else:
            # текущая запись была удалена — остаёмся на той же порядковой позиции
            pos = bisect.bisect_right(valid, self.index)
            if pos < len(valid):
                self.index = valid[pos]
                self._show_current()
            elif valid:
                self.index = valid[-1]
                self._show_current()
            else:
                self._clear_display()

    def prev(self):
        valid = self._valid_entries()
        if not valid:
            self._clear_display()
            self._refresh_list()
            return
        if self.index in valid:
            pos = valid.index(self.index)
            if pos - 1 >= 0:
                self.index = valid[pos - 1]
                self._show_current()
            else:
                self.status_var.set("Начало списка.")
        else:
            pos = max(0, bisect.bisect_left(valid, self.index) - 1)
            if valid:
                self.index = valid[pos]
                self._show_current()
            else:
                self._clear_display()

    # ---------- радиобатон ----------
    def _on_radio(self):
        if self._radio_guard:
            return
        val = self.radio_var.get()
        if val == -1:
            return
        if not self._valid_entries():
            return
        entry = self.entries[self.index]
        fname = entry["file"]
        src = self.input_dir / fname
        corr = self.corrected_dir / fname

        write_header = not self.report_file.exists() or self.report_file.stat().st_size == 0
        with open(self.report_file, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(["file", "true_orientation", "model_orientation",
                            "confidence", "correction_angle", "input_dir", "retrain_dir"])
            w.writerow([fname, val, entry.get("orientation", ""),
                        f"{entry.get('confidence', 0):.4f}",
                        entry.get("correction_angle", ""),
                        str(self.input_dir), str(self.retrain_dir)])

        if src.exists():
            dst = self.retrain_dir / str(val) / fname
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            try:
                shutil.move(str(src), str(dst))
            except Exception as e:
                print(f"[warn] не удалось переместить {src}: {e}")

        if corr.exists():
            try:
                corr.unlink()
            except OSError as e:
                print(f"[warn] не удалось удалить {corr}: {e}")

        self.status_var.set(f"Сохранено: true={val}° -> {self.retrain_dir / str(val) / fname}  "
                            f"; исправленное удалено.")
        # перестроить список (перемещённая пара исчезнет) и остаться на той же позиции
        self._refresh_list()
        self.next()


def main():
    root = tk.Tk()
    ReviewApp(root)


if __name__ == "__main__":
    main()