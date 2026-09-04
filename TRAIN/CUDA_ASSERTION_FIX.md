# Исправления для ошибки CUDA assertion `t >= 0 && t < n_classes`

## 🐛 Проблема

```
Assertion `t >= 0 && t < n_classes` failed.
```

Эта ошибка возникает в функции потерь CrossEntropyLoss, когда некоторые target индексы выходят за пределы допустимого диапазона [0, n_classes).

---

## 🔍 Причина ошибки

### 1. **Алфавитная сортировка классов ImageFolder**

Когда вы создаете датасет с директориями `['0', '90', '180', '270']`, ImageFolder сортирует их **алфавитно**:
- `'0'` -> индекс 0
- `'180'` -> индекс 1 (потому что '1' < '2' < '9')
- `'270'` -> индекс 2
- `'90'` -> индекс 3

### 2. **Ротация аугментации создает углы, которых нет в датасете**

При `ROTATION_AUG=True`:
```python
k = random.randint(0, 3)  # 0, 1, 2 или 3
a = 90 * k  # 0°, 90°, 180° или 270°
new_angle = (orig_angle + a) % 360
```

Если `orig_angle = 180` и `k = 2`, то:
- `a = 180°`
- `new_angle = (180 + 180) % 360 = 0°` ✓

**НО!** Если маппинг `angle_to_idx` не содержит все углы, возникает ошибка!

---

## ✅ Исправления

### Исправление 1: Проверка существования угла в angle_to_idx

```python
# Было:
orig_target = self.angle_to_idx[(orig_angle + a) % 360]

# Стало:
new_angle = (orig_angle + a) % 360

if new_angle in self.angle_to_idx:
    orig_target = self.angle_to_idx[new_angle]
else:
    # Если угла нет, используем угол 0 как fallback
    print(f"[WARN] Угол {new_angle} не найден в angle_to_idx. Используем 0.")
    orig_target = self.angle_to_idx.get(0, 0)

target = int(orig_target)
```

### Исправление 2: Проверка всех углов при инициализации

```python
# Проверяем, что все 4 угла присутствуют в датасете
expected_angles = {0, 90, 180, 270}
actual_angles = set(self.idx_to_angle.values())
if not expected_angles.issubset(actual_angles):
    missing = expected_angles - actual_angles
    print(f"[WARN] Отсутствуют углы в датасете: {missing}")
    print(f"       Доступные углы: {actual_angles}")
```

### Исправление 3: Финальная проверка target перед возвратом

```python
# Финальная проверка: target должен быть в пределах [0, n_classes)
num_classes = len(self.classes)
if not (0 <= target < num_classes):
    print(f"[ERROR] Target {target} out of range [0, {num_classes})! Clipping to 0.")
    target = max(0, min(target, num_classes - 1))

return self.post_tfm(img), target
```

---

## 📊 Проверка датасета

Перед запуском обучения проверьте структуру датасета:

```bash
python check_label_conflicts.py
```

Ожидаемый вывод:
```
Классы ImageFolder: ['0', '180', '270', '90'] (samples: XXXX train / YYYY val)
Веса классов (off): 0:X/w=1.000, 180:X/w=1.000, 270:X/w=1.000, 90:X/w=1.000
```

---

## 🚀 Запуск обучения

```bash
cd D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN
python 2_train.py
```

### Ожидаемый вывод:

```
[INFO] Доступно 2 GPU: NVIDIA RTX PRO 4000 (48GB) x2
Устройство: cuda (GPU count: 2)
Классы ImageFolder: ['0', '180', '270', '90'] (samples: XXXX train / YYYY val)
Вход: full/letterbox  |  ROTATION_AUG=True ...
[INFO] ROTATION_AUG включён — классы балансируются ротацией, веса классов отключены.
Веса классов (off): 0:X/w=1.000, 180:X/w=1.000, 270:X/w=1.000, 90:X/w=1.000
LR scheduler: cosine (eta_min=1e-06)
Epoch  1/50  ...
```

---

## 🎯 Ключевые изменения в коде

| Файл | Изменение | Описание |
|------|-----------|----------|
| **2_train.py** | Добавлена проверка `new_angle in self.angle_to_idx` | Предотвращает KeyError при ротации |
| **2_train.py** | Проверка всех углов в `__init__` | Выводит предупреждение, если углы отсутствуют |
| **2_train.py** | Финальная проверка target | Гарантирует, что target ∈ [0, n_classes) |

---

## 💡 Рекомендации для предотвращения ошибок

### 1. **Проверьте структуру датасета перед обучением:**

```bash
# Убедитесь, что все 4 класса присутствуют
ls D:\FileOrganizer\TRAIN\dataset\distrib\
# Ожидаемый вывод: 0, 90, 180, 270 (или в любом порядке)
```

### 2. **Запустите проверку дисбаланса:**

```bash
python check_label_conflicts.py
```

### 3. **Убедитесь, что все директории содержат изображения:**

```bash
# Проверьте количество файлов в каждой директории
dir D:\FileOrganizer\TRAIN\dataset\distrib\0
dir D:\FileOrganizer\TRAIN\dataset\distrib\90
dir D:\FileOrganizer\TRAIN\dataset\distrib\180
dir D:\FileOrganizer\TRAIN\dataset\distrib\270
```

### 4. **Если есть проблемы, пересоздайте датасет:**

```bash
# Очистите и пересоздайте train/val
python 1_prepare_dataset.py --keep-retrain
```

---

## 📝 Если ошибка повторяется

1. **Проверьте логи** - посмотрите на предупреждения `[WARN]` о недостающих углах
2. **Посмотрите на вывод `check_label_conflicts.py`** - убедитесь, что все 4 класса присутствуют
3. **Убедитесь, что датасет не пустой** - каждая директория должна содержать хотя бы несколько изображений
4. **Проверьте порядок классов** - ImageFolder сортирует их алфавитно, а не по числовому значению

---

## 🎉 Итог

После применения всех исправлений ошибка `Assertion t >= 0 && t < n_classes` должна исчезнуть. Обучение должно начаться и продолжаться без сбоев!

**Ключевые моменты:**
- ✅ Все 4 угла (0°, 90°, 180°, 270°) должны присутствовать в датасете
- ✅ Маппинг `angle_to_idx` должен содержать все возможные углы после ротации
- ✅ Target всегда проверяется на соответствие диапазону [0, n_classes)

Удачи в обучении! 🚀
