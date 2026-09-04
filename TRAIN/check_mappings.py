"""Проверка маппингов классов в OrientationDataset."""
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

import config as C
from torchvision import datasets


def check_mappings():
    """Проверяет корректность маппингов классов."""
    
    print("=" * 60)
    print("Проверка маппингов классов для OrientationDataset")
    print("=" * 60 + "\n")
    
    # Загружаем ImageFolder
    base_train = datasets.ImageFolder(str(C.TRAIN_DIR))
    
    print(f"Классы ImageFolder: {base_train.classes}")
    print(f"class_to_idx: {base_train.class_to_idx}\n")
    
    # Создаем маппинги как в OrientationDataset
    try:
        idx_to_angle = {i: int(name) for i, name in enumerate(base_train.classes)}
        angle_to_idx = {int(name): i for i, name in enumerate(base_train.classes)}
        
        print("idx_to_angle (индекс -> угол):")
        for k, v in sorted(idx_to_angle.items()):
            print(f"  {k} -> {v}°")
        
        print("\nangle_to_idx (угол -> индекс):")
        for k, v in sorted(angle_to_idx.items()):
            print(f"  {k}° -> {v}")
        
        # Проверяем все углы
        expected_angles = {0, 90, 180, 270}
        actual_angles = set(idx_to_angle.values())
        
        print("\nПроверка всех углов:")
        for angle in sorted(expected_angles):
            if angle in idx_to_idx := angle_to_idx:
                print(f"  ✓ {angle}° -> индекс {idx_to_idx[angle]}")
            else:
                print(f"  ✗ {angle}° НЕ НАЙДЕН!")
        
        # Проверяем ротацию аугментации
        print("\nПроверка ротации аугментации:")
        for orig_angle in sorted(expected_angles):
            if orig_angle not in idx_to_angle:
                continue
            
            class_idx = idx_to_angle[orig_angle]
            
            for k in range(4):  # 0, 1, 2, 3 поворота
                a = 90 * k
                new_angle = (orig_angle + a) % 360
                
                if new_angle in angle_to_idx:
                    target = angle_to_idx[new_angle]
                    print(f"  {orig_angle}° + {a}° -> {new_angle}° -> класс {target}")
                else:
                    print(f"  {orig_angle}° + {a}° -> {new_angle}° -> КЛАСС НЕ НАЙДЕН!")
        
        # Проверяем вертикальный флип
        print("\nПроверка вертикального флипа:")
        for orig_angle in sorted(expected_angles):
            if orig_angle not in idx_to_angle:
                continue
            
            class_idx = idx_to_angle[orig_angle]
            
            # Определяем классы для 0° и 180°, 90° и 270°
            angle_0_idx = angle_to_idx.get(0, -1)
            angle_180_idx = angle_to_idx.get(180, -1)
            angle_90_idx = angle_to_idx.get(90, -1)
            angle_270_idx = angle_to_idx.get(270, -1)
            
            print(f"  {orig_angle}° -> класс {class_idx}")
            if orig_angle == 0:
                print(f"    После VerticalFlip: {angle_180_idx}")
            elif orig_angle == 180:
                print(f"    После VerticalFlip: {angle_0_idx}")
            elif orig_angle == 90:
                print(f"    После VerticalFlip: {angle_270_idx}")
            elif orig_angle == 270:
                print(f"    После VerticalFlip: {angle_90_idx}")
        
        # Проверка на дисбаланс
        print("\nПроверка количества изображений по классам:")
        for cls in base_train.classes:
            count = sum(1 for _ in base_train.samples if str(cls) in str(_[0]))
            print(f"  {cls}: {count} изображений")
        
    except ValueError as e:
        print(f"[WARN] Обнаружен нечисловой класс: {e}")
        idx_to_angle = {}
        angle_to_idx = {}
        for i, name in enumerate(base_train.classes):
            try:
                angle = int(name)
                idx_to_angle[i] = angle
                angle_to_idx[angle] = i
            except ValueError:
                other_idx = i
                idx_to_angle[i] = -1
                angle_to_idx[-1] = i
        
        print("\nidx_to_angle (с нечисловыми классами):")
        for k, v in sorted(idx_to_angle.items()):
            print(f"  {k} -> {v}")
        
        print("\nangle_to_idx (с нечисловыми классами):")
        for k, v in sorted(angle_to_idx.items()):
            print(f"  {k} -> {v}")


if __name__ == "__main__":
    check_mappings()
