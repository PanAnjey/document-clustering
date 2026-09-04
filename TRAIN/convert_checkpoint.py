"""Конвертация старой модели (state_dict) в новый формат (полный объект модели)."""
import torch
import torch.nn as nn
import timm
from pathlib import Path

import config as C


class ViTAttention(nn.Module):
    """Attention-механизм (должен совпадать с 2_train.py)."""
    def __init__(self, in_features, hidden_size=512, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 4:
            attention_weights = self.attention(x)
            return x * attention_weights
        else:
            attention_weights = self.attention(x.unsqueeze(-1)).squeeze(-1)
            return x * attention_weights


def convert_model():
    """Конвертирует старую модель в новый формат."""
    old_path = C.MODELS_DIR / C.BEST_MODEL_FILENAME
    new_path = C.MODELS_DIR / f"{C.BEST_MODEL_FILENAME}_new"
    
    if not old_path.exists():
        print(f"[ERROR] Модель не найдена: {old_path}")
        return False
    
    print(f"[INFO] Загрузка старой модели: {old_path.name}")
    ckpt = torch.load(old_path, map_location="cpu", weights_only=True)
    
    # Определяем размер входа из имени файла
    model_name_str = old_path.name.lower()
    if "384" in model_name_str:
        input_size = 384
    elif "224" in model_name_str:
        input_size = 224
    else:
        input_size = C.IMAGE_SIZE
    
    print(f"[INFO] Размер входа: {input_size}x{input_size}")
    
    # Создаём архитектуру модели
    if "vit_large" in C.MODEL_NAME.lower():
        model = timm.create_model(f'vit_large_patch16_{input_size}', pretrained=False, num_classes=0)
        in_features = 1024
    else:
        model = timm.create_model(f'vit_base_patch16_{input_size}', pretrained=False, num_classes=0)
        in_features = 768
    
    # Добавляем attention-механизм (если включён)
    if C.USE_ATTENTION and hasattr(model, 'head'):
        model.head = ViTAttention(in_features, hidden_size=C.ATTENTION_HIDDEN_SIZE, 
                                  dropout=C.ATTENTION_DROPOUT)
        print(f"[INFO] Attention-механизм добавлен (hidden={C.ATTENTION_HIDDEN_SIZE})")
    
    # Заменяем классификатор
    if hasattr(model, 'head'):
        in_features = model.head.in_features if hasattr(model.head, 'in_features') else in_features
        model.head = nn.Linear(in_features, C.OUTPUT_CLASSES)
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, C.OUTPUT_CLASSES)
    
    # Загружаем веса
    print("[INFO] Загрузка весов...")
    model.load_state_dict(ckpt["model_state"], strict=False)
    
    # Сохраняем в новом формате (полный объект модели)
    print(f"[INFO] Сохранение новой модели: {new_path.name}")
    torch.save(model, new_path)
    
    print(f"[OK] Конвертация завершена!")
    print(f"     Старый файл: {old_path.name}")
    print(f"     Новый файл:  {new_path.name}")
    print(f"     Размер: {new_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return True


if __name__ == "__main__":
    if convert_model():
        print("\n[INFO] Теперь 3_test_inference.py может использовать новый формат!")
    else:
        print("\n[ERROR] Конвертация не выполнена.")
