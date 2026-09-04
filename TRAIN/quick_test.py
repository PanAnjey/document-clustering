"""Быстрый тест для проверки работоспособности ViT модели."""
import torch
import timm
from PIL import Image
import numpy as np


def quick_test():
    """Проверяет, что ViT модель загружается и работает."""
    
    print("=" * 60)
    print("Быстрый тест Vision Transformer")
    print("=" * 60 + "\n")
    
    # Проверка GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[OK] Доступно {num_gpus} GPU:")
        for i in range(num_gpus):
            print(f"  - {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    else:
        print("[WARN] GPU не обнаружен, используется CPU")
    
    # Загрузка модели
    print("\n[INFO] Загрузка ViT Base (384x384 для A4 сканов)...")
    try:
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=0)
        print(f"[OK] Модель загружена: {model.__class__.__name__}")
        
        # Проверка параметров
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Параметры модели: {num_params:,} ({num_params / 1e6:.2f}M)")
        
    except Exception as e:
        print(f"[ERROR] Загрузка модели: {e}")
        return False
    
    # Тестовый инференс
    print("\n[INFO] Тестовый прогон...")
    
    # Создаём тестовое изображение (случайный шум)
    test_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image).convert("RGB")
    
    # Трансформация
    from torchvision import transforms
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        norm,
    ])
    
    tensor = transform(pil_image).unsqueeze(0)
    
    # Инференс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        
        print(f"[OK] Инференс завершён:")
        print(f"  Предсказание: класс {pred}")
        print(f"  Уверенность: {confidence:.2%}")
    
    # Проверка памяти GPU (если есть)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n[INFO] Использование GPU:")
        print(f"  Выделено: {allocated:.2f} GB")
        print(f"  Резерв:   {reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("Тест завершён успешно!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
