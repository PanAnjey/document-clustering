"""Анализ датасета retrain для улучшения распознавания ориентации."""
from pathlib import Path


def analyze_retrain_dataset():
    """Анализирует распределение классов и потенциальные проблемы в реtrain."""
    
    print("=" * 70)
    print("АНАЛИЗ ДАТАСЕТА ДЛЯ ДООБУЧЕНИЯ (retrain)")
    print("=" * 70 + "\n")
    
    retrain_dir = Path(r"D:\FileOrganizer\TRAIN\retrain")
    
    if not retrain_dir.exists():
        print(f"[ERROR] Директория не найдена: {retrain_dir}")
        return
    
    # Подсчитаем количество файлов в каждой директории
    class_counts = {}
    for cls in [0, 90, 180, 270]:
        cls_dir = retrain_dir / str(cls)
        if cls_dir.exists():
            files = list(cls_dir.glob("*"))
            class_counts[cls] = len(files)
            
            # Покажем первые 3 файла для проверки
            if files:
                print(f"\nКласс {cls}° ({len(files)} файлов):")
                for f in files[:3]:
                    size_kb = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size_kb:.1f} KB)")
        else:
            class_counts[cls] = 0
    
    # Вывод общей статистики
    total = sum(class_counts.values())
    
    print("\n" + "=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"\nВсего файлов: {total}")
    
    if total > 0:
        print("\nРаспределение по классам:")
        for cls, count in sorted(class_counts.items()):
            pct = 100 * count / total
            bar = "█" * int(pct / 2)
            status = ""
            
            if cls == 180 and count == 0:
                status = " ⚠️ ПУСТО!"
            elif count < 3:
                status = " ⚠️ МАЛО!"
            
            print(f"  {cls:>4}°: {count:5d} ({pct:6.2f}%) {bar}{status}")
        
        # Проверка дисбаланса
        non_zero_counts = [c for c in class_counts.values() if c > 0]
        max_count = max(non_zero_counts)
        min_count = min(non_zero_counts)
        
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nДисбаланс классов:")
        print(f"  Максимальный класс: {max_count} изображений")
        print(f"  Минимальный класс:  {min_count} изображений")
        print(f"  Соотношение макс/мин: {ratio:.2f}x")
        
        if ratio > 3:
            print("\n⚠️  ВНИМАНИЕ: Сильный дисбаланс классов!")
    
    # Проверка на наличие ошибок 0° vs 180°
    count_0 = class_counts.get(0, 0)
    count_180 = class_counts.get(180, 0)
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА СИММЕТРИИ 0° vs 180°")
    print("=" * 70)
    print(f"\nКласс 0°: {count_0} изображений")
    print(f"Класс 180°: {count_180} изображений")
    
    if count_0 > 0 and count_180 == 0:
        print("\n⚠️  КРИТИЧЕСКАЯ ПРОБЛЕМА: Класс 180° ПУСТОЙ!")
        print("   Это объясняет путаницу между 0° и 180°.")
    elif count_0 > 0 and count_180 > 0:
        ratio_0_180 = max(count_0, count_180) / min(count_0, count_180)
        print(f"Соотношение: {ratio_0_180:.2f}x")
        
        if ratio_0_180 > 2:
            print("\n⚠️  ВНИМАНИЕ: Дисбаланс между 0° и 180°!")
    
    # Рекомендации
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 70)
    
    if count_180 == 0:
        print("\n🔥 КРИТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
        print("   1. Добавить минимум 5-10 изображений в класс 180°")
        print("   2. Или использовать VerticalFlip аугментацию для дублирования 0° → 180°")
        print("   3. Рассмотреть возможность обучения с меньшим количеством эпох")
    
    if total < 50:
        print(f"\n📊 МАЛО ДАННЫХ (всего {total} файлов):")
        print("   Рекомендуется собрать больше ошибок для дообучения")
        print("   Минимум 100-200 изображений для стабильного обучения")
    
    if ratio > 3:
        print(f"\n⚖️  СИЛЬНЫЙ ДИСБАЛАНС ({ratio:.1f}x):")
        print("   Используйте CLASS_WEIGHTS='effective_num' в config.py")
        print("   Или добавьте аугментацию для меньших классов")


if __name__ == "__main__":
    analyze_retrain_dataset()
