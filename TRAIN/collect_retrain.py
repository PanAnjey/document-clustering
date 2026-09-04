"""Сбор дополнительных файлов для датасета retrain."""
import shutil
from pathlib import Path


def collect_additional_files():
    """Собирает дополнительные файлы из distrib и val в retrain."""
    
    print("=" * 60)
    print("Сбор дополнительных файлов для retrain")
    print("=" * 60 + "\n")
    
    # Путь к исходному датасету
    distrib_dir = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")
    val_dir = Path(r"D:\FileOrganizer\TRAIN\dataset\val")
    retrain_dir = Path(r"D:\FileOrganizer\TRAIN\retrain")
    
    # Классы для сбора
    target_classes = [0, 90, 180, 270]
    
    # Текущее количество файлов в retrain
    current_counts = {}
    for cls in target_classes:
        cls_dir = retrain_dir / str(cls)
        if cls_dir.exists():
            files = list(cls_dir.glob("*"))
            current_counts[cls] = len(files)
        else:
            current_counts[cls] = 0
    
    print("Текущее количество файлов в retrain:")
    for cls, count in sorted(current_counts.items()):
        print(f"  {cls:>4}°: {count}")
    
    # Цель — минимум 10 файлов на класс
    target_count = 10
    
    # Собираем файлы из distrib и val
    collected = {}
    for cls in target_classes:
        needed = max(0, target_count - current_counts[cls])
        
        if needed == 0:
            print(f"\n{cls:>4}°: Достаточно файлов ({current_counts[cls]})")
            continue
        
        # Сначала пробуем из distrib
        source_dir = distrib_dir / str(cls)
        if not source_dir.exists():
            print(f"[WARN] Директория не найдена: {source_dir}")
            collected[cls] = 0
            continue
        
        files = [f for f in source_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        
        # Берём файлы, которых ещё нет в retrain
        retrain_files = set((retrain_dir / str(cls)).glob("*")) if (retrain_dir / str(cls)).exists() else []
        new_files = [f for f in files if f not in retrain_files]
        
        # Ограничиваем количество
        to_collect = min(needed, len(new_files))
        
        if to_collect > 0:
            target_retrain_dir = retrain_dir / str(cls)
            target_retrain_dir.mkdir(parents=True, exist_ok=True)
            
            for f in new_files[:to_collect]:
                shutil.copy(f, target_retrain_dir / f.name)
            
            collected[cls] = to_collect
            print(f"{cls:>4}°: добавлено {to_collect} файлов (всего: {current_counts[cls] + to_collect})")
        else:
            collected[cls] = 0
            print(f"{cls:>4}°: нет новых файлов в distrib ({len(new_files)} всего)")
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("Итоговая статистика:")
    total_collected = sum(collected.values())
    for cls in target_classes:
        new_total = current_counts[cls] + collected.get(cls, 0)
        status = "✅" if new_total >= target_count else "⚠️"
        print(f"{status} {cls:>4}°: {new_total} файлов (цель: {target_count})")
    
    print(f"\nВсего добавлено: {total_collected} файлов")


if __name__ == "__main__":
    collect_additional_files()
