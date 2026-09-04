"""Сравнивает метки классов файлов из RETRAIN_DIR с их метками в DISTRIB_DIR.

Цель — выявить конфликтующие метки: один и тот же файл в retrain лежит в папке
одного угла, а в distrib — в папке другого. Это указывает на ошибку разметки
исходного датасета (модель «ошиблась» потому, что училась с неверной меткой).

Также показывает файлы, которых нет в distrib (вновь добавленные).

Запуск: py -3 check_label_conflicts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def index_dir(root: Path):
    """Вернуть {filename: angle_str} для всех изображений в root/{angle}."""
    out = {}
    if not root.exists():
        return out
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in C.IMAGE_EXTENSIONS:
                out[f.name] = sub.name
    return out


def main():
    retrain = index_dir(C.RETRAIN_DIR)
    distrib = index_dir(C.DISTRIB_DIR)
    if not retrain:
        print(f"RETRAIN_DIR пуст или не существует: {C.RETRAIN_DIR}")
        return
    if not distrib:
        print(f"[WARN] DISTRIB_DIR пуст или не существует: {C.DISTRIB_DIR}")

    conflicts = []
    not_in_distrib = []
    matches = []

    for fname, retrain_angle in retrain.items():
        if fname not in distrib:
            not_in_distrib.append((fname, retrain_angle))
        else:
            distrib_angle = distrib[fname]
            if distrib_angle != retrain_angle:
                conflicts.append((fname, distrib_angle, retrain_angle))
            else:
                matches.append((fname, distrib_angle))

    print(f"=== Сравнение retrain ({len(retrain)}) vs distrib ({len(distrib)}) ===\n")
    print(f"Совпадают метки:        {len(matches)}")
    print(f"Конфликтуют метки:      {len(conflicts)}")
    print(f"Нет в distrib (новые):  {len(not_in_distrib)}")

    if conflicts:
        print(f"\n--- КОНФЛИКТЫ (distrib -> retrain) ---")
        print(f"{'ФАЙЛ':<60} {'distrib':>8} -> {'retrain':>8}")
        for fname, da, ra in sorted(conflicts):
            print(f"{fname[:60]:<60} {da:>8} -> {ra:>8}")
        print(f"\nВсего конфликтов: {len(conflicts)}")
        print("Эти файлы имеют РАЗНЫЕ метки в distrib и retrain.")
        print("Если retrain-метка верна (по мнению оператора) — испрвьте distrib и переобучите с нуля.")

    if not_in_distrib:
        print(f"\n--- НОВЫЕ ФАЙЛЫ (есть в retrain, нет в distrib) ---")
        for fname, ra in sorted(not_in_distrib):
            print(f"  {fname[:70]:<70} retrain={ra}")

    if not conflicts and not not_in_distrib:
        print("\nВсе метки совпадают. Конфликтов нет.")


if __name__ == "__main__":
    main()