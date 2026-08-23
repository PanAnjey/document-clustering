"""Запуск анализа датасета retrain."""
import sys
from pathlib import Path

# Добавляем путь к скрипту
sys.path.insert(0, str(Path(__file__).parent))

from analyze_retrain_simple import analyze_retrain_dataset


if __name__ == "__main__":
    analyze_retrain_dataset()
