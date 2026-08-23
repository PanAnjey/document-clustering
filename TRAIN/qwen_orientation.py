"""Определение ориентации документов через Qwen3.5-35B с llama.cpp"""
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from PIL import Image
import sys

# Конфигурация
LLAMA_CLI = Path(r"D:\llama.cpp\llama-mtmd-cli.exe")
MODEL_PATH = Path(r"D:\MODELS\STUDIO\HauhauCS\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
MMPROJ_PATH = Path(r"D:\MODELS\STUDIO\HauhauCS\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive\mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf")
PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

ANGLE_ORDER = [0, 90, 180, 270]

PROMPT = """Analyze this document image and determine its orientation.
The document could be rotated at 0°, 90°, 180°, or 270°.

Look at the text direction, layout, and content to determine the correct orientation.

Respond with ONLY the angle number (0, 90, 180, or 270).
Answer:"""

def build_ground_truth():
    """Строит словарь filename -> true_angle из distrib"""
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    gt[f.name] = angle
    return gt

def predict_orientation(image_path: Path) -> tuple[int, float]:
    """Определяет ориентацию через llama.cpp"""
    try:
        # Запускаем llama-cli с vision support
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
            "--mmproj", str(MMPROJ_PATH),
            "--image", str(image_path),
            "-p", PROMPT,
            "-n", "10",  # max tokens
            "--temp", "0.1",
            "-ngl", "99",  # GPU layers
            "--image-min-tokens", "1024"  # recommended for Qwen-VL
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout.strip()
        
        # Если stdout пустой, проверяем stderr
        if not output:
            output = result.stderr.strip()
        
        # Извлекаем число из ответа
        for angle in ANGLE_ORDER:
            if str(angle) in output:
                return angle, 0.9  # confidence placeholder
        
        return 0, 0.0
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return 0, 0.0

def main():
    print("=== Определение ориентации через Qwen3.5-35B ===\n")
    
    # Проверяем наличие файлов
    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli не найден: {LLAMA_CLI}")
        return
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Модель не найдена: {MODEL_PATH}")
        return
    
    if not MMPROJ_PATH.exists():
        print(f"ERROR: mmproj не найден: {MMPROJ_PATH}")
        return
    
    # Загружаем ground truth
    ground_truth = build_ground_truth()
    print(f"Ground truth: {len(ground_truth)} файлов\n")
    
    # Получаем список файлов
    files = [f for f in PDF_IMAGES_DIR.iterdir() 
             if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    print(f"Файлов для обработки: {len(files)}\n")
    
    results = []
    t0 = time.time()
    
    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        
        if fname not in ground_truth:
            continue
        
        true_angle = ground_truth[fname]
        pred_angle, conf = predict_orientation(fpath)
        
        results.append({
            "file": fname,
            "true_angle": true_angle,
            "qwen_pred": pred_angle,
            "qwen_conf": conf,
            "qwen_correct": pred_angle == true_angle
        })
        
        if i % 10 == 0:
            elapsed = time.time() - t0
            correct = sum(1 for r in results if r["qwen_correct"])
            acc = correct / len(results) * 100 if results else 0
            print(f"Обработано {i}/{len(files)} | Accuracy: {acc:.2f}% | {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    total = len(results)
    
    if total == 0:
        print("Нет результатов для анализа")
        return
    
    # Подсчёт метрик
    correct = sum(1 for r in results if r["qwen_correct"])
    accuracy = correct / total
    
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"Qwen3.5-35B: {accuracy:.2%} ({correct}/{total})")
    
    # Per-class метрики
    print(f"\n=== Per-class метрики ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in results if r["true_angle"] == angle]
        if angle_results:
            correct = sum(1 for r in angle_results if r["qwen_correct"])
            total_angle = len(angle_results)
            print(f"  {angle:3d}°: {correct}/{total_angle} = {correct/total_angle:.2%}")
    
    # Confusion matrix
    print(f"\n=== Confusion matrix (Qwen3.5-35B) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["qwen_pred"]] += 1
    
    header = "         pred: " + "  ".join(f"{a:3d}" for a in ANGLE_ORDER)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in ANGLE_ORDER]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]")
    
    # Ошибки
    errors = [r for r in results if not r["qwen_correct"]]
    print(f"\n=== Ошибки ({len(errors)} файлов) ===")
    for r in errors[:20]:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  qwen={r['qwen_pred']:3d}")
    
    # Сохранение результатов
    report_path = Path(r"D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "qwen_accuracy": round(accuracy, 4),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")

if __name__ == "__main__":
    main()
