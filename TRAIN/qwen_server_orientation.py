"""Определение ориентации через Qwen3.5-35B с llama-server"""
import json
import subprocess
import time
import requests
from pathlib import Path
from collections import defaultdict
from PIL import Image
import base64
import sys
import io

# Конфигурация
LLAMA_SERVER = Path(r"D:\llama.cpp\llama-server.exe")
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

def image_to_base64(image_path: Path) -> str:
    """Конвертирует изображение в base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def predict_orientation(image_path: Path, server_url: str) -> tuple[int, float]:
    """Определяет ориентацию через llama-server API"""
    try:
        # Конвертируем изображение в base64
        img_base64 = image_to_base64(image_path)
        
        # OpenAI-compatible API с отключением reasoning
        payload = {
            "model": "Qwen3.5-35B-A3B-Uncensored-HauhauCS",  # Имя модели (соответствует файлу)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Определи ориентацию текста на изображении. Ответь одним числом: 0, 90, 180 или 270 (в градусах по часовой стрелке).

Это число означает угол поворота изображения по часовой стрелке, который нужно применить к текущему виду, чтобы текст стал горизонтальным и читался нормально (сверху вниз, слева направо):

- 0 — текст уже горизонтален и читается нормально.
- 90 — верхняя часть документа сейчас находится справа (нужно повернуть вправо на 90°).
- 180 — документ перевернут вверх ногами (верхняя часть внизу).
- 270 — верхняя часть документа сейчас слева (нужно повернуть влево на 90°, или вправо на 270°).

Ответь только числом, без пояснений. Если на изображении нет текста, ответь -1."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Правильный парсинг: ищем точное совпадение числа
            import re
            match = re.search(r'\b(-1|0|90|180|270)\b', content)
            if match:
                return int(match.group(1)), 0.9
        
        return -1, 0.0
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return 0, 0.0

def main():
    print("=== Определение ориентации через Qwen3.5-35B (llama-server) ===\n")
    
    # Проверяем наличие файлов
    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server не найден: {LLAMA_SERVER}")
        return
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Модель не найдена: {MODEL_PATH}")
        return
    
    if not MMPROJ_PATH.exists():
        print(f"ERROR: mmproj не найден: {MMPROJ_PATH}")
        return
    
    # Запускаем llama-server
    print("Запуск llama-server...")
    server_process = subprocess.Popen([
        str(LLAMA_SERVER),
        "-m", str(MODEL_PATH),
        "--mmproj", str(MMPROJ_PATH),
        "-ngl", "99",
        "--port", "8081",
        "--host", "127.0.0.1",
        "--image-min-tokens", "1024"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    server_url = "http://127.0.0.1:8081"
    
    # Ждём, пока сервер запустится
    print("Ожидание запуска сервера...")
    time.sleep(10)
    
    try:
        # Проверяем, что сервер работает
        requests.get(f"{server_url}/health", timeout=5)
        print("Сервер запущен\n")
    except:
        print("ERROR: Сервер не запустился")
        server_process.terminate()
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
        pred_angle, conf = predict_orientation(fpath, server_url)
        
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
        server_process.terminate()
        return
    
    # Подсчёт метрик
    # Файлы с pred=-1 (нет текста) — отдельная категория
    no_text = [r for r in results if r["qwen_pred"] == -1]
    classified = [r for r in results if r["qwen_pred"] != -1]
    
    correct = sum(1 for r in classified if r["qwen_correct"])
    accuracy = correct / len(classified) if classified else 0
    
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"Всего файлов: {total}")
    print(f"Нет текста (pred=-1): {len(no_text)} ({len(no_text)/total:.2%})")
    print(f"Классифицировано: {len(classified)}")
    print(f"Правильных из классифицированных: {correct}/{len(classified)} = {accuracy:.2%}")
    
    # Per-class метрики (только для классифицированных)
    print(f"\n=== Per-class метрики ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in classified if r["true_angle"] == angle]
        angle_no_text = [r for r in no_text if r["true_angle"] == angle]
        if angle_results or angle_no_text:
            correct_angle = sum(1 for r in angle_results if r["qwen_correct"])
            total_angle = len(angle_results) + len(angle_no_text)
            print(f"  {angle:3d}°: {correct_angle}/{total_angle} = {correct_angle/total_angle:.2%}  (нет текста: {len(angle_no_text)})")
    
    # Confusion matrix (включая -1)
    print(f"\n=== Confusion matrix (Qwen3.5-35B) ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["qwen_pred"]] += 1
    
    all_preds = [-1] + ANGLE_ORDER
    header = "         pred: " + "  ".join(f"{a:4d}" for a in all_preds)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in all_preds]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d}]")
    
    # Ошибки (только классифицированные неправильно)
    errors = [r for r in classified if not r["qwen_correct"]]
    print(f"\n=== Ошибки классификации ({len(errors)} файлов) ===")
    for r in errors[:20]:
        print(f"  {r['file'][:55]:55s} true={r['true_angle']:3d}  qwen={r['qwen_pred']:3d}")
    
    # Сохранение результатов
    report_path = Path(r"D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "no_text_count": len(no_text),
            "classified_count": len(classified),
            "qwen_accuracy": round(accuracy, 4),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён: {report_path}")
    
    # Останавливаем сервер
    server_process.terminate()

if __name__ == "__main__":
    main()
