"""Быстрый тест gemma-4-e4b-it с новым промптом на 50 файлах (LM Studio)"""
import json
import time
import requests
from pathlib import Path
from collections import defaultdict
from PIL import Image
import base64
import sys

# Импорт модуля для работы с MySQL
sys.path.insert(0, r"D:\Yandex.Disk\PYTHON\DIADOC\optimized")
import db_mysql_optimized as mysql_db

# Конфигурация
LM_STUDIO_URL = "http://127.0.0.1:1234"
PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

# Модель, загруженная в LM Studio
LM_STUDIO_MODEL = "gemma-4-e4b-it"

ANGLE_ORDER = [0, 90, 180, 270]

# Конфигурация БД (из db_mysql_optimized.py)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "passwd": "mysql",
    "port": 3308,
    "charset": "utf8mb4",
    "collation": "utf8mb4_general_ci"
}

def init_db():
    """Инициализация подключения к БД."""
    mysql_db.init_db(DB_CONFIG)

def build_ground_truth():
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    gt[f.name] = angle
    return gt

def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def predict_orientation(image_path: Path, server_url: str, true_angle:int) -> tuple[str, int, float]:
    try:
        img_base64 = image_to_base64(image_path)
        
        payload = {
            "model": LM_STUDIO_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":(f"Обрати внимание что данный документ повернут на {true_angle} градусов по часовой стрелке относительно правильной ориентации. "
                            f"На основе анализа  изображения из этого файла документа напиши промп который поможет модели на основе выделения характерных элементов, "
                            f"определять ориентацию изображения других подобный документов в такой же ориентации, которые в основном будут сканами. "
                            f"Не включай в текст промпта утверждений что этот документ повернут на конкретный угол. Добавь в текст критерии оценки и определение степени уверенности"
                            f"текст промпта должен быть на русском языке"
                                )
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
            return content
        
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("=== Быстрый тест Qwen3.5-35B через LM Studio ===\n")
    
    server_url = LM_STUDIO_URL
    
    try:
        r = requests.get(f"{server_url}/v1/models", timeout=5)
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"Сервер LM Studio работает. Доступные модели: {models}\n")
        else:
            print(f"ERROR: Сервер вернул {r.status_code}")
            return
    except requests.ConnectionError:
        print("ERROR: LM Studio не запущен. Запустите LM Studio, загрузите модель и запустите сервер (порт 1234)")
        return
    
    ground_truth = build_ground_truth()
    print(f"Ground truth: {len(ground_truth)} файлов\n")
    
    files = [f for f in PDF_IMAGES_DIR.iterdir() 
             if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    print(f"Тестируем на {len(files)} файлах\n")
    
    results = []
    t0 = time.time()
    
    init_db()
    
    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        
        if fname not in ground_truth:
            continue
        
        true_angle = ground_truth[fname]
        content = predict_orientation(fpath, server_url, true_angle)
        
        print(f"{i:3d}/{len(files)} ")
        
        if content != None:
            insert_sql = """
                INSERT INTO nltk.test_prompt 
                (model, file_name, content, true_angle)
                VALUES (%s, %s,%s,%s)
            """
            try:
                mysql_db.exec_val(insert_sql, (
                    LM_STUDIO_MODEL,
                    fname,
                    content,
                    true_angle
                ))
            except Exception as e:
                print(f"Ошибка сохранения в БД для {fname}: {e}")
    
    elapsed = time.time() - t0
    total = len(results)
    
    if total == 0:
        print("Нет результатов")
        return
    
    no_text = [r for r in results if r["qwen_pred"] == -1]
    classified = [r for r in results if r["qwen_pred"] != -1]
    
    correct = sum(1 for r in classified if r["qwen_correct"])
    accuracy = correct / len(classified) if classified else 0
    
    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"Всего файлов: {total}")
    print(f"Нет текста (pred=-1): {len(no_text)} ({len(no_text)/total:.2%})")
    print(f"Классифицировано: {len(classified)}")
    print(f"Правильных из классифицированных: {correct}/{len(classified)} = {accuracy:.2%}")
    
    print(f"\n=== Per-class метрики ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in classified if r["true_angle"] == angle]
        angle_no_text = [r for r in no_text if r["true_angle"] == angle]
        if angle_results or angle_no_text:
            correct_angle = sum(1 for r in angle_results if r["qwen_correct"])
            total_angle = len(angle_results) + len(angle_no_text)
            print(f"  {angle:3d}: {correct_angle}/{total_angle} = {correct_angle/total_angle:.2%}  (net text: {len(angle_no_text)})")
    
    print(f"\n=== Confusion matrix ===")
    confmatrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        confmatrix[r["true_angle"]][r["qwen_pred"]] += 1
    
    all_preds = [-1] + ANGLE_ORDER
    header = "         pred: " + "  ".join(f"{a:4d}" for a in all_preds)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [confmatrix[true_a][pred_a] for pred_a in all_preds]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d}]")
    
    report_path = Path(r"D:\FileOrganizer\TRAIN\models\qwen_new_prompt_test.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "qwen_accuracy": round(accuracy, 4),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт: {report_path}")
    
    mysql_db.close()

if __name__ == "__main__":
    main()
