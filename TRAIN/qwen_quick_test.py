"""Быстрый тест qwen2.5-vl-7b-instruct через LM Studio с prompt.md и JSON-ответом"""
import json
import re
import sys
import time
import base64
import requests
from pathlib import Path
from collections import defaultdict

# Импорт модуля для работы с MySQL
sys.path.insert(0, r"D:\Yandex.Disk\PYTHON\DIADOC\optimized")
import db_mysql_optimized as mysql_db

# Конфигурация
LM_STUDIO_URL = "http://localhost:1234"
MODEL_ID = "qwen2.5-vl-32b-instruct"
MODEL_DISPLAY_NAME = "qwen2.5-vl-32b-instruct"

PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")
PROMPT_FILE = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN\prompt.md")

ANGLE_ORDER = [0, 90, 180, 270]

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "passwd": "mysql",
    "port": 3308,
    "charset": "utf8mb4",
    "collation": "utf8mb4_general_ci"
}

TEST_LIMIT = 10  # Ограничение на количество файлов для быстрого теста


def load_prompt() -> str:
    """Загрузка промпта из файла prompt.md."""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def init_db():
    """Инициализация БД и очистка старых записей."""
    mysql_db.init_db(DB_CONFIG)
    try:
        mysql_db.exec_val("DELETE FROM nltk.test_orient WHERE model_name = %s", (MODEL_DISPLAY_NAME,))
        mysql_db.mydb.commit()
        print(f"[DB] Удалены старые записи для модели {MODEL_DISPLAY_NAME}\n")
    except Exception as e:
        print(f"[WARN] Ошибка при удалении старых записей: {e}")


def build_ground_truth() -> dict[str, int]:
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    gt[f.name] = angle
    return gt


def parse_json_response(content: str) -> dict | None:
    """Парсинг JSON-ответа модели.

    Пробует извлечь JSON из ответа (может быть внутри ```json ... ```).
    """
    if not content:
        return None

    # Попытка найти JSON внутри markdown-блока
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Попытка найти JSON-объект напрямую
    json_match = re.search(r'\{.*?\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Попытка распарсить весь ответ как JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    return None


def predict_orientation(image_path: Path, prompt: str) -> dict:
    """Отправляет запрос к LM Studio и возвращает полный результат."""
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            }
        ],
        # "max_tokens": 2048,
        "temperature": 0.1
    }

    result = {
        "raw_response": None,
        "parsed_json": None,
        "pred_angle": -1,
        "confidence": 0.0
    }

    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            timeout=180
        )

        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json["choices"][0]["message"]["content"].strip()
            result["raw_response"] = content

            parsed = parse_json_response(content)
            if parsed:
                result["parsed_json"] = parsed
                angle = parsed.get("orientation", -1)
                # Нормализация: -90 -> 270, -180 -> 180, >360 -> mod 360
                if angle not in (-1, 0, 90, 180, 270) and angle is not None:
                    angle = (angle % 360 + 360) % 360
                    angle = angle if angle in (0, 90, 180, 270) else angle
                result["pred_angle"] = angle
                result["confidence"] = parsed.get("confidence", 0.0)

        return result

    except Exception as e:
        print(f"Error: {e}")
        return result


def save_to_db(fname: str, true_angle: int, raw_response: str, parsed: dict | None, pred_angle: int):
    """Сохранение результата в БД."""
    try:
        if parsed:
            votes = parsed.get("votes", {})
            evidence = json.dumps(parsed.get("key_evidence", []), ensure_ascii=False)
            contradictions = json.dumps(parsed.get("contradictions", []), ensure_ascii=False)

            insert_sql = """
                INSERT INTO nltk.test_orient 
                (model_name, file_name, true_angle, pred_angle,
                 orientation, correction_angle, correction_direction, confidence,
                 votes_0, votes_90, votes_180, votes_270,
                 applied_criteria, key_evidence, contradictions, raw_response)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            mysql_db.exec_val(insert_sql, (
                MODEL_DISPLAY_NAME,
                fname,
                true_angle,
                pred_angle,
                parsed.get("orientation", -1),
                parsed.get("correction_angle", 0),
                parsed.get("correction_direction", "none"),
                parsed.get("confidence", 0.0),
                votes.get("0", 0),
                votes.get("90", 0),
                votes.get("180", 0),
                votes.get("270", 0),
                parsed.get("applied_criteria", 0),
                evidence if evidence != "[]" else None,
                contradictions if contradictions != "[]" else None,
                raw_response[:32000] if raw_response else None
            ))
        else:
            # Ответ не JSON — сохраняем только базовые поля
            insert_sql = """
                INSERT INTO nltk.test_orient 
                (model_name, file_name, true_angle, pred_angle, raw_response)
                VALUES (%s, %s, %s, %s, %s)
            """
            mysql_db.exec_val(insert_sql, (
                MODEL_DISPLAY_NAME,
                fname,
                true_angle,
                pred_angle,
                raw_response[:32000] if raw_response else None
            ))

    except Exception as e:
        print(f"Ошибка сохранения в БД для {fname}: {e}")


def main():
    print(f"=== Тест {MODEL_DISPLAY_NAME} с prompt.md ({TEST_LIMIT} файлов) ===\n")
    print(f"LM Studio: {LM_STUDIO_URL}")

    try:
        r = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            if MODEL_ID not in models:
                print(f"WARN: Модель {MODEL_ID} не найдена в LM Studio!")
                print(f"Доступные модели: {models}")
    except Exception:
        print("ERROR: LM Studio не запущен")
        return

    # Загрузка промпта
    prompt = load_prompt()
    print(f"Промпт загружен: {len(prompt)} символов\n")
    print(prompt)
    ground_truth = build_ground_truth()
    print(f"Ground truth: {len(ground_truth)} файлов\n")

    # Файлы для обработки
    all_files = [f for f in PDF_IMAGES_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    files = all_files[:TEST_LIMIT]
    print(f"Файлов для обработки: {len(files)}\n")

    # Инициализация БД (удаляет старые записи)
    init_db()

    results = []
    n_json = 0
    t0 = time.time()

    for i, fpath in enumerate(files, 1):
        fname = fpath.name

        if fname not in ground_truth:
            continue

        true_angle = ground_truth[fname]
        pred = predict_orientation(fpath, prompt)

        pred_angle = pred["pred_angle"]
        correct = (pred_angle == true_angle)

        results.append({
            "file": fname,
            "true_angle": true_angle,
            "qwen_pred": pred_angle,
            "qwen_conf": pred["confidence"],
            "qwen_correct": correct,
            "is_json": pred["parsed_json"] is not None
        })

        if pred["parsed_json"]:
            n_json += 1

        status = f"OK (c={pred['confidence']:.1%})" if correct else f"FAIL (c={pred['confidence']:.1%})"
        json_mark = "[JSON]" if pred["parsed_json"] else "[RAW]"
        print(f"{i:3d}/{len(files)} | true={true_angle:3d} pred={pred_angle:3d} {status} {json_mark}")

        # Сохранение в БД сразу после каждого файла
        save_to_db(fname, true_angle, pred["raw_response"], pred["parsed_json"], pred_angle)

    elapsed = time.time() - t0
    total = len(results)

    if total == 0:
        print("Нет результатов")
        mysql_db.close()
        return

    # Подсчёт метрик
    no_pred = [r for r in results if r["qwen_pred"] == -1]
    classified = [r for r in results if r["qwen_pred"] != -1]
    correct = sum(1 for r in classified if r["qwen_correct"])
    accuracy = correct / len(classified) if classified else 0

    print(f"\n=== Результаты ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"Всего файлов: {total}")
    print(f"JSON-ответов: {n_json}/{total} ({n_json/total:.2%})")
    print(f"Нет предсказания (pred=-1): {len(no_pred)}")
    print(f"Классифицировано: {len(classified)}")
    print(f"Правильных из классифицированных: {correct}/{len(classified)} = {accuracy:.2%}")

    print(f"\n=== Per-class метрики ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in classified if r["true_angle"] == angle]
        angle_no_pred = [r for r in no_pred if r["true_angle"] == angle]
        if angle_results or angle_no_pred:
            correct_angle = sum(1 for r in angle_results if r["qwen_correct"])
            total_angle = len(angle_results) + len(angle_no_pred)
            print(f"  {angle:3d}deg: {correct_angle}/{total_angle} = {correct_angle/total_angle:.2%}"
                  f"  (no_pred: {len(angle_no_pred)})")

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

    # Сохранение отчёта
    report_path = Path(r"D:\FileOrganizer\TRAIN\models\qwen_prompt_test.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "json_responses": n_json,
            "qwen_accuracy": round(accuracy, 4),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nОтчёт: {report_path}")

    mysql_db.close()
    print("БД закрыта")


if __name__ == "__main__":
    main()
