"""Тест qwen2.5-vl-7b-instruct с простым английским промптом"""
import requests, base64, json, time
from pathlib import Path

test_files = [
    (r"D:\FileOrganizer\Extracted\PDF_Images\000647281.b60a8a03-339f-4c38-8d02-45e2ebeb47bb.png", 0),
    (r"D:\FileOrganizer\Extracted\PDF_Images\+Показания по эл.энергии 1й лист.d7feb45e-6bf7-46c3-9a0f-6877db25accd.png", 90),
]

# Английский промпт
prompt = """Analyze this document and determine its orientation angle.
The orientation is the clockwise rotation needed to make the document upright (0 degrees).

Options:
- 0: document is upright, text reads left-to-right top-to-bottom
- 90: document is rotated 90° clockwise (top is on the right)
- 180: document is upside down (top is at the bottom)
- 270: document is rotated 270° clockwise (top is on the left)

Return a JSON:
{"orientation": <angle>, "votes": {"0": N, "90": N, "180": N, "270": N}, "applied_criteria": N}
"""

for file_path, true_angle in test_files:
    fname = Path(file_path).name[:40]
    
    with open(file_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": "qwen2.5-vl-7b-instruct",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    t0 = time.time()
    r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=120)
    dt = time.time() - t0
    
    content = r.json()["choices"][0]["message"]["content"]
    print(f"\nFile: {fname}")
    print(f"True: {true_angle}  |  Time: {dt:.1f}s")
    print(f"Response: {content[:300]}")
