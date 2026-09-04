"""Тест LM Studio vision API"""
import base64, requests, json
from pathlib import Path

test_file = Path(r"D:\FileOrganizer\Extracted\PDF_Images\000647281.b60a8a03-339f-4c38-8d02-45e2ebeb47bb.png")

with open(test_file, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "model": "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive@q4_k_m",
    "messages": [
        {"role": "user",
        "content": [
            {"type": "text", "text": "What is the orientation of this document? Answer with only: 0, 90, 180, or 270."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}
    ],
    "max_tokens": 500,
    "temperature": 0.1
}

r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=60)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    full = r.json()
    msg = full["choices"][0]["message"]
    content = msg.get('content', '')
    reasoning = msg.get('reasoning_content', '')
    print(f"Content: '{content}'")
    if reasoning:
        print(f"Reasoning (last 200): '...{reasoning[-200:]}'")
else:
    print(r.text[:300])
