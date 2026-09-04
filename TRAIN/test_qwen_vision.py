import requests
import base64
from pathlib import Path

# Тестовый файл
test_file = Path(r"D:\FileOrganizer\Extracted\PDF_Images\000647281.b60a8a03-339f-4c38-8d02-45e2ebeb47bb_rot090.png")

# Конвертируем в base64
with open(test_file, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# OpenAI-compatible API
payload = {
    "model": "qwen",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the orientation of this document? Answer with only: 0, 90, 180, or 270."
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
    "http://127.0.0.1:8082/v1/chat/completions",
    json=payload,
    timeout=60
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
