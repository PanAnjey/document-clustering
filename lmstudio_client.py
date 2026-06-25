# lmstudio_client.py
# Клиент для работы с LM Studio API (Qwen3.6-35B для уточнения кластеров).

import requests
from typing import Optional, List, Dict
from config import cfg
from logger_utils import logger


class LmStudioClient:
    """Клиент для работы с LM Studio API."""
    
    def __init__(self):
        self.base_url = cfg.LM_STUDIO_URL.rstrip('/')
        self.model = cfg.QWEN_36B_MODEL
    
    def refine_cluster(self, format_type: str, cluster_id: int, 
                       documents: str, doc_count: int) -> Optional[str]:
        """Уточнение кластера через Qwen3.6-35B /chat/completions.
        
        Args:
            format_type: тип формата (pdf_text, word_docx и т.д.)
            cluster_id: ID кластера
            documents: текст документов из кластера (объединённый)
            doc_count: количество документов в кластере
        
        Returns:
            Уточнённое описание кластера или None при ошибке
        """
        prompt = f"""Анализируй документы этого кластера и сгенерируй краткое описание темы.

Формат файлов: {format_type}
Количество документов: {doc_count}

Документы:
{documents[:4000]}  # Ограничиваем контекст

Опиши тему кластера в 1-2 предложениях:"""

        messages = [
            {"role": "system", "content": "Ты — эксперт по классификации документов. Генерируй краткие описания тем."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 256,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if content:
                    logger.info(f"Cluster {cluster_id} refined for {format_type}: {content[:100]}...")
                    return content
            
            logger.warning(f"LM Studio API error: {response.status_code} - {response.text}")
            
        except requests.exceptions.ConnectionError:
            logger.error(f"LM Studio not reachable at {self.base_url}")
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Проверяет доступность LM Studio API."""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
