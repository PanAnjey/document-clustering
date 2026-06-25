# pipelines/base_pipeline.py
# Базовый класс Pipeline для обработки файлов конкретного формата.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from config import cfg
from logger_utils import logger


class BasePipeline(ABC):
    """Базовый класс пайплайна для обработки файлов конкретного формата."""
    
    def __init__(self, format_type: str):
        self.format_type = format_type  # pdf_text, word_docx и т.д.
        self.table_name = f"documents_{format_type}"
        
    @abstractmethod
    def extract(self, file_path: Path) -> Dict:
        """Извлечение текста/изображений из файла.
        
        Returns:
            {
                "source": str(file_path),
                "type": format_type,
                "text": Optional[str],
                "image": Optional[str],  # путь к изображению (если есть)
                "error": Optional[str],
                "text_quality": str  # "good", "poor", "none"
            }
        """
        pass
    
    def summarize(self, data: Dict) -> Dict:
        """LLM саммаризация через Qwen3.5-4B (transformers).
        
        По умолчанию — пропускает (не все форматы нуждаются в саммаризации).
        Переопределить в подклассе если нужна специфичная логика.
        """
        return data
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Генерация текстового эмбеддинга через nomic-embed-text-v1.5."""
        if not text or not text.strip():
            return None
        
        try:
            from embeddings_engine import get_engine
            engine = get_engine()
            emb_dir = cfg.EMBEDDINGS_DIR
            emb_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = engine.get_text_embeddings([text], emb_dir, [self.format_type])
            if embeddings.size > 0:
                return embeddings[0]
        except Exception as e:
            logger.warning(f"Text embedding failed for {self.format_type}: {e}")
        
        return None
    
    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """Генерация графического эмбеддинга через nomic-embed-vision-v1.5."""
        if not image_path or not Path(image_path).exists():
            return None
        
        try:
            from embeddings_engine import get_engine
            engine = get_engine()
            emb_dir = cfg.EMBEDDINGS_DIR
            emb_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = engine.get_image_embeddings([image_path], emb_dir, [self.format_type])
            if embeddings.size > 0:
                return embeddings[0]
        except Exception as e:
            logger.warning(f"Image embedding failed for {self.format_type}: {e}")
        
        return None
    
    def cluster(self, data: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Грубая кластеризация через HDBSCAN на combined embeddings."""
        from cluster_engine import ClusterAnalyzer
        
        if len(data) < 2:
            for d in data:
                d.setdefault("cluster", 0)
            return data
        
        analyzer = ClusterAnalyzer(similarity_threshold=similarity_threshold)
        
        vectors = []
        for d in data:
            v = None
            if "text_embedding" in d and "image_embedding" in d:
                v = (d["text_embedding"] + d["image_embedding"]) / 2
            elif "text_embedding" in d:
                v = d["text_embedding"]
            elif "image_embedding" in d:
                v = d["image_embedding"]
            
            vectors.append(v)
        
        valid = [(i, v) for i, v in enumerate(vectors) if v is not None]
        if len(valid) < 2:
            for d in data:
                d["cluster"] = 0
            return data
        
        emb_matrix = np.array([v for _, v in valid])
        emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        
        labels = analyzer.perform_clustering(emb_matrix)
        
        for idx, (orig_i, _) in enumerate(valid):
            data[orig_i]["cluster"] = int(labels[idx])
        
        no_emb = [i for i, v in enumerate(vectors) if v is None]
        for i in no_emb:
            data[i]["cluster"] = 0
        
        return data
    
    def refine_clusters(self, clusters_data: List[Dict]) -> List[Dict]:
        """Уточнение кластеров через Qwen3.6-35B (LM Studio /chat/completions).
        
        Для каждого кластера: анализ документов → генерация уточнённого описания.
        Без поддержки /embed (только чат).
        """
        if not cfg.CLUSTER_REFINEMENT_ENABLED:
            logger.info(f"Cluster refinement disabled for {self.format_type}")
            return clusters_data
        
        try:
            from lmstudio_client import LmStudioClient
            client = LmStudioClient()
            
            # Группируем документы по кластерам
            cluster_groups = {}
            for d in clusters_data:
                cid = d.get("cluster", 0)
                if cid not in cluster_groups:
                    cluster_groups[cid] = []
                cluster_groups[cid].append(d)
            
            # Для каждого кластера генерируем уточнённое описание
            for cid, docs in cluster_groups.items():
                if len(docs) < 2:
                    continue
                
                # Формируем промпт для LLM
                texts = []
                for d in docs[:5]:  # Берём первые 5 документов из кластера
                    text = d.get("text", "") or ""
                    topic = d.get("topic", "") or ""
                    if topic:
                        texts.append(f"Topic: {topic}\nText: {text[:500]}")
                    else:
                        texts.append(text[:1000])
                
                prompt = "\n\n---\n\n".join(texts)
                
                # Запрос к LLM для уточнения кластера
                description = client.refine_cluster(
                    format_type=self.format_type,
                    cluster_id=cid,
                    documents=prompt,
                    doc_count=len(docs)
                )
                
                if description:
                    for d in docs:
                        d["cluster_description"] = description
            
            logger.info(f"Cluster refinement completed for {self.format_type}: {len(cluster_groups)} clusters")
            
        except Exception as e:
            logger.warning(f"Cluster refinement failed for {self.format_type}: {e}")
        
        return clusters_data
    
    def rollback(self, file_paths: List[str], restore_dir: Path):
        """Откат обработки формата: возврат файлов + очистка БД."""
        from database import DatabaseManager
        
        try:
            db = DatabaseManager()
            db.rollback_by_format(self.format_type, file_paths, restore_dir)
            db.close()
            
            # Возврат файлов из FailedExtraction/ErrorFiles в Sorted/{format}
            for fp in file_paths:
                current_path = Path(fp)
                if not current_path.exists():
                    continue
                
                parent_str = str(current_path.parent).lower().rstrip('\\')
                failed_dir = str(cfg.FAILED_EXTRACTION_DIR).lower().rstrip('\\')
                errors_dir = str(cfg.ERRORS_DIR).lower().rstrip('\\')
                
                if parent_str in (failed_dir, errors_dir):
                    new_path = restore_dir / current_path.name
                    
                    # Обработка коллизий имён
                    counter = 1
                    original_new_path = new_path
                    while new_path.exists():
                        stem = original_new_path.stem
                        ext = original_new_path.suffix
                        new_path = restore_dir / f"{stem}_{counter}{ext}"
                        counter += 1
                    
                    try:
                        import shutil
                        shutil.move(str(current_path), str(new_path))
                        logger.info(f"Возвращён файл: {current_path.name} -> {new_path}")
                    except Exception as e:
                        logger.error(f"Не удалось вернуть файл {current_path.name}: {e}")
        
        except Exception as e:
            logger.error(f"Ошибка отката формата {self.format_type}: {e}")
