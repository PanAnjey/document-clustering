# stages/stage3_clustering.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Этап 3 - Кластеризация и уточнение (LLM).

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np

from config import cfg
from logger_utils import logger


def _cluster_group(docs: list, analyzer) -> list:
    """Кластеризация группы документов."""
    if len(docs) < 2:
        if docs:
            docs[0].setdefault("sub_cluster", 0)
        return docs

    vectors = []
    for d in docs:
        if "combined_cluster" in d and "text_embedding" in d and "image_embedding" in d:
            text_emb = d.get("text_embedding")
            img_emb = d.get("image_embedding")
            
            if text_emb is not None and img_emb is not None:
                v = (text_emb + img_emb) / 2
            elif text_emb is not None:
                v = text_emb
            elif img_emb is not None:
                v = img_emb
            else:
                continue
        
        elif "text_embedding" in d and d["text_embedding"] is not None:
            v = d["text_embedding"]
        
        elif "image_embedding" in d and d["image_embedding"] is not None:
            v = d["image_embedding"]
        
        else:
            continue

        vectors.append(v)

    valid = [(i, v) for i, v in enumerate(vectors) if v is not None]
    
    if len(valid) < 2:
        for d in docs:
            d["sub_cluster"] = 0
        return docs

    emb_matrix = np.array([v for _, v in valid])
    emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    
    labels = analyzer.perform_clustering(emb_matrix)

    for idx, (_, v) in enumerate(valid):
        docs[idx].pop("sub_cluster", None)
    
    for idx, (orig_i, _) in enumerate(valid):
        docs[orig_i]["sub_cluster"] = int(labels[idx])

    no_emb = [i for i, v in enumerate(vectors) if v is None]
    for i in no_emb:
        docs[i]["sub_cluster"] = 0

    return docs


async def run_stage_3_cluster_refine(data: List[Dict]) -> List[Dict]:
    """Выполняет этап 3: кластеризация и уточнение через LLM."""
    from cluster_engine import ClusterAnalyzer
    from database import DatabaseManager

    try:
        db = DatabaseManager()
        db_data = db.get_all_data_with_embeddings()
        db.close()
        
        if db_data:
            logger.info(f"Загружено {len(db_data)} документов из БД для кластеризации.")
            data = db_data
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные из БД, используем JSON: {e}")

    analyzer = ClusterAnalyzer(similarity_threshold=cfg.SIMILARITY_THRESHOLD)

    def _win_path(p):
        """Нормализация пути для Windows."""
        if sys.platform == 'win32':
            s = str(p).replace('/', '\\\\')
            if len(s) >= 260 and not s.startswith('\\\\?\\\\'):
                s = '\\\\\\\\?\\\\' + s
            return s
        return str(p)

    has_topic = any("topic" in d for d in data)
    has_doc_type = any("doc_type" in d for d in data)

    def _get_doc_vector(d):
        """Получение вектора документа."""
        if "combined_cluster" in d and "text_embedding" in d and "image_embedding" in d:
            v = (d["text_embedding"] + d["image_embedding"]) / 2
        elif "text_embedding" in d:
            v = d["text_embedding"]
        elif "image_embedding" in d:
            v = d["image_embedding"]
        else:
            return None
        
        return v

    # --- Уровень 1: ТЕМА (верхнеуровневая кластеризация) ---
    if has_topic:
        logger.info("Уровень 1: Кластеризация по ТЕМАМ (LLM-метки)...")
        
        topic_groups = defaultdict(list)
        no_topic = []
        
        for d in data:
            t = d.get("topic")
            if t:
                topic_groups[t].append(d)
            else:
                no_topic.append(d)

        topic_names = sorted(topic_groups.keys())
        topic_map = {}
        
        for i, name in enumerate(topic_names):
            topic_map[name] = i
            for d in topic_groups[name]:
                d["topic_cluster"] = i

        logger.info(f"Темы (LLM): {len(topic_names)} исходных групп")

        # Слияние LLM-групп по близости центроидов эмбеддингов
        topic_centroids = {}
        
        for name, docs in topic_groups.items():
            vectors = [_get_doc_vector(d) for d in docs]
            valid_vectors = [v for v in vectors if v is not None]
            
            if len(valid_vectors) > 0:
                centroid = np.mean(valid_vectors, axis=0)
                topic_centroids[name] = centroid

        # Вычисление матрицы сходств между темами
        logger.info("Вычисление сходства между темами...")
        
        if len(topic_centroids) >= 2:
            from cluster_engine import ClusterAnalyzer
            
            centroids_list = list(topic_centroids.values())
            names_list = list(topic_centroids.keys())
            
            analyzer_temp = ClusterAnalyzer(similarity_threshold=cfg.SIMILARITY_THRESHOLD * 0.8)
            sim_matrix = analyzer_temp.compute_similarity_matrix_gpu(np.array(centroids_list))
            
            # Группировка похожих тем
            merged_topics = {}
            used = set()
            
            for i, name1 in enumerate(names_list):
                if name1 in used:
                    continue
                
                merged_topics[name1] = [name1]
                used.add(name1)
                
                for j, name2 in enumerate(names_list[i+1:], start=i+1):
                    if sim_matrix[i][j] > cfg.SIMILARITY_THRESHOLD * 0.8:
                        merged_topics[name1].append(name2)
                        used.add(name2)

        logger.info(f"Темы после слияния: {len(merged_topics)} групп")

    # --- Уровень 2: ПОДТЕМЫ (внутри каждой темы) ---
    if has_topic:
        logger.info("Уровень 2: Кластеризация по ПОДТЕМАМ...")
        
        for topic_name, docs in topic_groups.items():
            sub_clusters = _cluster_group(docs, analyzer)
            
            # Присвоение подкластеров
            cluster_ids = sorted(set(d.get("sub_cluster", 0) for d in sub_clusters))
            
            for i, cluster_id in enumerate(cluster_ids):
                if cluster_id == 0:
                    continue
                
                logger.info(f"Тема '{topic_name}': кластер {cluster_id} содержит {sum(1 for d in sub_clusters if d.get('sub_cluster') == cluster_id)} документов")

    # --- Уровень 3: УТОЧНЕНИЕ (LLM) ---
    if cfg.CLUSTER_REFINEMENT_ENABLED:
        logger.info("Уровень 3: Уточнение кластеров через LLM...")
        
        from llm_summarizer import summarize_cluster
        
        for topic_name, docs in topic_groups.items():
            cluster_ids = sorted(set(d.get("sub_cluster", 0) for d in docs))
            
            for cluster_id in cluster_ids:
                if cluster_id == 0:
                    continue
                
                cluster_docs = [d for d in docs if d.get("sub_cluster") == cluster_id]
                
                if len(cluster_docs) >= cfg.MIN_CLUSTER_SIZE // 5:
                    try:
                        summary = summarize_cluster(cluster_docs, topic_name, cluster_id)
                        
                        for doc in cluster_docs:
                            doc["cluster_summary"] = summary
                        
                        logger.info(f"Кластер {topic_name}-{cluster_id}: сгенерировано описание")
                    
                    except Exception as e:
                        logger.warning(f"Ошибка уточнения кластера {topic_name}-{cluster_id}: {e}")

    # --- Финальная группировка и статистика ---
    final_clusters = defaultdict(list)
    
    for d in data:
        topic = d.get("topic", "Без темы")
        sub_cluster = d.get("sub_cluster", 0)
        
        if sub_cluster == 0 and has_topic:
            # Документы без подтемы в рамках темы
            key = f"{topic}-общее"
        else:
            key = f"{topic}-{sub_cluster}"
        
        final_clusters[key].append(d)

    logger.info(f"\nИтого кластеров: {len(final_clusters)}")
    
    # Статистика по размерам кластеров
    cluster_sizes = [len(docs) for docs in final_clusters.values()]
    if cluster_sizes:
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        
        logger.info(f"Средний размер кластера: {avg_size:.1f}")
        logger.info(f"Максимальный размер: {max_size}")
        logger.info(f"Минимальный размер: {min_size}")

    # Возврат результатов
    result = []
    for cluster_name, docs in final_clusters.items():
        result.append({
            "cluster_name": cluster_name,
            "count": len(docs),
            "documents": docs,
        })

    return result
