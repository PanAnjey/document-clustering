# stages/stage3_clustering.py
# Версия: 5.1
# Дата: 2026-06-27
# Описание: Этап 3 - Кластеризация и уточнение (LLM).

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

import numpy as np

from config import cfg
from logger_utils import logger


def _get_doc_vector(d: Dict) -> np.ndarray | None:
    """Получение вектора документа с учётом приоритета Combined > Text > Image."""
    text_emb = d.get("text_embedding")
    img_emb = d.get("image_embedding")

    if text_emb is not None and img_emb is not None:
        return (np.asarray(text_emb) + np.asarray(img_emb)) / 2
    if text_emb is not None:
        return np.asarray(text_emb)
    if img_emb is not None:
        return np.asarray(img_emb)
    return None


def _cluster_group(docs: list, analyzer) -> list:
    """Кластеризация группы документов по эмбеддингам."""
    if len(docs) < 2:
        if docs:
            docs[0].setdefault("sub_cluster", 0)
        return docs

    vectors = []
    doc_indices = []
    for i, d in enumerate(docs):
        v = _get_doc_vector(d)
        if v is not None:
            vectors.append(v)
            doc_indices.append(i)

    if len(vectors) < 2:
        for d in docs:
            d["sub_cluster"] = 0
        return docs

    emb_matrix = np.array(vectors)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    labels = analyzer.perform_clustering(emb_matrix)

    for idx, label in enumerate(labels):
        docs[doc_indices[idx]]["sub_cluster"] = int(label)

    for i, d in enumerate(docs):
        if i not in doc_indices:
            d.setdefault("sub_cluster", 0)

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

    has_topic = any(d.get("topic") for d in data)

    # --- Уровень 1: ТЕМА (верхнеуровневая группировка) ---
    topic_groups: Dict[str, List[Dict]] = defaultdict(list)
    no_topic: List[Dict] = []

    for d in data:
        t = d.get("topic")
        if t:
            topic_groups[str(t)].append(d)
        else:
            no_topic.append(d)

    if has_topic:
        logger.info(f"Уровень 1: Кластеризация по ТЕМАМ (LLM-метки)... {len(topic_groups)} тем")

        # Слияние LLM-групп по близости центроидов эмбеддингов
        topic_centroids = {}
        for name, docs in topic_groups.items():
            vectors = [_get_doc_vector(d) for d in docs]
            valid_vectors = [v for v in vectors if v is not None]
            if valid_vectors:
                topic_centroids[name] = np.mean(valid_vectors, axis=0)

        merged_topics: Dict[str, str] = {}  # исходная тема -> имя группы
        if len(topic_centroids) >= 2:
            from cluster_engine import ClusterAnalyzer
            centroids_list = list(topic_centroids.values())
            names_list = list(topic_centroids.keys())
            analyzer_temp = ClusterAnalyzer(similarity_threshold=cfg.SIMILARITY_THRESHOLD * 0.8)
            sim_matrix = analyzer_temp.compute_similarity_matrix_gpu(np.array(centroids_list))
            used = set()
            for i, name1 in enumerate(names_list):
                if name1 in used:
                    continue
                merged_topics[name1] = name1
                used.add(name1)
                for j, name2 in enumerate(names_list[i + 1:], start=i + 1):
                    if name2 in used:
                        continue
                    if sim_matrix[i][j] > cfg.SIMILARITY_THRESHOLD * 0.8:
                        merged_topics[name2] = name1
                        used.add(name2)
        else:
            for name in topic_centroids:
                merged_topics[name] = name

        # Перегруппируем документы по объединённым темам
        merged_groups: Dict[str, List[Dict]] = defaultdict(list)
        for name, docs in topic_groups.items():
            group_name = merged_topics.get(name, name)
            merged_groups[group_name].extend(docs)
        if no_topic:
            merged_groups["Без темы"].extend(no_topic)
        topic_groups = merged_groups

        logger.info(f"Темы после слияния: {len(topic_groups)} групп")
    else:
        # Без LLM-тем все документы в одну группу
        topic_groups["Без темы"] = list(data)

    # --- Уровень 2: ПОДТЕМЫ (внутри каждой темы) ---
    logger.info("Уровень 2: Кластеризация по ПОДТЕМАМ...")
    for topic_name, docs in topic_groups.items():
        _cluster_group(docs, analyzer)
        cluster_ids = sorted(set(d.get("sub_cluster", 0) for d in docs))
        for cluster_id in cluster_ids:
            if cluster_id == 0:
                continue
            count = sum(1 for d in docs if d.get("sub_cluster") == cluster_id)
            logger.info(f"Тема '{topic_name}': кластер {cluster_id} содержит {count} документов")

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
                if len(cluster_docs) >= max(2, cfg.MIN_CLUSTER_SIZE // 5):
                    try:
                        summary = summarize_cluster(cluster_docs, topic_name, cluster_id)
                        for doc in cluster_docs:
                            doc["cluster_summary"] = summary
                        logger.info(f"Кластер {topic_name}-{cluster_id}: описание сгенерировано")
                    except Exception as e:
                        logger.warning(f"Ошибка уточнения кластера {topic_name}-{cluster_id}: {e}")

    # --- Финальная группировка и статистика ---
    final_clusters = defaultdict(list)

    for topic_name, docs in topic_groups.items():
        for d in docs:
            sub_cluster = d.get("sub_cluster", 0)
            if sub_cluster == 0:
                key = f"{topic_name}-общее"
            else:
                key = f"{topic_name}-{sub_cluster}"
            final_clusters[key].append(d)

    logger.info(f"\nИтого кластеров: {len(final_clusters)}")

    cluster_sizes = [len(docs) for docs in final_clusters.values()]
    if cluster_sizes:
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        logger.info(f"Средний размер кластера: {avg_size:.1f}")
        logger.info(f"Максимальный размер: {max_size}")
        logger.info(f"Минимальный размер: {min_size}")

    result = []
    for cluster_name, docs in final_clusters.items():
        result.append({
            "cluster_name": cluster_name,
            "count": len(docs),
            "documents": docs,
        })

    return result
