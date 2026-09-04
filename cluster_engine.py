# cluster_engine.py
# Версия: 3.0
# Дата: 2026-06-06
# Описание: Кластерный анализ с HDBSCAN на GPU.

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict

try:
    from hdbscan import HDBSCAN
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

from config import cfg
from logger_utils import logger


class ClusterAnalyzer:
    def __init__(self, similarity_threshold: float = 0.70):
        self.threshold = similarity_threshold
        self.distance_threshold = 1.0 - similarity_threshold
        self.device = cfg.EMB_GPU_DEVICE
        self.chunk_size = cfg.DIST_MATRIX_CHUNK

    def compute_similarity_matrix_gpu(self, embeddings: np.ndarray) -> np.ndarray:
        """Вычисление матрицы сходств на GPU (чанковое)."""
        if len(embeddings) == 0:
            return np.array([])

        n = len(embeddings)
        logger.info(f"Computing similarity matrix for {n} vectors on GPU (chunk={self.chunk_size})...")

        tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

        if n <= self.chunk_size:
            with torch.no_grad():
                similarity_matrix = torch.mm(tensor, tensor.transpose(0, 1)).cpu().numpy()
            return similarity_matrix

        similarity_matrix = np.empty((n, n), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                chunk = tensor[start:end]
                sim_block = torch.mm(chunk, tensor.transpose(0, 1)).cpu().numpy()
                similarity_matrix[start:end, :] = sim_block

                if (start // self.chunk_size) % 5 == 0:
                    logger.info(f"  Similarity matrix: rows {start}-{end}/{n}")

        logger.info("Similarity matrix computed.")
        return similarity_matrix

    def perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Кластеризация через HDBSCAN (без O(N²) матрицы расстояний)."""
        n = len(embeddings)
        if _HDBSCAN_AVAILABLE:
            logger.info(f"Performing HDBSCAN clustering ({n} docs, metric=cosine)...")

            clustering = HDBSCAN(
                metric='cosine',
                min_cluster_size=cfg.MIN_CLUSTER_SIZE,
                cluster_selection_epsilon=self.distance_threshold,
                core_dist_n_jobs=-1,
            )

            labels = clustering.fit_predict(embeddings)
            noise_count = np.sum(labels == -1)
            unique_clusters = len(set(labels)) - (1 if noise_count > 0 else 0)
            logger.info(f"HDBSCAN done: {unique_clusters} clusters, {noise_count} noise points")
            return labels
        else:
            logger.warning("HDBSCAN not available, falling back to AgglomerativeClustering...")
            distance_matrix = self._compute_distance_matrix(embeddings)
            return self._agglomerative_clustering(distance_matrix)

    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback: вычисление матрицы расстояний для AgglomerativeClustering."""
        n = len(embeddings)
        sim = self.compute_similarity_matrix_gpu(embeddings)
        return (1.0 - sim).astype(np.float32)

    def _agglomerative_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fallback: AgglomerativeClustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        logger.info("Performing Agglomerative Clustering (CPU)...")

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=self.distance_threshold
        )

        return clustering.fit_predict(distance_matrix)

    def analyze_clusters(self, labels: np.ndarray, file_data: List[Dict]) -> List[Dict]:
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(file_data[idx])

        cluster_stats = []

        for label, files in clusters.items():
            if len(files) == 0:
                continue

            formats = defaultdict(int)
            for f in files:
                formats[f['type']] += 1

            cluster_stats.append({
                "id": label,
                "count": len(files),
                "formats": dict(formats),
                "files": files
            })

        cluster_stats.sort(key=lambda x: x['count'], reverse=True)
        return cluster_stats

    def get_top_clusters(self, cluster_stats: List[Dict], top_n: int) -> List[Dict]:
        return cluster_stats[:top_n]