# phase_m9_char_ngrams.py
# Символьные n-граммы как CPU-альтернатива embedding-моделям.

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from em_config import EMB_DIR, REPORTS_DIR

MODEL_FILES = {
    'nomic_v15': 'nomic_v15__mixed_cpu512__shard0.npy',
    'nomic_v2': 'nomic_v2__mixed_cpu512__shard0.npy',
    'e5_large': 'e5_large__mixed_cpu512__shard0.npy',
    'rubert_tiny2': 'rubert_tiny2__mixed_cpu512__shard0.npy',
    'rubert_base_dp': 'rubert_base_dp__mixed_cpu512__shard0.npy',
    'sbert_ru': 'sbert_ru__mixed_cpu512__shard0.npy',
    'bge_m3': 'bge_m3__mixed_cpu512__shard0.npy',
    'qwen3e_06b': 'qwen3e_06b__mixed_cpu512__shard0.npy',
    'qwen3e_4b': 'qwen3e_4b__mixed_cpu512_b16__shard0.npy',
}


def build_raw(texts):
    t0 = time.time()
    vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                            n_features=4096, lowercase=True,
                            norm='l2', alternate_sign=False)
    x = vec.transform(texts).toarray().astype(np.float32)
    dt = time.time() - t0
    return x, {'dim': 4096, 'elapsed_s': dt, 'docs_per_s': len(texts) / dt}


def build_tfidf(texts):
    t0 = time.time()
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                          lowercase=True, min_df=2, max_features=120000,
                          sublinear_tf=True)
    sparse = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=512, random_state=20260818)
    x = normalize(svd.fit_transform(sparse)).astype(np.float32)
    dt = time.time() - t0
    return x, {'dim': 512, 'vocab': sparse.shape[1], 'elapsed_s': dt,
               'docs_per_s': len(texts) / dt,
               'svd_variance': float(svd.explained_variance_ratio_.sum())}


def cluster_stats(x, groups):
    import hdbscan

    t0 = time.time()
    cl = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=95,
                         cluster_selection_epsilon=float(np.sqrt(0.6)),
                         core_dist_n_jobs=4)
    labels = cl.fit_predict(x)
    dt = time.time() - t0
    n_cl = len(set(labels) - {-1})
    noise = float((labels == -1).mean())
    return {
        'labels': labels,
        'clusters': n_cl,
        'noise': noise,
        'hdbscan_s': dt,
        'nmi_group': float(normalized_mutual_info_score(groups, labels)),
        'ari_group': float(adjusted_rand_score(groups, labels)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='experiment_models/mixed_sample_2000_noscan.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.sample)
    texts = [str(t or '')[:8000] for t in df['text']]
    groups = df['source_group'].astype(str).to_numpy()

    vectors, metas = {}, {}
    vectors['char_raw_4096'], metas['char_raw_4096'] = build_raw(texts)
    vectors['char_tfidf_svd_512'], metas['char_tfidf_svd_512'] = build_tfidf(texts)
    for name, filename in MODEL_FILES.items():
        vectors[name] = np.load(EMB_DIR / filename)
        metas[name] = {'dim': int(vectors[name].shape[1])}

    labels = {}
    stats = {}
    for name, x in vectors.items():
        print(f'cluster: {name}', flush=True)
        item = cluster_stats(x, groups)
        labels[name] = item.pop('labels')
        stats[name] = item | metas[name]
        print(f"  clusters={item['clusters']} noise={item['noise']:.1%} "
              f"nmi={item['nmi_group']:.3f} ari={item['ari_group']:.3f}", flush=True)

    reference = 'rubert_base_dp'
    pairs = {}
    for name, lab in labels.items():
        if name == reference:
            continue
        pairs[name] = {
            'ari_vs_rubert_base_dp': float(adjusted_rand_score(labels[reference], lab)),
            'nmi_vs_rubert_base_dp': float(normalized_mutual_info_score(labels[reference], lab)),
        }

    out = {'n_docs': len(df), 'sample': args.sample, 'stats': stats, 'pairs': pairs}
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / 'm9_char_ngrams.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'written {path}', flush=True)


if __name__ == '__main__':
    main()
