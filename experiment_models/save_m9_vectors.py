# save_m9_vectors.py
# Сохранение символьных векторов M9 для повторного анализа.

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from em_config import EMB_DIR


df = pd.read_csv('experiment_models/mixed_sample_2000_noscan.csv')
texts = [str(t or '')[:8000] for t in df['text']]
vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=4096,
                        lowercase=True, norm='l2', alternate_sign=False)
raw = vec.transform(texts).toarray().astype(np.float32)
np.save(EMB_DIR / 'char_raw_4096__mixed_cpu__shard0.npy', raw)

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True,
                      min_df=2, max_features=120000, sublinear_tf=True)
sparse = vec.fit_transform(texts)
svd = TruncatedSVD(n_components=512, random_state=20260818)
dense = normalize(svd.fit_transform(sparse)).astype(np.float32)
np.save(EMB_DIR / 'char_tfidf_svd_512__mixed_cpu__shard0.npy', dense)
print(raw.nbytes / 1024**2, dense.nbytes / 1024**2)
