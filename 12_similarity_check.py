# -*- coding: utf-8 -*-
"""Крок 12 (ЛР2). Перевірка контентної схожості: частка 10 найближчих сусідів курсу з тієї самої прихованої тематичної області."""
import json
from cb_lib import *
u, c, it, X, R, ut, ct = load_clean()
n = len(c)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
T = tf.fit_transform(c.text_clean)
S_t = (T @ T.T).toarray()
Bk, _ = sets_to_binary(c.keywords)
S_k = 1 - jaccard_distance_binary(Bk)
S_a = np.zeros((n, n))
for col in ["category", "language", "format", "assessment"]:
    v = c[col].fillna("?").to_numpy(dtype=object)
    S_a += (v[:, None] == v[None, :])
d = c.difficulty.fillna(c.difficulty.median()).to_numpy(float)
S_a += 1 - np.abs(d[:, None] - d[None, :]) / 4
S_a /= 5
area = ct.loc[c.course_id, "area"].to_numpy(dtype=object)
same = area[:, None] == area[None, :]
def prec(S, k=10):
    S = S.copy(); np.fill_diagonal(S, -1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return float(same[np.arange(n)[:, None], idx].mean())
res = {"tfidf": prec(S_t), "keywords_jaccard": prec(S_k), "attributes": prec(S_a), "hybrid_0.5_0.3_0.2": prec(0.5 * S_t + 0.3 * S_k + 0.2 * S_a),
       "hybrid_0.6_0.25_0.15": prec(0.6 * S_t + 0.25 * S_k + 0.15 * S_a)}
json.dump(res, open(RES / "similarity_check.json", "w"), indent=1)
print(res)
