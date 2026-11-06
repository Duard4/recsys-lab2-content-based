# -*- coding: utf-8 -*-
"""Крок 7 (ЛР2). Пули №1–№4 за стратегіями А, Б, В та ТОП-10 контентних рекомендацій для всіх користувачів."""
import json
from cb_lib import *

CLUSTER = "cluster_C1"      # кластеризація курсів з ЛР1: косинусна за TF-IDF описів (k = 10), охоплює всі 3000 курсів
u, c, it, ut, ct, cl, uid_idx, cid_idx, sd = load_context(CLUSTER)
U, C = len(u), len(c)
S, tf, T = content_similarity(c)
NB = neighbours(S)
print("кластери курсів:", np.bincount(cl), "| сусідів на курс:", NB.sum(1).mean().round(2))

# --- історія: ВСІ взаємодії (будь-якого типу, у т.ч. негативні оцінки) виключаються з рекомендацій
ui, ci = sd["user_id"].map(uid_idx).to_numpy(), sd["course_id"].map(cid_idx).to_numpy()
W = sp.csr_matrix((sd["strength"].to_numpy(), (ui, ci)), shape=(U, C))
E = sp.csr_matrix((sd["n_events"].to_numpy(float), (ui, ci)), shape=(U, C))
hist = np.zeros((U, C), bool)
hist[ui, ci] = True

# --- холодний старт для «нових» користувачів: насіння з заявлених інтересів (не вважається взаємодією!)
cats = sorted(c["category"].unique())
cat_idx = {x: i for i, x in enumerate(cats)}
new_ids = u.loc[u["activity_group"] == "нові (0)", "user_id"].tolist()
seeds = seeds_for_new_users(u, c, T, tf, sd, cats, cat_idx, uid_idx, new_ids)
Wd, Ed = W.toarray(), E.toarray()
for uid, (top, sc, pi) in seeds.items():
    i = uid_idx[uid]
    Wd[i, top] = sc
    Ed[i, top] = sc / sc.sum() * len(top)
Wseed, Eseed = sp.csr_matrix(Wd), sp.csr_matrix(Ed)

pools, recs, aux = build_pools(Wseed, Eseed, S, NB, cl, hist)

# --- перевірка головної вимоги: жодного курсу з історії у пулах і рекомендаціях
bad = 0
for nm in pools:
    for i in range(U):
        bad += int(hist[i, pools[nm][i]].sum()) + int(hist[i, recs[nm][i][0]].sum())
print("порушень виключення історії:", bad)
assert bad == 0

# --- запис пулів (одна таблиця на пул) і списків рекомендацій
(RES / "pools").mkdir(exist_ok=True)
cid = c["course_id"].to_numpy()
uid_arr = u["user_id"].to_numpy()
grp = u["activity_group"].to_numpy()
for nm in pools:
    rows = [dict(user_id=uid_arr[i], activity_group=grp[i], pool_size=len(pools[nm][i]), course_ids=";".join(cid[pools[nm][i]])) for i in range(U)]
    pd.DataFrame(rows).to_csv(RES / "pools" / f"pool_{nm}.csv", index=False, encoding=ENC)
title = c["title"].to_numpy()
cat_ = c["category"].to_numpy()
rows = []
for nm in pools:
    for i in range(U):
        idx, sc = recs[nm][i]
        for r, (j, s_) in enumerate(zip(idx, sc), 1):
            rows.append((uid_arr[i], grp[i], nm, r, cid[j], title[j], int(cl[j]), cat_[j], round(float(s_), 4)))
pd.DataFrame(rows, columns=["user_id", "activity_group", "pool", "rank", "course_id", "title", "cluster", "category", "score"]).to_csv(
    RES / "recommendations_top10.csv", index=False, encoding=ENC)

# --- допоміжні таблиці
pd.DataFrame(dict(course_id=cid, cluster=cl, category=cat_)).to_csv(RES / "course_clusters_used.csv", index=False, encoding=ENC)
seed_rows = [(uid, cid[t][r], c["title"].iloc[t[r]], round(float(sc[r]), 3)) for uid, (t, sc, pi) in seeds.items() for r in range(len(t))]
pd.DataFrame(seed_rows, columns=["user_id", "seed_course_id", "title", "seed_score"]).to_csv(RES / "new_users_seed_courses.csv", index=False, encoding=ENC)
np.savez_compressed(RES / "aux_pools.npz", Eclu=aux["Eclu"], cl=cl)
meta = dict(cluster_column=CLUSTER, k=int(cl.max() + 1), tau=TAU, k_neigh=K_NEIGH, pool_m=POOL_M, n_seed=N_SEED, n_users=U, n_courses=C,
            cluster_sizes=np.bincount(cl).tolist(), groups=u["activity_group"].value_counts().to_dict(), violations=int(bad))
json.dump(meta, open(RES / "lab2_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for nm in ["P1", "P2", "P3", "P4_50"]:
    s = pd.Series([len(p) for p in pools[nm]]).groupby(grp).mean().round(1).to_dict()
    print(nm, s)
