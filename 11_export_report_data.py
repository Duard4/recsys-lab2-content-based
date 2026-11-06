# -*- coding: utf-8 -*-
"""Крок 11 (ЛР2). Збирання даних для звіту (results/report_data.json)."""
import json
from cb_lib import *

meta = json.load(open(RES / "lab2_meta.json", encoding="utf-8"))
K = meta["k"]
u, c, it, ut, ct, cl, uid_idx, cid_idx, sd = load_context(meta["cluster_column"])
POOLS = ["P1", "P2", "P3"] + [f"P4_{m}" for m in POOL_M]
rd = dict(meta=meta)
read = lambda f: pd.read_csv(RES / f, encoding=ENC)
nan2none = lambda d: json.loads(json.dumps(d, ensure_ascii=False, default=float).replace("NaN", "null"))

size = read("stats_pool_sizes.csv")
rd["sizes"] = {nm: {g: dict(mean=r["mean"], min=int(r["min"]), max=int(r["max"]), median=r["median"], share=r["share_catalog_mean"], empty=int(r["empty"]))
                    for g, r in size[size.pool == nm].set_index("group").iterrows()} for nm in POOLS}
for key, f in [("dist_pool", "stats_cluster_distribution_pools.csv"), ("dist_top10", "stats_cluster_distribution_top10.csv")]:
    d = read(f)
    rd[key] = {nm: {g: [[r["mean"], int(r["min"]), int(r["max"]), float(r["share_users_nonzero"])] for _, r in d[(d.pool == nm) & (d.group == g)].sort_values("cluster").iterrows()]
                    for g in GROUPS} for nm in POOLS}
rd["clusters"] = read("stats_cluster_profiles.csv").to_dict("records")
q = read("stats_top10_quality.csv")
rd["quality"] = {nm: {g: r.to_dict() for g, r in q[q.pool == nm].set_index("group").iterrows()} for nm in POOLS}
ho = read("stats_holdout.csv")
rd["holdout"] = {m: {g: r.to_dict() for g, r in ho[ho.method == m].set_index("group").iterrows()} for m in ho.method.unique()}
rd["sens"] = read("stats_sensitivity_clustering.csv").to_dict("records")
rd["thrB"] = read("stats_strategyB_threshold.csv").to_dict("records")

# --- довжини списків та перекриття ТОП-10 між стратегіями
rec = read("recommendations_top10.csv")
grp = u.set_index("user_id")["activity_group"]
ln = rec.groupby(["pool", "user_id"]).size().unstack(0).fillna(0)
rd["short_lists"] = {nm: int((ln[nm] < 10).sum()) for nm in POOLS}
rd["short_lists_by_group"] = {nm: {g: int(((ln[nm] < 10) & (grp.loc[ln.index] == g)).sum()) for g in GROUPS} for nm in ["P1"]}
sets = {nm: rec[rec.pool == nm].groupby("user_id")["course_id"].apply(set) for nm in POOLS}
def overlap(a, b):
    out = {}
    for g in GROUPS:
        ids = [x for x in grp[grp == g].index if x in sets[a].index and x in sets[b].index]
        out[g] = float(np.mean([len(sets[a][x] & sets[b][x]) / len(sets[a][x] | sets[b][x]) for x in ids])) if ids else None
    return out
rd["overlap"] = {f"{a}|{b}": overlap(a, b) for a, b in [("P1", "P2"), ("P1", "P3"), ("P1", "P4_100"), ("P2", "P3"), ("P3", "P4_100"), ("P2", "P4_100")]}

# --- перевірка виключення історії (незалежно від 07)
hist = set(zip(sd["user_id"], sd["course_id"]))
bad = sum((a, b) in hist for a, b in zip(rec["user_id"], rec["course_id"]))
pb = 0
for nm in POOLS:
    df = read(f"pools/pool_{nm}.csv")
    hs = sd.groupby("user_id")["course_id"].apply(set)
    for uid, s in zip(df["user_id"], df["course_ids"]):
        if isinstance(s, str) and uid in hs.index:
            pb += len(set(s.split(";")) & hs[uid])
rd["checks"] = dict(rec_rows=len(rec), rec_violations=int(bad), pool_violations=int(pb), users=len(u), n_history_pairs=len(hist))

# --- приклади користувачів
ex = json.load(open(RES / "example_users.json", encoding="utf-8"))
ex["звичайні"] = u[u.activity_group == "звичайні"].sort_values("n_courses").iloc[(u.activity_group == "звичайні").sum() // 2]["user_id"]
clu = dict(zip(c.course_id, cl))
ttl = dict(zip(c.course_id, c.title))
seeds = read("new_users_seed_courses.csv")
exs = {}
for g, uid in ex.items():
    row = u[u.user_id == uid].iloc[0]
    h = sd[sd.user_id == uid].copy()
    h["cl"] = h.course_id.map(clu)
    sh = h.groupby("cl")["n_events"].sum().sort_values(ascending=False)
    top_h = h.sort_values("strength", ascending=False).head(3)
    e = dict(user_id=uid, group=g, faculty=row.faculty, interests=row.interests, career=row.career_goal, fmt=row.preferred_format,
             n_courses=len(h), n_events=int(h.n_events.sum()), hist_clusters=[[int(k), int(v)] for k, v in sh.items()],
             top_hist=[[r.course_id, ttl[r.course_id], int(r.cl)] for r in top_h.itertuples()],
             recs={nm: rec[(rec.user_id == uid) & (rec.pool == nm)].sort_values("rank")[["course_id", "title", "cluster", "score"]].to_dict("records")
                   for nm in ["P1", "P2", "P3", "P4_100"]},
             pool_sizes={nm: int(size_of) for nm, size_of in
                         [(nm, len(str(read(f"pools/pool_{nm}.csv").set_index("user_id").loc[uid, "course_ids"]).split(";")) if isinstance(read(f"pools/pool_{nm}.csv").set_index("user_id").loc[uid, "course_ids"], str) else 0) for nm in ["P1", "P2", "P3", "P4_100"]]})
    if g == "нові (0)":
        e["seeds"] = seeds[seeds.user_id == uid][["seed_course_id", "title", "seed_score"]].to_dict("records")
    exs[g] = e
rd["examples"] = exs
rd["new_users_strict_empty"] = int((u.activity_group == "нові (0)").sum())
rd["catalog"] = dict(courses=len(c), users=len(u), new_courses=int((c.popularity_group == "нові (0)").sum()))
json.dump(nan2none(rd), open(RES / "report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(rd["checks"]), rd["short_lists"], rd["overlap"])
print({g: (e["user_id"], e["n_courses"], e["pool_sizes"]) for g, e in exs.items()})
