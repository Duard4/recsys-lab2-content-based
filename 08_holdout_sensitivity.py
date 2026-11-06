# -*- coding: utf-8 -*-
"""Крок 8 (ЛР2). Додатковий аналіз: (а) відкладена вибірка — чи потрапляють «майбутні» курси користувача в пули/ТОП-10;
(б) чутливість до вибору кластеризації курсів з ЛР1 (C1 / C3 / C4)."""
import json
from cb_lib import *

meta = json.load(open(RES / "lab2_meta.json", encoding="utf-8"))
rng = np.random.default_rng(7)
POOLS = ["P1", "P2", "P3"] + [f"P4_{m}" for m in POOL_M]

def prepare(cluster_col):
    u, c, it, ut, ct, cl, uid_idx, cid_idx, sd = load_context(cluster_col)
    S, tf, T = content_similarity(c)
    return dict(u=u, c=c, it=it, ut=ut, ct=ct, cl=cl, uid_idx=uid_idx, cid_idx=cid_idx, sd=sd, S=S, tf=tf, T=T, NB=neighbours(S))

def mats(ctx, sd):
    U, C = len(ctx["u"]), len(ctx["c"])
    ui, ci = sd["user_id"].map(ctx["uid_idx"]).to_numpy(), sd["course_id"].map(ctx["cid_idx"]).to_numpy()
    W = sp.csr_matrix((sd["strength"].to_numpy(), (ui, ci)), shape=(U, C))
    E = sp.csr_matrix((sd["n_events"].to_numpy(float), (ui, ci)), shape=(U, C))
    H = np.zeros((U, C), bool); H[ui, ci] = True
    return W, E, H

# ======================================================= (а) відкладена вибірка
ctx = prepare(meta["cluster_column"])
u, c, it, sd, cl = ctx["u"], ctx["c"], ctx["it"], ctx["sd"], ctx["cl"]
U, C = len(u), len(c)
first = it.groupby(["user_id", "course_id"])["timestamp"].min().rename("t").reset_index()
sd2 = sd.merge(first, on=["user_id", "course_id"], how="left")
sd2["t"] = sd2["t"].fillna(sd2["t"].max())
MIN_H = 4
test_pairs, keep = set(), np.ones(len(sd2), bool)
elig = []
for uid, g in sd2.groupby("user_id"):
    if len(g) >= MIN_H:
        g = g.sort_values("t")
        n_test = max(1, int(round(0.2 * len(g))))
        for idx in g.index[-n_test:]:
            keep[idx] = False
            test_pairs.add((uid, g.loc[idx, "course_id"]))
        elig.append(uid)
train = sd2[keep]
W, E, H = mats(ctx, train)
elig_idx = np.array([ctx["uid_idx"][x] for x in elig])
mask = np.zeros(U, bool); mask[elig_idx] = True
Wm, Em = sp.diags(mask.astype(float)) @ W, sp.diags(mask.astype(float)) @ E
pools, recs, _ = build_pools(Wm.tocsr(), Em.tocsr(), ctx["S"], ctx["NB"], cl, H)
test = {ctx["uid_idx"][a]: set() for a, _ in test_pairs}
for a, b in test_pairs:
    test[ctx["uid_idx"][a]].add(ctx["cid_idx"][b])
# базові лінії: популярність (за навчальними даними) та випадковий вибір серед «нових для користувача»
pop_train = np.asarray((W > 0).sum(0)).ravel()
grp = u["activity_group"].to_numpy()
rows = []
for nm in POOLS + ["POP", "RAND"]:
    acc = {g: dict(hr=[], p=[], r=[], pr=[], sz=[]) for g in GROUPS}
    for i in elig_idx:
        t = test[i]
        if nm in pools:
            top, pl = recs[nm][i][0], pools[nm][i]
        else:
            cand = np.where(~H[i])[0]
            top = cand[np.argsort(-pop_train[cand], kind="stable")[:10]] if nm == "POP" else rng.choice(cand, 10, replace=False)
            pl = cand
        hit = len(t & set(top.tolist()))
        a = acc[grp[i]]
        a["hr"].append(hit > 0); a["p"].append(hit / 10); a["r"].append(hit / len(t))
        a["pr"].append(len(t & set(pl.tolist())) / len(t)); a["sz"].append(len(pl) / C)
    for g in GROUPS:
        a = acc[g]
        if a["hr"]:
            rows.append(dict(method=nm, group=g, n_users=len(a["hr"]), pool_recall=np.mean(a["pr"]), pool_share_of_catalog=np.mean(a["sz"]),
                             HR10=np.mean(a["hr"]), P10=np.mean(a["p"]), R10=np.mean(a["r"])))
ho = pd.DataFrame(rows)
ho.to_csv(RES / "stats_holdout.csv", index=False, encoding=ENC)
print(ho.round(3).query("group in ['активні','пасивні','звичайні']").to_string())

# ======================================================= (б) чутливість до кластеризації
sens = []
for col in ["cluster_C1", "cluster_C3", "cluster_C4"]:
    cx = ctx if col == "cluster_C1" else prepare(col)
    cl2 = cx["cl"]
    W2, E2, H2 = mats(cx, cx["sd"])
    cats = sorted(cx["c"]["category"].unique()); cat_idx = {x: i for i, x in enumerate(cats)}
    new_ids = cx["u"].loc[cx["u"]["activity_group"] == "нові (0)", "user_id"].tolist()
    seeds = seeds_for_new_users(cx["u"], cx["c"], cx["T"], cx["tf"], cx["sd"], cats, cat_idx, cx["uid_idx"], new_ids)
    Wd, Ed = W2.toarray(), E2.toarray()
    for uid, (top, sc, pi) in seeds.items():
        i = cx["uid_idx"][uid]; Wd[i, top] = sc; Ed[i, top] = sc / sc.sum() * len(top)
    pl2, rc2, _ = build_pools(sp.csr_matrix(Wd), sp.csr_matrix(Ed), cx["S"], cx["NB"], cl2, H2, pool_m=[100])
    K2 = int(cl2.max() + 1)
    for g in GROUPS:
        ids = np.where(grp == g)[0]
        sens.append(dict(clustering=col, k=K2, group=g, P1=np.mean([len(pl2["P1"][i]) for i in ids]), P2=np.mean([len(pl2["P2"][i]) for i in ids]),
                         P3=np.mean([len(pl2["P3"][i]) for i in ids]), clusters_P2_pool=np.mean([len(set(cl2[pl2["P2"][i]])) for i in ids]),
                         clusters_top10_P4=np.mean([len(set(cl2[rc2["P4_100"][i][0]])) for i in ids]),
                         share_P2_catalog=np.mean([len(pl2["P2"][i]) for i in ids]) / C))
sn = pd.DataFrame(sens)
sn.to_csv(RES / "stats_sensitivity_clustering.csv", index=False, encoding=ENC)
print(sn.round(2).to_string())
