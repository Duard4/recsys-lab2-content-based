# -*- coding: utf-8 -*-
"""Крок 9 (ЛР2). Розподіл об'єктів пулів і ТОП-10 за кластерами (середнє / мін / макс) для активних, пасивних, нових; діаграми."""
import json
import matplotlib.pyplot as plt
from cb_lib import *

meta = json.load(open(RES / "lab2_meta.json", encoding="utf-8"))
K = meta["k"]
u, c, it, ut, ct, cl, uid_idx, cid_idx, sd = load_context(meta["cluster_column"])
U, C = len(u), len(c)
POOLS = ["P1", "P2", "P3"] + [f"P4_{m}" for m in POOL_M]
PNAME = {"P1": "Пул №1 (А: схожі курси)", "P2": "Пул №2 (А: кластери історії)", "P3": "Пул №3 (Б: домінантний кластер)"}
PNAME.update({f"P4_{m}": f"Пул №4 (В, M={m})" for m in POOL_M})
grp = u["activity_group"].to_numpy()
cid = c["course_id"].to_numpy()
cid_pos = {x: i for i, x in enumerate(cid)}

# ---------------------------------------------------------------- завантаження пулів і рекомендацій з CSV (як у поставці)
pool_idx = {}
for nm in POOLS:
    df = pd.read_csv(RES / "pools" / f"pool_{nm}.csv", encoding=ENC).set_index("user_id").loc[u["user_id"]]
    pool_idx[nm] = [np.array([cid_pos[x] for x in s.split(";")], int) if isinstance(s, str) and s else np.array([], int) for s in df["course_ids"]]
rec = pd.read_csv(RES / "recommendations_top10.csv", encoding=ENC)
rec["ci"] = rec["course_id"].map(cid_pos)
rec["ui"] = rec["user_id"].map(uid_idx)
top_idx = {nm: [np.array([], int)] * U for nm in POOLS}
for nm, g in rec.groupby("pool"):
    lst = [np.array([], int)] * U
    for ui, gg in g.groupby("ui"):
        lst[ui] = gg.sort_values("rank")["ci"].to_numpy()
    top_idx[nm] = lst

def counts(lists):
    M = np.zeros((U, K), int)
    for i, idx in enumerate(lists):
        if len(idx):
            M[i] = np.bincount(cl[idx], minlength=K)
    return M

# ---------------------------------------------------------------- 1. розміри пулів
rows = []
for nm in POOLS:
    size = np.array([len(x) for x in pool_idx[nm]])
    for g in GROUPS:
        s = size[grp == g]
        rows.append(dict(pool=nm, group=g, n_users=len(s), mean=s.mean(), min=s.min(), max=s.max(), median=np.median(s),
                         share_catalog_mean=s.mean() / C, empty=int((s == 0).sum())))
size_df = pd.DataFrame(rows)
size_df.to_csv(RES / "stats_pool_sizes.csv", index=False, encoding=ENC)

# ---------------------------------------------------------------- 2. розподіл за кластерами: середнє / мін / макс
def dist_table(kind, src):
    rows = []
    for nm in POOLS:
        M = counts(src[nm])
        for g in GROUPS:
            m = M[grp == g]
            for k in range(K):
                rows.append(dict(kind=kind, pool=nm, group=g, cluster=k, mean=m[:, k].mean(), min=int(m[:, k].min()), max=int(m[:, k].max()),
                                 share_users_nonzero=float((m[:, k] > 0).mean())))
    return pd.DataFrame(rows)
dp, dt = dist_table("pool", pool_idx), dist_table("top10", top_idx)
dp.to_csv(RES / "stats_cluster_distribution_pools.csv", index=False, encoding=ENC)
dt.to_csv(RES / "stats_cluster_distribution_top10.csv", index=False, encoding=ENC)

# ---------------------------------------------------------------- 3. профілі кластерів курсів
tf = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_df=0.5, sublinear_tf=True)
Tm = tf.fit_transform(c["text_clean"])
terms = np.array(tf.get_feature_names_out())
prof = []
for k in range(K):
    m = cl == k
    cen = np.asarray(Tm[m].mean(0)).ravel()
    cats = c.loc[m, "category"].value_counts()
    prof.append(dict(cluster=k, size=int(m.sum()), top_category=cats.index[0], top_category_share=cats.iloc[0] / m.sum(),
                     top_terms=", ".join(terms[np.argsort(-cen)[:5]]),
                     popular=int((c.loc[m, "popularity_group"] == "популярні").sum()), rare=int((c.loc[m, "popularity_group"] == "рідкісні").sum()),
                     new=int((c.loc[m, "popularity_group"] == "нові (0)").sum()), events=int(c.loc[m, "n_events"].sum()),
                     mean_rating=float(c.loc[m, "avg_rating"].mean())))
pd.DataFrame(prof).to_csv(RES / "stats_cluster_profiles.csv", index=False, encoding=ENC)

# ---------------------------------------------------------------- 4. якість списків ТОП-10 (без відкладеної вибірки)
S, _, _ = content_similarity(c)
pop = c["popularity_group"].to_numpy(dtype=object)
area = ct.loc[c["course_id"], "area"].to_numpy(dtype=object)
dom = ut.loc[u["user_id"], "dominant_area"].to_numpy(dtype=object)
rows = []
for nm in POOLS:
    for g in GROUPS:
        ids = np.where(grp == g)[0]
        nclu, ils, rel, popsh, rare, newsh, lens = [], [], [], [], [], [], []
        allrec = set()
        for i in ids:
            idx = top_idx[nm][i]
            lens.append(len(idx))
            if len(idx) == 0:
                continue
            allrec.update(idx.tolist())
            nclu.append(len(set(cl[idx])))
            if len(idx) > 1:
                sub = S[np.ix_(idx, idx)]
                ils.append(1 - (sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
            rel.append((area[idx] == dom[i]).mean())
            popsh.append((pop[idx] == "популярні").mean()); rare.append((pop[idx] == "рідкісні").mean()); newsh.append((pop[idx] == "нові (0)").mean())
        rows.append(dict(pool=nm, group=g, mean_len=np.mean(lens), mean_clusters_in_top10=np.mean(nclu), min_clusters=int(np.min(nclu)), max_clusters=int(np.max(nclu)),
                         intra_list_diversity=np.mean(ils), share_popular=np.mean(popsh), share_rare=np.mean(rare), share_new=np.mean(newsh),
                         precision_true_area=np.mean(rel), unique_courses=len(allrec), catalog_coverage=len(allrec) / C))
qual = pd.DataFrame(rows)
qual.to_csv(RES / "stats_top10_quality.csv", index=False, encoding=ENC)

# ---------------------------------------------------------------- ГРАФІКИ
GC = {"активні": "#4C78A8", "звичайні": "#9D9D9D", "пасивні": "#F58518", "нові (0)": "#54A24B"}
SHOW = ["активні", "пасивні", "нові (0)"]       # групи, обов'язкові за завданням (звичайні — додатково, у таблицях)

# 4.1 розміри пулів (лог-шкала)
fig, ax = plt.subplots(figsize=(9.5, 4.2))
w = 0.2
for j, g in enumerate(GROUPS):
    d = size_df[size_df.group == g].set_index("pool").loc[POOLS]
    x = np.arange(len(POOLS)) + (j - 1.5) * w
    ax.bar(x, d["mean"], w, color=GC[g], label=g)
    ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["min"], d["max"] - d["mean"]], fmt="none", ecolor="k", elinewidth=.7, capsize=1.5)
ax.set_yscale("log"); ax.set_xticks(range(len(POOLS))); ax.set_xticklabels(["П1", "П2", "П3"] + [f"П4\nM={m}" for m in POOL_M])
ax.set_ylabel("розмір пулу, курсів (лог. шкала)"); ax.legend(ncol=4, fontsize=8, frameon=False)
ax.set_title("Розмір пулів: середнє та діапазон мін–макс за групами користувачів")
fig.tight_layout(); fig.savefig(FIG / "l2_pool_sizes.png", bbox_inches="tight"); plt.close(fig)

# 4.2 розподіл за кластерами для кожного пулу: середнє з вусами мін–макс, групи активні/пасивні/нові
def cluster_panel(d, nm, title, fname, ylabel):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
    for ax, g in zip(axes, SHOW):
        x = d[(d.pool == nm) & (d.group == g)].sort_values("cluster")
        ax.bar(x["cluster"], x["mean"], color=GC[g], width=.75)
        ax.errorbar(x["cluster"], x["mean"], yerr=[x["mean"] - x["min"], x["max"] - x["mean"]], fmt="none", ecolor="k", elinewidth=.7, capsize=1.5)
        ax.set_title(f"{g} (n={int((grp == g).sum())})", fontsize=9); ax.set_xlabel("кластер курсів"); ax.set_xticks(range(K))
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10.5, y=1.02); fig.tight_layout(); fig.savefig(FIG / fname, bbox_inches="tight"); plt.close(fig)

for nm in ["P1", "P2", "P3", "P4_100"]:
    cluster_panel(dp, nm, f"{PNAME[nm]}: кількість курсів кожного кластера в пулі (середнє; вуса — мін–макс)", f"l2_pool_clusters_{nm}.png", "курсів у пулі")
for nm in ["P1", "P2", "P3", "P4_100"]:
    cluster_panel(dt, nm, f"{PNAME[nm]}: курси кожного кластера в ТОП-10 (середнє; вуса — мін–макс)", f"l2_top10_clusters_{nm}.png", "курсів у ТОП-10")

# 4.3 теплокарта середніх часток кластерів у ТОП-10 (пул × кластер) за групами
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, g in zip(axes, SHOW):
    mat = np.array([dt[(dt.pool == nm) & (dt.group == g)].sort_values("cluster")["mean"].to_numpy() for nm in POOLS])
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xlabel("кластер курсів"); ax.set_title(g, fontsize=9)
    for a in range(len(POOLS)):
        for b in range(K):
            ax.text(b, a, f"{mat[a, b]:.1f}", ha="center", va="center", fontsize=6, color="white" if mat[a, b] > 6 else "black")
axes[0].set_yticks(range(len(POOLS))); axes[0].set_yticklabels(["П1", "П2", "П3"] + [f"П4 (M={m})" for m in POOL_M], fontsize=8)
fig.suptitle("Середня кількість курсів кожного кластера у списках ТОП-10", fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "l2_top10_heatmap.png", bbox_inches="tight"); plt.close(fig)

# 4.4 вплив порога M у стратегії В: розмір пулу, кількість кластерів у пулі, охоплення відкладених курсів (08_holdout_sensitivity.py)
ho = pd.read_csv(RES / "stats_holdout.csv", encoding=ENC)
nclu_pool = {}
for m in POOL_M:
    nclu_pool[m] = np.array([len(set(cl[x])) for x in pool_idx[f"P4_{m}"]])
pd.DataFrame([dict(M=m, group=g, mean_pool_size=size_df[(size_df.pool == f"P4_{m}") & (size_df.group == g)]["mean"].iloc[0],
                   mean_clusters_in_pool=nclu_pool[m][grp == g].mean(),
                   mean_clusters_in_top10=qual[(qual.pool == f"P4_{m}") & (qual.group == g)]["mean_clusters_in_top10"].iloc[0],
                   pool_recall=(ho[(ho.method == f"P4_{m}") & (ho.group == g)]["pool_recall"].iloc[0] if g in ("активні", "звичайні", "пасивні") else np.nan))
           for m in POOL_M for g in GROUPS]).to_csv(RES / "stats_strategyB_threshold.csv", index=False, encoding=ENC)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
for g in SHOW:
    sz = size_df[size_df.group == g].set_index("pool")
    xs = [f"P4_{m}" for m in POOL_M]
    axes[0].plot(POOL_M, sz.loc[xs, "mean"], "o-", color=GC[g], label=g)
    axes[1].plot(POOL_M, [nclu_pool[m][grp == g].mean() for m in POOL_M], "o-", color=GC[g], label=g)
axes[0].set_ylabel("середній розмір пулу №4"); axes[1].set_ylabel("кластерів, представлених у пулі (середнє)")
for ax in axes:
    ax.set_xlabel("порогове значення M"); ax.set_xscale("log"); ax.set_xticks(POOL_M); ax.set_xticklabels(POOL_M); ax.set_xticks([], minor=True)   # без проміжних міток 3×10¹, 4×10¹ … — лише значення M
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Стратегія В: вплив порогового розміру пулу №4", fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "l2_strategyB_threshold.png", bbox_inches="tight"); plt.close(fig)

# 4.5 приклад: склад історії, пулів і ТОП-10 для одного користувача кожної групи (медіанний за розміром історії)
Eclu = np.load(RES / "aux_pools.npz")["Eclu"]
ex = {}
for g in SHOW:
    ids = np.where(grp == g)[0]
    sizes = np.array([len(pool_idx["P2"][i]) for i in ids])
    ex[g] = int(ids[np.argsort(sizes)[len(ids) // 2]])
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=False)
labels = ["історія", "П1", "П2", "П3", "П4\n(M=100)", "ТОП-10\nП1", "ТОП-10\nП3", "ТОП-10\nП4"]
for ax, g in zip(axes, SHOW):
    i = ex[g]
    cols = [Eclu[i] / max(Eclu[i].sum(), 1e-9)]
    for nm in ["P1", "P2", "P3", "P4_100"]:
        v = np.bincount(cl[pool_idx[nm][i]], minlength=K).astype(float); cols.append(v / max(v.sum(), 1))
    for nm in ["P1", "P3", "P4_100"]:
        v = np.bincount(cl[top_idx[nm][i]], minlength=K).astype(float); cols.append(v / max(v.sum(), 1))
    M = np.array(cols).T
    bottom = np.zeros(M.shape[1])
    for k in range(K):
        ax.bar(range(M.shape[1]), M[k], bottom=bottom, color=cmap(k), width=.8, label=f"{k}")
        bottom += M[k]
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_title(f"{g}: {u.loc[i, 'user_id']}", fontsize=9); ax.set_ylim(0, 1)
axes[0].set_ylabel("частка кластера")
axes[-1].legend(title="кластер", fontsize=7, title_fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
fig.suptitle("Приклади: склад кластерів в історії, пулах і ТОП-10 (по одному користувачеві кожної групи)", fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "l2_examples_composition.png", bbox_inches="tight"); plt.close(fig)
json.dump(ex_ := {g: u.loc[i, "user_id"] for g, i in ex.items()}, open(RES / "example_users.json", "w", encoding="utf-8"), ensure_ascii=False)
print("приклади:", ex_)
print(size_df.query("pool in ['P1','P2','P3','P4_100']").round(1).to_string())
print(qual.query("pool in ['P1','P2','P3','P4_100']").round(3).to_string())
