# -*- coding: utf-8 -*-
"""Крок 10 (ЛР2). Графіки відкладеної вибірки: охоплення «майбутніх» курсів пулами та HR@10."""
import matplotlib.pyplot as plt
from cb_lib import *
ho = pd.read_csv(RES / "stats_holdout.csv", encoding=ENC)
GC = {"активні": "#4C78A8", "звичайні": "#9D9D9D", "пасивні": "#F58518"}
meths = ["P1", "P2", "P3", "P4_100", "P4_500", "POP", "RAND"]
lab = ["П1", "П2", "П3", "П4\nM=100", "П4\nM=500", "Популяр-\nність", "Випадк."]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
w = 0.26
for j, g in enumerate(["активні", "звичайні", "пасивні"]):
    d = ho[ho.group == g].set_index("method").loc[meths]
    x = np.arange(len(meths)) + (j - 1) * w
    axes[0].bar(x, d["pool_recall"], w, color=GC[g], label=g)
    axes[1].bar(x, d["HR10"], w, color=GC[g], label=g)
for ax, t in zip(axes, ["Охоплення відкладених курсів пулом (pool recall)", "Влучність ТОП-10 (HR@10)"]):
    ax.set_xticks(range(len(meths))); ax.set_xticklabels(lab, fontsize=7.5); ax.set_title(t, fontsize=9.5)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "l2_holdout.png", bbox_inches="tight"); plt.close(fig)
