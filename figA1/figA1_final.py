import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = {r["c"]: r for r in json.load(open("figA1_panelA.json"))}
B = json.load(open("figA1_panelB.json"))
cs = sorted(c for c in A if c > 0)


def tpr(c, fsa):
    key = f"c={'None' if c == 0 else c}_fsa={fsa}"
    return B[key]["tpr"]


PA = {
    0.0: (0.9826, 0.8168),
    0.02: (0.9946, 0.7672),
    0.14: (0.9902, 0.7832),
    0.16: (0.9894, 0.7848),
}
PB = {
    25: {0.0: 0.6456, 0.02: 0.2896, 0.16: 0.4656},
    50: {0.0: 0.5368, 0.02: 0.1712, 0.16: 0.3760},
    100: {0.0: 0.2752, 0.02: 0.1056, 0.16: 0.2440},
}
STYLE = {25: ("#e8833a", "^"), 50: ("#4a90c4", "s"), 100: ("#8e6bbf", "D")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

ax1.plot(cs, [A[c]["tpr"] for c in cs], "o-", c="#1a9e9e", label="TPR (DeepTrust)")
ax1.plot(
    cs, [A[c]["tnr"] for c in cs], "*-", c="#d1495b", ms=9, label="TNR (DeepTrust)"
)
ax1.plot(0, A[0.0]["tpr"], "X", c="#1a9e9e", ms=9)
ax1.plot(0, A[0.0]["tnr"], "X", c="#d1495b", ms=9)
ax1.plot(
    list(PA),
    [v[1] for v in PA.values()],
    "o",
    mfc="none",
    mec="#1a9e9e",
    ms=12,
    ls="none",
    label="paper",
)
ax1.plot(
    list(PA),
    [v[0] for v in PA.values()],
    "*",
    mfc="none",
    mec="#d1495b",
    ms=14,
    ls="none",
)
for c in (0.0, 0.02, 0.16):
    for k in ("tnr", "tpr"):
        ax1.annotate(
            f"{A[c][k]:.4f}",
            (c, A[c][k]),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
        )
ax1.set_ylim(0.70, 1.06)
ax1.set_ylabel("Metric Value")
ax1.set_title("(a) TNR goodware; TPR malware (TSM)", fontsize=10)

for fsa in (25, 50, 100):
    col, mk = STYLE[fsa]
    ax2.plot(
        cs,
        [tpr(c, fsa) for c in cs],
        mk + "-",
        c=col,
        ms=6,
        label=f"{fsa}-FSA (DeepTrust)",
    )
    ax2.plot(0, tpr(0, fsa), "X", c=col, ms=9)
    ax2.plot(
        list(PB[fsa]), list(PB[fsa].values()), mk, mfc="none", mec=col, ms=12, ls="none"
    )
    for c in (0.0, 0.02, 0.16):
        ax2.annotate(
            f"{tpr(c, fsa):.4f}",
            (c, tpr(c, fsa)),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
        )
ax2.plot([], [], "o", mfc="none", mec="gray", ls="none", label="paper")
ax2.set_ylim(0.05, 0.72)
ax2.set_ylabel("TPR (Recall)")
ax2.set_title("(b) TPR on TSM under {25,50,100}-FSA", fontsize=10)

for ax in (ax1, ax2):
    ax.set_xlabel("Contamination Hyperparameter")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

fig.suptitle(
    "Fig. A.1 replication — filled: this work (n=50 attack), "
    "hollow: Pulido-Cortázar et al. (n=1250)",
    fontsize=9.5,
)
fig.tight_layout()
fig.savefig("figA1_final.png", dpi=220)
fig.savefig("figA1_final.pdf")
print("saved figA1_final.png / .pdf")
