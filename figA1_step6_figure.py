import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("figA1_panelA.json"))
B = json.load(open("figA1_panelB.json"))

pa = {r["c"]: r for r in A}
pb = {(r["c"] if r["c"] is not None else 0.0): r for r in B.values()}
cs = [c for c in sorted(pa) if c > 0]

PAPER_A = {
    0.0: (0.9826, 0.8168),
    0.02: (0.9946, 0.7672),
    0.14: (0.9902, 0.7832),
    0.16: (0.9894, 0.7848),
}
PAPER_B = {0.0: 0.2752, 0.02: 0.1056, 0.14: 0.1992, 0.16: 0.2440}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))

# ---- panel (a)
ax1.plot(cs, [pa[c]["tpr"] for c in cs], "o-", color="#1a9e9e", label="TPR (DeepTrust)")
ax1.plot(
    cs, [pa[c]["tnr"] for c in cs], "*-", color="#d1495b", ms=9, label="TNR (DeepTrust)"
)
ax1.plot(0, pa[0.0]["tpr"], "o", color="#1a9e9e")
ax1.plot(0, pa[0.0]["tnr"], "*", color="#d1495b", ms=9)
ax1.plot(
    list(PAPER_A),
    [v[1] for v in PAPER_A.values()],
    "o",
    mfc="none",
    mec="#1a9e9e",
    ms=11,
    ls="none",
    label="paper",
)
ax1.plot(
    list(PAPER_A),
    [v[0] for v in PAPER_A.values()],
    "*",
    mfc="none",
    mec="#d1495b",
    ms=13,
    ls="none",
)
for c in (0.0, 0.02, 0.16):
    for k in ("tnr", "tpr"):
        ax1.annotate(
            f"{pa[c][k]:.4f}",
            (c, pa[c][k]),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
ax1.set_ylim(0.70, 1.06)
ax1.set_ylabel("Metric Value")
ax1.set_title("(a) TNR goodware; TPR malware (TSM)", fontsize=9)

# ---- panel (b)
ax2.plot(
    cs, [pb[c]["tpr"] for c in cs], "D-", color="#8e6bbf", label="100-FSA (DeepTrust)"
)
ax2.plot(0, pb[0.0]["tpr"], "D", color="#8e6bbf")
ax2.plot(
    list(PAPER_B),
    list(PAPER_B.values()),
    "D",
    mfc="none",
    mec="#8e6bbf",
    ms=11,
    ls="none",
    label="paper",
)
for c in (0.0, 0.02, 0.16):
    ax2.annotate(
        f"{pb[c]['tpr']:.4f}",
        (c, pb[c]["tpr"]),
        fontsize=7,
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
    )
ax2.set_ylabel("TPR (Recall)")
ax2.set_title("(b) TPR on TSM under 100-FSA", fontsize=9)

for ax in (ax1, ax2):
    ax.set_xlabel("Contamination Hyperparameter")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

fig.suptitle(
    "Fig. A.1 replication — filled: this work (n=50 attack), "
    "hollow: Pulido-Cortázar et al.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig("figA1_replication.png", dpi=220)
fig.savefig("figA1_replication.pdf")
print("saved figA1_replication.png / .pdf")
