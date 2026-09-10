import numpy as np, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H1, H2, H4 = 0.78, 0.5, 0.5
g = np.load("figA1_cache_goodware.npz")
m = np.load("figA1_cache_malware.npz")
train_scores = np.load("figA1_train_scores.npy")


def cascade(guard, base, anom, offset):
    """offset=None reproduces inspectRF=None (the x=0 point)."""
    pred = np.zeros(len(guard), dtype=int)
    s1 = guard >= H1
    pred[s1] = 1
    rest = ~s1
    mal = rest & (base >= H2)
    pred[mal] = 1
    und = rest & (base < H2)
    inlier = np.zeros(len(guard), bool) if offset is None else (anom - offset) >= 0
    pred[und & inlier] = 0
    c = und & ~inlier
    pred[c] = (guard[c] >= H4).astype(int)
    return pred


def metrics(offset):
    tnr = 1.0 - cascade(g["guard"], g["base"], g["anom"], offset).mean()
    tpr = cascade(m["guard"], m["base"], m["anom"], offset).mean()
    return tnr, tpr


# --- validation at the shipped c=0.14 ---
off14 = np.percentile(train_scores, 14.0)
tnr, tpr = metrics(off14)
print(f"c=0.14 -> TNR {tnr:.4f} (yours 0.9902) | TPR {tpr:.4f} (yours 0.7832)")
assert abs(off14 - (-0.3424891715606571)) < 1e-12

# --- sweep ---
cs = [round(x, 2) for x in np.arange(0.02, 0.161, 0.02)]
rows = [{"c": 0.0, "offset": None, **dict(zip(("tnr", "tpr"), metrics(None)))}]
for c in cs:
    o = float(np.percentile(train_scores, 100 * c))
    t, p = metrics(o)
    rows.append({"c": c, "offset": o, "tnr": t, "tpr": p})
for r in rows:
    print(f"  c={r['c']:.2f}  TNR={r['tnr']:.4f}  TPR={r['tpr']:.4f}")
json.dump(rows, open("figA1_panelA.json", "w"), indent=2)

# --- plot ---
sw = [r for r in rows if r["c"] > 0]
z = rows[0]
x = [r["c"] for r in sw]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(x, [r["tpr"] for r in sw], "o-", color="#1a9e9e", label="TPR (DeepTrust)")
ax.plot(x, [r["tnr"] for r in sw], "*-", color="#d1495b", ms=9, label="TNR (DeepTrust)")
ax.plot(0, z["tpr"], "o", color="#1a9e9e")
ax.plot(0, z["tnr"], "*", color="#d1495b", ms=9)
for xx, yy in [
    (0, z["tpr"]),
    (0, z["tnr"]),
    (x[0], sw[0]["tpr"]),
    (x[0], sw[0]["tnr"]),
    (x[-1], sw[-1]["tpr"]),
    (x[-1], sw[-1]["tnr"]),
]:
    ax.annotate(
        f"{yy:.4f}",
        (xx, yy),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=7,
    )
ax.axhline(0.99, ls=":", lw=0.8, color="gray")
ax.set_xlabel("Contamination Hyperparameter")
ax.set_ylabel("Metric Value")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figA1_panelA.png", dpi=200)
fig.savefig("figA1_panelA.pdf")
print("saved figA1_panelA.png / .pdf")
