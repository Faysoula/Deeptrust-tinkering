import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

TAGS = [
    ("vanilla", "vanilla-MLP"),
    ("adv75", "adversarial(75)-MLP"),
    ("adv100", "adversarial(100)-MLP"),
    ("smooth2", "smoothed(0.2)-MLP"),
    ("smooth4", "smoothed(0.4)-MLP"),
]

lab = np.load("fig5_labels.npy")
fig, axes = plt.subplots(5, 2, figsize=(9, 22))

for r, (tag, title) in enumerate(TAGS):
    d = np.load(f"fig5_emb_{tag}.npz")
    E, lg = d["emb"], d["logit"]
    v = np.percentile(np.abs(lg), 90)

    proj = {
        "UMAP": umap.UMAP(random_state=0).fit_transform(E),
        "t-SNE": TSNE(
            n_components=2, random_state=0, init="pca", perplexity=30
        ).fit_transform(E),
    }
    for c, (name, P) in enumerate(proj.items()):
        ax = axes[r, c]
        for cls, mk, ec in [(0, "o", "k"), (1, "x", "r")]:
            k = lab == cls
            ax.scatter(
                P[k, 0],
                P[k, 1],
                c=lg[k],
                cmap="coolwarm",
                vmin=-v,
                vmax=v,
                marker=mk,
                s=9,
                linewidths=0.4,
                edgecolors=ec if cls == 0 else None,
            )
        ax.set_title(f"{title} — {name}", fontsize=9)
        ax.set_xlabel("dim 1", fontsize=7)
        ax.set_ylabel("dim 2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    print(f"{tag} done", flush=True)

fig.suptitle(
    "Fig. 5 replication — 1k goodware + 1k malware embeddings, " "last hidden layer",
    fontsize=10,
    y=0.999,
)
fig.tight_layout()
fig.savefig("fig5_replication.png", dpi=170)
fig.savefig("fig5_replication.pdf")
print("saved fig5_replication.png / .pdf")
