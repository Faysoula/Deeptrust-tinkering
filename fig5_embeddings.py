import sys, os, gc, random

sys.path.append(os.path.abspath("android-detectors/src"))
import numpy as np, torch, dill as pkl
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from models.natural_mlp.natural_mlp import NaturalMLP
from models.robust_mlp.robust_mlp import RobustMLP
from models.utils import load_features, load_labels

MODELS = [
    ("vanilla", NaturalMLP),
    ("adv75", RobustMLP),
    ("adv100", RobustMLP),
    ("smooth2", RobustMLP),
    ("smooth4", RobustMLP),
]

with open("fig5_models/vanilla_vectorizer.pkl", "rb") as f:
    vec = pkl.load(f)
feats = load_features("data/training_set_features.zip")
y = np.asarray(load_labels("data/training_set_features.zip", "data/training_set.zip"))
X = vec.transform(feats)
del feats, vec
gc.collect()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
_, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y)

rng = np.random.RandomState(0)
good = rng.choice(val_idx[y[val_idx] == 0], 1000, replace=False)
mal = rng.choice(val_idx[y[val_idx] == 1], 1000, replace=False)
sel = np.concatenate([good, mal])
lab = np.concatenate([np.zeros(1000, int), np.ones(1000, int)])
Xs = X[sel]
del X
gc.collect()
np.save("fig5_labels.npy", lab)
print("selected", Xs.shape, flush=True)

for tag, cls in MODELS:
    m = cls(OmegaConf.load(f"fig5_models/{tag}_cfg.yaml"))
    m.load_state_dict(
        torch.load(f"fig5_models/{tag}.pth", map_location=m.device, weights_only=True)
    )
    m.eval()
    E, L = [], []
    with torch.no_grad():
        for i in range(0, Xs.shape[0], 64):
            b = torch.from_numpy(Xs[i : i + 64].toarray().astype(np.float32)).to(
                m.device
            )
            out, emb = m.forward(b, return_embedding=True)
            E.append(emb.cpu().numpy())
            L.append(out.cpu().numpy().ravel())
    np.savez(f"fig5_emb_{tag}.npz", emb=np.concatenate(E), logit=np.concatenate(L))
    print(f"{tag}: {np.concatenate(E).shape}", flush=True)
    del m
    torch.cuda.empty_cache()
    gc.collect()
print("done")
