import sys, os, time, gc

sys.path.append(os.path.abspath("android-detectors/src"))
import torch, numpy as np, dill as pkl
from omegaconf import OmegaConf
from models.natural_mlp.natural_mlp import NaturalMLP
from models.utils import load_features, load_labels

os.makedirs("fig5_models", exist_ok=True)

cfg = OmegaConf.create(
    {
        "model": {
            "in_dim": 1461078,
            "out_dim": 1,
            "hidden_sizes": [128, 64],
            "activation": "leaky_relu",
            "dropout": 0.0,
        },
        "trainer": {
            "batch_size": 32,
            "patience": 3,
            "min_epochs": 3,
            "max_epochs": 6,
            "lr_rate": 0.001,
            "optimizer": "adam",
            "optimizer_params": {
                "betas": [0.99, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "loss": "bce",
            "loss_params": {"pos_weight": 9.0},
        },
    }
)
OmegaConf.save(cfg, "fig5_models/vanilla_cfg.yaml")

m = NaturalMLP(cfg)
print("device:", m.device, flush=True)

feats = load_features("data/training_set_features.zip")
y = load_labels("data/training_set_features.zip", "data/training_set.zip")
print("vectorizing...", flush=True)
X = m._vectorizer.fit_transform(feats)
m._input_features = m._vectorizer.get_feature_names_out().tolist()

with open("fig5_models/vanilla_vectorizer.pkl", "wb") as f:
    pkl.dump(m._vectorizer, f)  # save BEFORE training

del feats
gc.collect()  # <-- frees several GB
print("X:", X.shape, flush=True)

t0 = time.time()
m._fit(X, y)
print(f"TRAINED in {(time.time()-t0)/60:.1f} min", flush=True)
torch.save(m.state_dict(), "fig5_models/vanilla.pth")
print("saved", flush=True)
