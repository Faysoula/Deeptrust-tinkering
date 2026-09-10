import sys, os

sys.path.append(os.path.abspath("android-detectors/src"))
import numpy as np, torch, dill as pkl
from torch.nn.functional import sigmoid
from omegaconf import OmegaConf
from models.natural_mlp.natural_mlp import NaturalMLP
from models.utils import load_features

m = NaturalMLP(OmegaConf.load("fig5_models/vanilla_cfg.yaml"))
m.load_state_dict(
    torch.load("fig5_models/vanilla.pth", map_location=m.device, weights_only=True)
)
with open("fig5_models/vanilla_vectorizer.pkl", "rb") as f:
    m._vectorizer = pkl.load(f)
m.eval()

BS = 64


def rate(zip_path, positive):
    X = m._vectorizer.transform(load_features(zip_path))
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BS):
            b = torch.from_numpy(X[i : i + BS].toarray().astype(np.float32)).to(
                m.device
            )
            p = sigmoid(m.forward(b)).cpu().numpy().ravel()
            preds.append((p >= 0.5).astype(int))
    preds = np.concatenate(preds)
    print(f"  {zip_path.split('/')[-1]}: n={len(preds)}")
    return preds.mean() if positive else 1.0 - preds.mean()


tnr = rate("data/test_set_fp_check_features.zip", False)
tpr = rate("data/test_set_adv_features.zip", True)
print(f"\nvanilla-MLP  TNR {tnr:.4f} (paper .9962) | TPR {tpr:.4f} (paper .7544)")
