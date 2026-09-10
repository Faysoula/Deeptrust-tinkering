import sys, os, importlib.util, random
import numpy as np, torch
from torch.nn.functional import sigmoid

sys.path.append(os.path.abspath("android-detectors/src"))
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

spec = importlib.util.spec_from_file_location(
    "dl", "android-detectors/src/loaders/deeptrust_loader.py"
)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)
clf = dl.load()

from models.utils import load_features


def cache(zip_path, tag):
    feats = load_features(zip_path)
    X = clf.trustNet._vectorizer.transform(feats)  # mirrors predict()
    if getattr(clf.trustNet, "used_features", None) is not None:
        X = X[:, clf.trustNet.used_features]
    print(f"{tag}: X {X.shape}", flush=True)

    loader, _ = clf.trustNet.load_pt_dataset(X)  # y=None -> no shuffle
    g, b, a = [], [], []
    clf.trustNet.eval()
    clf.guardNet.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            f = batch.to(clf.device)
            gp = sigmoid(clf.guardNet.forward(f)).cpu().numpy().ravel()
            out, emb = clf.trustNet.forward(f, return_embedding=True)
            bp = sigmoid(out).cpu().numpy().ravel()
            sc = clf.inspectRF.score_samples(emb.cpu().numpy())
            g.append(gp)
            b.append(bp)
            a.append(sc)
            if i % 50 == 0:
                print(f"  {tag} batch {i}", flush=True)

    g, b, a = np.concatenate(g), np.concatenate(b), np.concatenate(a)
    np.savez(f"figA1_cache_{tag}.npz", guard=g, base=b, anom=a)
    print(f"{tag}: cached {g.shape[0]} samples\n", flush=True)


cache("data/test_set_fp_check_features.zip", "goodware")
cache("data/test_set_adv_features.zip", "malware")
