import sys, os, importlib.util, random
import numpy as np, torch

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
print("loaded. stored offset_:", repr(clf.inspectRF.offset_))

from models.utils import load_features, load_labels

feats = load_features("data/training_set_features.zip")
y = load_labels("data/training_set_features.zip", "data/training_set.zip")
print("n samples:", len(y), "n goodware:", int((np.asarray(y) == 0).sum()))

X = clf._vectorizer.transform(feats)
print("X:", X.shape)

np.random.seed(0)  # mirror __init__ seeding before the split
clf.trustNet.load_pt_dataset(X, y, clf.trustNet.distillation)

embs = []
clf.trustNet.eval()
with torch.no_grad():
    for i, (f, lab, hard) in enumerate(clf.trustNet.trainloader):
        f = f.to(clf.device)
        hard = hard.to(clf.device).squeeze()
        f = f[hard == 0, :]
        if f.shape[0] == 0:
            continue
        _, e = clf.trustNet.forward(f, return_embedding=True)
        embs.append(e.cpu())
        if i % 200 == 0:
            print(f"  batch {i}", flush=True)

E = torch.cat(embs).numpy()
print("embeddings:", E.shape, "(target: 54000 x 256)")
np.save("figA1_train_goodware_embeddings.npy", E)

s = clf.inspectRF.score_samples(E)
np.save("figA1_train_scores.npy", s)
rec = np.percentile(s, 14.0)
print("recomputed offset_:", repr(rec))
print("stored     offset_:", repr(clf.inspectRF.offset_))
print("abs diff:", abs(rec - clf.inspectRF.offset_))
