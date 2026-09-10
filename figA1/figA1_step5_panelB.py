import sys, os, json, importlib.util, random, time
import numpy as np, torch

sys.path.append(os.path.abspath("android-detectors/src"))
sys.path.append(os.path.abspath("track_1"))
sys.path.append(os.path.abspath("track_1/feature_space_attack"))

BUDGETS = [25, 50]
CS = [None] + [round(x, 2) for x in np.arange(0.02, 0.161, 0.02)]
OUT = "figA1_panelB.json"

spec = importlib.util.spec_from_file_location(
    "dl", "android-detectors/src/loaders/deeptrust_loader.py"
)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)
clf = dl.load()  # load ONCE

from models.utils import load_features, load_labels
from feature_space_attack import FeatureSpaceAttack

train_scores = np.load("figA1_train_scores.npy")
orig_forest = clf.inspectRF

# --- sample draw: seed AFTER load_classifier (your Finding 2) ---
np.random.seed(0)
indices = np.random.choice(1250, 50, replace=False)
print("indices[:10]:", indices[:10])
print("expected    : [711 898 186 867  18 1152 192 184 824 1058]")


class Gen:
    def __init__(s, f, idx):
        s.f, s.idx = f, set(idx)

    def __iter__(s):
        for i, x in enumerate(s.f):
            if i in s.idx:
                yield x


results = json.load(open(OUT)) if os.path.exists(OUT) else {}

for c in CS:
    if c is None:
        clf.inspectRF = None
        off = None
    else:
        clf.inspectRF = orig_forest
        off = float(np.percentile(train_scores, 100 * c))
        clf.inspectRF.offset_ = off
    for nf in BUDGETS:
        key = f"c={c}_fsa={nf}"
        if key in results:
            print("skip", key)
            continue
        t0 = time.time()
        y_tr = load_labels("data/training_set_features.zip", "data/training_set.zip")
        good = (
            s
            for s, l in zip(load_features("data/training_set_features.zip"), y_tr)
            if l == 0
        )
        mal = Gen(load_features("data/test_set_adv_features.zip"), indices)

        atk = FeatureSpaceAttack(classifier=clf)
        adv = atk.run(mal, good, n_iterations=100, n_features=nf, n_candidates=50)
        y_pred, _ = clf.predict(adv)
        tpr = float(np.mean(y_pred))
        results[key] = {
            "c": c,
            "offset": off,
            "fsa": nf,
            "tpr": tpr,
            "n_detected": int(np.sum(y_pred)),
            "mins": round((time.time() - t0) / 60, 1),
        }
        json.dump(results, open(OUT, "w"), indent=2)
        print(
            f"{key}: TPR={tpr:.4f} ({int(np.sum(y_pred))}/50) "
            f"[{results[key]['mins']} min]",
            flush=True,
        )

print("done ->", OUT)
