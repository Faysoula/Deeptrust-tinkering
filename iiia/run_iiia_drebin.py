import sys, json, logging, importlib.util
import numpy as np

sys.path.insert(0, "android-detectors/src")
sys.path.insert(0, "track_1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def load_classifier(path):
    spec = importlib.util.spec_from_file_location("clf_loader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["clf_loader"] = module
    spec.loader.exec_module(module)
    return module.load()

from track_1.iiia_evaluate import iiia_evaluate

if __name__ == "__main__":
    clf = load_classifier("android-detectors/src/loaders/drebin_loader.py")
    np.random.seed(0)
    metrics = iiia_evaluate(clf)
    print(json.dumps(metrics, indent=2, default=float))
    with open("iiia_metrics_drebin.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
