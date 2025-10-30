import json
import numpy as np
import pandas as pd
from pathlib import Path
from features import REALTIME_FEATURES

REF = Path("monitor/ref_features.parquet")
TODAY = Path("monitor/today_features.parquet")

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, qs)
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    return float(np.sum((a - e) * np.log((a + eps) / (e + eps))))

def run():
    ref = pd.read_parquet(REF)
    cur = pd.read_parquet(TODAY)
    alerts = {}
    for f in REALTIME_FEATURES:
        v = psi(ref[f].values, cur[f].values)
        if v > 0.25:
            alerts[f] = v
    print(json.dumps({"drift_alerts": alerts}, indent=2))

if __name__ == "__main__":
    run()
