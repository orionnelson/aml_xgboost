import json
import pathlib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from features import transform_raw_to_features, REALTIME_FEATURES

DATA_PATH = "data/historical.parquet"
ARTIFACT_DIR = pathlib.Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    X = df.apply(lambda r: pd.Series(transform_raw_to_features(r.to_dict())), axis=1)
    y = df["is_fraud"].astype(int)
    return X[REALTIME_FEATURES], y

def train():
    X, y = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos = ytr.sum()
    neg = len(ytr) - pos
    spw = (neg / max(pos, 1))

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=REALTIME_FEATURES)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=REALTIME_FEATURES)

    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": spw,
        "lambda": 1.0,
        "alpha": 0.0,
    }

    bst = xgb.train(params, dtr, num_boost_round=300, evals=[(dte, "val")], verbose_eval=False)
    y_hat = bst.predict(dte)
    auc = roc_auc_score(yte, y_hat)

    model_path = ARTIFACT_DIR / "model_v1.xgb"
    bst.save_model(model_path.as_posix())

    spec = {
        "feature_order": REALTIME_FEATURES,
        "thresholds": {"decline": 0.90, "review": 0.70},
        "metrics": {"val_auc": float(auc)},
        "model_version": "v1",
    }
    (ARTIFACT_DIR / "model_v1.json").write_text(json.dumps(spec, indent=2))

if __name__ == "__main__":
    train()
