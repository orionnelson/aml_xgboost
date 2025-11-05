# train.py
import json
import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Platt calibration
import joblib
from scipy.special import expit

from features import (
    transform_raw_to_features,
    REALTIME_FEATURES,
    BASE_FEATURES,
)

PARQUET_PATH = pathlib.Path("data/historical.parquet")
CSV_PATH = pathlib.Path("data/historical.csv")
ARTIFACT_DIR = pathlib.Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

CALIBRATOR_PATH = ARTIFACT_DIR / "calibrator_v1.joblib"
MODEL_PATH = ARTIFACT_DIR / "model_v1.xgb"
SPEC_PATH = ARTIFACT_DIR / "model_v1.json"

def _load_raw():
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError("No historical.{parquet|csv} found under data/")

def load_dataset():
    df = _load_raw()
    X = df.apply(lambda r: pd.Series(transform_raw_to_features(r.to_dict())), axis=1)
    X = X.reindex(columns=REALTIME_FEATURES, fill_value=0.0).astype(float)
    y = df["is_fraud"].astype(int)
    return X, y

def _monotone_vector(feature_names):
    base_dir = {
        "amount": +1,
        "device_trust_score": -1,
        "ip_risk_score": +1,
        "acct_age_days": -1,
        "txns_last_5m": +1,
        "declines_last_24h": +1,
        "chargebacks_90d": +1,
    }
    return "(" + ",".join(str(base_dir.get(f, 0)) for f in feature_names) + ")"

def train():
    X, y = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    scale_pos_weight = float((neg / max(pos, 1)) ** 0.5)

    feature_names = REALTIME_FEATURES
    monotone = _monotone_vector(feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 4,
        "eta": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.5,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "max_bin": 256,
        "monotone_constraints": monotone,
        "seed": 42,
    }

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=feature_names)

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=2000,
        evals=[(dte, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # Raw margins
    y_val_margin = bst.predict(dte, output_margin=True)
    # AUC on margins is OK (rank-based), but logloss must use probabilities
    y_val_prob = expit(y_val_margin)
    auc_raw = roc_auc_score(yte, y_val_prob)
    logloss_raw = log_loss(yte, y_val_prob)

    # Platt on margins
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(y_val_margin.reshape(-1, 1), yte)
    y_val_cal = platt.predict_proba(y_val_margin.reshape(-1, 1))[:, 1]
    auc_cal = roc_auc_score(yte, y_val_cal)
    logloss_cal = log_loss(yte, y_val_cal)

    # Persist artifacts
    bst.save_model(MODEL_PATH.as_posix())
    joblib.dump(platt, CALIBRATOR_PATH.as_posix())

    # Thresholds from calibrated scores
    decline_thr = float(np.quantile(y_val_cal, 0.95))
    review_thr = float(np.quantile(y_val_cal, 0.75))

    spec = {
        "feature_order": feature_names,
        "thresholds": {"decline": decline_thr, "review": review_thr},
        "metrics": {
            "val_auc_raw": float(auc_raw),
            "val_logloss_raw": float(logloss_raw),
            "val_auc_cal": float(auc_cal),
            "val_logloss_cal": float(logloss_cal),
            "best_iteration": int(getattr(bst, "best_iteration", len(y_val_margin))),
            "class_pos_rate_train": float(pos / (pos + neg)),
            "scale_pos_weight": scale_pos_weight,
        },
        "model_version": "v1",
        "monotone_constraints": monotone,
        "base_features": BASE_FEATURES,
        "calibrator": {
            "type": "platt",
            "path": CALIBRATOR_PATH.name,
        },
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2))

if __name__ == "__main__":
    train()
