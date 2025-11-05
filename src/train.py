# train.py
# Adds logging, JSONL metrics, and an interaction audit (amount × category).

import json
import logging
import sys
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

from features import (
    transform_raw_to_features,
    REALTIME_FEATURES,
    BASE_FEATURES,
)

# ----------------------------
# Paths
# ----------------------------
PARQUET_PATH = pathlib.Path("data/historical.parquet")
CSV_PATH = pathlib.Path("data/historical.csv")
ARTIFACT_DIR = pathlib.Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

CALIBRATOR_PATH = ARTIFACT_DIR / "calibrator_v1.joblib"
MODEL_PATH = ARTIFACT_DIR / "model_v1.xgb"
SPEC_PATH = ARTIFACT_DIR / "model_v1.json"
LOG_PATH = ARTIFACT_DIR / "train.log"
METRICS_JSONL = ARTIFACT_DIR / "metrics.jsonl"
FI_JSON = ARTIFACT_DIR / "feature_importance.json"

# ----------------------------
# Logging
# ----------------------------
def _logger() -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = "%(asctime)s\t%(levelname)s\t%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh)
    return logger

log = _logger()

def _metrics_write(record: dict) -> None:
    with METRICS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ----------------------------
# Data
# ----------------------------
def _load_raw():
    if PARQUET_PATH.exists():
        log.info("Loading parquet %s", PARQUET_PATH.as_posix())
        return pd.read_parquet(PARQUET_PATH)
    if CSV_PATH.exists():
        log.info("Loading CSV %s", CSV_PATH.as_posix())
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError("No historical.{parquet|csv} found under data/")

def load_dataset():
    df = _load_raw()
    log.info("Raw rows=%d cols=%d pos_rate=%.4f", len(df), df.shape[1], float(df["is_fraud"].mean()))
    X = df.apply(lambda r: pd.Series(transform_raw_to_features(r.to_dict())), axis=1)
    X = X.reindex(columns=REALTIME_FEATURES, fill_value=0.0).astype(float)
    y = df["is_fraud"].astype(int)
    log.info("Features shape=%s", X.shape)
    return X, y

# ----------------------------
# Constraints
# ----------------------------
def _pair_category_interactions(feature_names, var_name: str):
    """Create per-tree cliques pairing `var_name` with each one-hot category."""
    cats = [f for f in feature_names if f.startswith("cat_")]
    return [[var_name, c] for c in cats]



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

# ----------------------------
# Interaction audit (amount × category)
# ----------------------------
def audit_var_category_interactions(booster: xgb.Booster, var_name: str) -> dict:
    """
    Count trees where `var_name` and any 'cat_*' both appear.
    Indicates the model can express var×category interaction paths.
    """
    try:
        df_trees = booster.trees_to_dataframe()
    except Exception as e:
        log.warning("trees_to_dataframe failed: %s", e)
        return {}

    trees_with_var = set(df_trees.loc[df_trees.Feature == var_name, "Tree"])
    df_cat = df_trees[df_trees.Feature.str.startswith("cat_", na=False)]
    trees_with_cat = set(df_cat["Tree"])
    overlap = trees_with_var & trees_with_cat

    from collections import defaultdict
    cat_counts = defaultdict(int)
    for t in overlap:
        for c in df_cat.loc[df_cat.Tree == t, "Feature"].unique().tolist():
            cat_counts[c] += 1

    top = sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    out = {
        "var": var_name,
        "n_trees": int(df_trees.Tree.nunique()),
        "trees_with_var": int(len(trees_with_var)),
        "trees_with_category": int(len(trees_with_cat)),
        "trees_with_both": int(len(overlap)),
        "top_var_x_category": top,
    }
    log.info("interaction_audit %s", json.dumps(out))
    return out

# ----------------------------
# Train
# ----------------------------
def train():
    log.info("train_start xgboost=%s numpy=%s pandas=%s", xgb.__version__, np.__version__, pd.__version__)

    X, y = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pos = int(ytr.sum()); neg = int(len(ytr) - pos)
    scale_pos_weight = float((neg / max(pos, 1)) ** 0.5)
    log.info("split pos=%d neg=%d pos_rate=%.5f spw=%.3f", pos, neg, pos/(pos+neg), scale_pos_weight)

    feature_names = REALTIME_FEATURES
    monotone = _monotone_vector(feature_names)
    interactions = _pair_category_interactions(feature_names,"amount") + _pair_category_interactions(feature_names, "txns_last_5m")
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss", "aucpr"],
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
        "interaction_constraints": interactions, 
        "monotone_constraints": monotone,
        "seed": 42,
    }

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=feature_names)

    log.info("Training booster...")
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=4000,
        evals=[(dte, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    best_it = int(getattr(bst, "best_iteration", 0))
    log.info("best_iteration=%d", best_it)

    # Raw margins -> probs for metrics
    y_val_margin = bst.predict(dte, output_margin=True)
    y_val_prob = expit(y_val_margin)
    auc_raw = roc_auc_score(yte, y_val_prob)
    aucpr_raw = average_precision_score(yte, y_val_prob)
    logloss_raw = log_loss(yte, y_val_prob)
    log.info("val_raw auc=%.6f aucpr=%.6f logloss=%.6f", auc_raw, aucpr_raw, logloss_raw)
    _metrics_write({"step": best_it, "split": "val_raw",
                    "auc": float(auc_raw), "aucpr": float(aucpr_raw), "logloss": float(logloss_raw)})

    # Calibration on margins (Platt)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(y_val_margin.reshape(-1, 1), yte)
    y_val_cal = platt.predict_proba(y_val_margin.reshape(-1, 1))[:, 1]
    auc_cal = roc_auc_score(yte, y_val_cal)
    aucpr_cal = average_precision_score(yte, y_val_cal)
    logloss_cal = log_loss(yte, y_val_cal)
    log.info("val_calibrated auc=%.6f aucpr=%.6f logloss=%.6f", auc_cal, aucpr_cal, logloss_cal)
    _metrics_write({"step": best_it, "split": "val_cal",
                    "auc": float(auc_cal), "aucpr": float(aucpr_cal), "logloss": float(logloss_cal)})

    # Persist artifacts
    bst.save_model(MODEL_PATH.as_posix())
    joblib.dump(platt, CALIBRATOR_PATH.as_posix())
    log.info("artifacts_saved model=%s calibrator=%s", MODEL_PATH.name, CALIBRATOR_PATH.name)

    # Thresholds from calibrated scores
    decline_thr = float(np.quantile(y_val_cal, 0.95))
    review_thr = float(np.quantile(y_val_cal, 0.75))
    log.info("thresholds decline=%.6f review=%.6f", decline_thr, review_thr)

    # Feature importance (gain)
    fi_gain = bst.get_score(importance_type="gain")
    fi_sorted = sorted(fi_gain.items(), key=lambda kv: kv[1], reverse=True)
    with FI_JSON.open("w", encoding="utf-8") as f:
        json.dump({"gain": fi_sorted}, f, indent=2)
    log.info("feature_importance_saved %s", FI_JSON.name)

    # Interaction audit
    audit_amt   = audit_var_category_interactions(bst, "amount")
    audit_txn5m = audit_var_category_interactions(bst, "txns_last_5m")

    # Spec
    spec = {
        "feature_order": feature_names,
        "thresholds": {"decline": decline_thr, "review": review_thr},
        "metrics": {
            "val_auc_raw": float(auc_raw),
            "val_aucpr_raw": float(aucpr_raw),
            "val_logloss_raw": float(logloss_raw),
            "val_auc_cal": float(auc_cal),
            "val_aucpr_cal": float(aucpr_cal),
            "val_logloss_cal": float(logloss_cal),
            "best_iteration": best_it,
            "class_pos_rate_train": float(pos / (pos + neg)),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "interaction_audit": {
            "amount_x_category": audit_amt,
            "txns5m_x_category": audit_txn5m
        },
        "model_version": "v1",
        "monotone_constraints": _monotone_vector(feature_names),
        "base_features": BASE_FEATURES,
        "calibrator": {"type": "platt", "path": CALIBRATOR_PATH.name},
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2))
    log.info("spec_written path=%s", SPEC_PATH.name)
    log.info("train_end")

if __name__ == "__main__":
    train()
