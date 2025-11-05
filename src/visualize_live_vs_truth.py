# visualize_live_vs_truth.py
# Directly imports testing helpers from test.py and uses them to query the live model.
# Plots simulator 'truth' curves vs live predictions; optional SHAP overlay for 'amount'.
#
# Usage:
#   set SCORE_BASE_URL=http://localhost:8080
#   python visualize_live_vs_truth.py --feature amount --categories electronics luxury groceries --grid 0 2000 101
#   python visualize_live_vs_truth.py --feature ip_risk_score --grid 0 1 101
#   python visualize_live_vs_truth.py --feature acct_age_days --grid 0 3650 121
#
# Notes:
# - Requires a running service exposing /score (base URL from SCORE_BASE_URL or default http://localhost:8080).
# - Uses `from test import _txn, _post, _score` for payloads and scoring.
# - Optional SHAP overlay reads artifacts/model_v1.xgb unless --model-path is provided.
# - Matplotlib only; single-plot policy; no seaborn; one feature per plot.

import os
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Testing helpers (explicit import as requested)
from test import _txn, _post, _score  # noqa: F401

# Default grids per feature; used when caller leaves the generic amount grid in place
FEATURE_DEFAULT_GRID = {
    "amount":             (0.0, 2000.0, 101),
    "ip_risk_score":      (0.0, 1.0,    101),
    "device_trust_score": (0.0, 1.0,    101),
    "acct_age_days":      (0.0, 3650.0, 121),
    "txns_last_5m":       (0.0, 10.0,   11),
    "declines_last_24h":  (0.0, 10.0,   11),
    "chargebacks_90d":    (0.0, 5.0,    6),
}

SIMILAR_REF = dict(
    merchant_category="electronics",
    amount=1000,
    device_trust_score=0.2,
    ip_risk_score=0.6,
    acct_age_days=100.0,
    txns_last_5m=1.0,
    declines_last_24h=0.0,
    chargebacks_90d=0.0,
)

# per-feature absolute windows (tight; adjust if you need more points)
SIMILAR_ATOL = dict(
    amount=500,                 # dollars
    device_trust_score=0.4,    # unitless prob
    ip_risk_score=1.0,         # unitless prob
    acct_age_days=2033.0,         # days
    txns_last_5m=5,           # count
    declines_last_24h=100,      # exact match
    chargebacks_90d=100,        # exact match
)

def _similar_mask(df: pd.DataFrame, ref: Dict, atol: Dict) -> pd.Series:
    m = (df["merchant_category"] == ref["merchant_category"])
    for col, tol in atol.items():
        m &= (df[col].astype(float).sub(float(ref[col])).abs() <= float(tol))
    return m


# Optional SHAP/XGBoost for local overlay
try:
    import xgboost as xgb
    import shap  # type: ignore
except Exception:
    xgb = None
    shap = None

# Optional project feature helpers for SHAP matrix construction
try:
    from features import REALTIME_FEATURES, transform_raw_to_features
except Exception:
    REALTIME_FEATURES = [
        "amount", "merchant_category", "device_trust_score", "ip_risk_score",
        "acct_age_days", "txns_last_5m", "declines_last_24h", "chargebacks_90d"
    ]
    def transform_raw_to_features(raw: Dict) -> Dict:
        # Minimal passthrough
        return {k: raw.get(k) for k in REALTIME_FEATURES}

DEFAULT_BASE_URL = os.getenv("SCORE_BASE_URL", "http://localhost:8080")

MERCHANT_CATEGORIES = [
    "electronics","groceries","fuel","travel","luxury","restaurants",
    "gaming","subscriptions","services","pharmacy","ticketing","apparel"
]

@dataclass
class Fixed:
    acct_age_days: float = 100.0
    device_trust_score: float = 0.5
    ip_risk_score: float = 0.5
    txns_last_5m: float = 1.0
    declines_last_24h: float = 0.0
    chargebacks_90d: float = 0.0
    merchant_category: str = "electronics"
    amount: float = 120.0

def simulator_truth_logit(row: Dict) -> float:
    mc = row["merchant_category"]
    amount = float(row["amount"])
    acct_age_days = float(row["acct_age_days"])
    device_trust = float(row["device_trust_score"])
    ip_risk = float(row["ip_risk_score"])
    tx5m = float(row["txns_last_5m"])
    declines24h = float(row["declines_last_24h"])
    chbk90d = float(row["chargebacks_90d"])
    logits = (
        0.003*(amount - 120.0)
        - 0.0002*acct_age_days
        - 1.2*device_trust
        + 2.0*ip_risk
        + 0.35*tx5m
        + 0.55*declines24h
        + 0.9*chbk90d
        + (0.8 if (mc in {"luxury","electronics","travel"}) and (amount > 800.0) else 0.0)
        + (0.5 if (mc == "gaming") and (tx5m > 3.0) else 0.0)
    )
    return logits


def scatter_empirical_points(
    ax,
    feature: str,
    category: str,
    csv_path="data/historical3.csv",
    ref: Optional[Dict] = None,
    atol: Optional[Dict] = None,
    max_rows: Optional[int] = 200,
    shuffle: bool = True,
):
    """
    Plot empirical points only for rows similar to a reference payload.
    - feature: x-axis feature name
    - category: kept for plot title; ref["merchant_category"] controls filtering
    - ref: dict of canonical values; defaults to SIMILAR_REF
    - atol: per-feature absolute windows; defaults to SIMILAR_ATOL
    """
    if not Path(csv_path).exists():
        print(f"[warn] {csv_path} not found, skipping empirical overlay"); return

    ref = dict(SIMILAR_REF if ref is None else ref)
    # allow caller's category selection to drive ref category
    ref["merchant_category"] = category or ref["merchant_category"]
    atol = dict(SIMILAR_ATOL if atol is None else atol)

    df = pd.read_csv(csv_path)
    # force required columns to float where applicable
    float_cols = ["amount","device_trust_score","ip_risk_score",
                  "acct_age_days","txns_last_5m","declines_last_24h","chargebacks_90d"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=float_cols + ["merchant_category"])

    # keep only similar rows
    mask = _similar_mask(df, ref, atol)
    df_sim = df.loc[mask].copy()
    if df_sim.empty:
        print(f"[warn] No similar rows for merchant_category={ref['merchant_category']} within atol={atol}")
        return

    if shuffle:
        df_sim = df_sim.sample(frac=1.0, random_state=42)
    if max_rows is not None and len(df_sim) > max_rows:
        df_sim = df_sim.iloc[:max_rows]

    xs = df_sim[feature].astype(float).values
    ys_truth, ys_pred = [], []

    with requests.Session() as s:
        for _, row in df_sim.iterrows():
            payload = _txn(
                request_id=f"rid-{uuid.uuid4().hex[:12]}",
                account_id=f"acct-{uuid.uuid4().hex[:12]}",
                merchant_category=row["merchant_category"],
                amount=float(row["amount"]),
                device_trust_score=float(row["device_trust_score"]),
                ip_risk_score=float(row["ip_risk_score"]),
                acct_age_days=float(row["acct_age_days"]),
                txns_last_5m=float(row["txns_last_5m"]),
                declines_last_24h=float(row["declines_last_24h"]),
                chargebacks_90d=float(row["chargebacks_90d"]),
            )
            ys_truth.append(simulator_truth_prob(payload))
            ys_pred.append(live_score(DEFAULT_BASE_URL, payload))

    ys_truth = np.asarray(ys_truth); ys_pred = np.asarray(ys_pred)

    ax.scatter(xs, ys_truth, alpha=0.25, s=12, label=f"empirical truth ~{len(xs)} similar")
    ax.scatter(xs, ys_pred,  alpha=0.25, s=12, label="empirical pred (live)")
    ax.legend()

def simulator_truth_prob(row: Dict) -> float:
    z = simulator_truth_logit(row)
    return float(1.0 / (1.0 + math.exp(-z)))

def make_payload(base: Fixed, **overrides) -> Dict:
    row = {**base.__dict__, **overrides}
    return _txn(**row)

def live_score(base_url: str, payload: Dict) -> float:
    with requests.Session() as s:
        return float(_score(s, base_url, **payload))

def _clamp(feature: str, v: float) -> float:
    # Keep bounded features in-domain; keep counts non-negative
    if feature in {"ip_risk_score", "device_trust_score"}:
        return max(0.0, min(1.0, float(v)))
    if feature in {"txns_last_5m", "declines_last_24h", "chargebacks_90d"}:
        return max(0.0, float(v))
    return float(v)

def sweep_feature(base_url: str, feature: str, grid: np.ndarray, categories: List[str], fixed: Fixed) -> pd.DataFrame:
    rows = []
    for cat in categories:
        for v in grid:
            v = _clamp(feature, v)
            payload = make_payload(fixed, merchant_category=cat, **{feature: float(v)})
            truth_p = simulator_truth_prob(payload)
            pred_p = live_score(base_url, payload)
            rows.append(dict(category=cat, feature=feature, value=float(v), truth_p=truth_p, pred_p=pred_p))
    return pd.DataFrame(rows)

def plot_curves(df: pd.DataFrame, feature: str, category: str, ax=None):
    d = df[(df["feature"] == feature) & (df["category"] == category)].sort_values("value")
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(d["value"], d["truth_p"], label="Truth (sim)", linewidth=2)
    ax.plot(d["value"], d["pred_p"], label="Model (live)", linestyle="--")
    ax.set_xlabel(feature); ax.set_ylabel("Probability")
    ax.set_title(f"{feature} → p(fraud) — {category}")
    ax.grid(True, alpha=0.25); ax.legend()
    return ax

def interactive_slider(df: pd.DataFrame, feature: str, categories: List[str]):
    fig, ax = plt.subplots(); plt.subplots_adjust(bottom=0.25)
    idx = 0; cat = categories[idx]
    d = df[(df["feature"] == feature) & (df["category"] == cat)].sort_values("value")
    truth_line, = ax.plot(d["value"], d["truth_p"], label="Truth (sim)", linewidth=2)
    pred_line,  = ax.plot(d["value"], d["pred_p"], label="Model (live)", linestyle="--")
    ax.set_xlabel(feature); ax.set_ylabel("Probability")
    ax.set_title(f"{feature} → p(fraud) — {cat}")
    ax.grid(True, alpha=0.25); ax.legend()

    s_ax = plt.axes([0.15, 0.08, 0.7, 0.03])
    s = Slider(s_ax, "cat idx", 0, len(categories)-1, valinit=0, valstep=1)
    def on_change(val):
        i = int(s.val); c = categories[i]
        dd = df[(df["feature"] == feature) & (df["category"] == c)].sort_values("value")
        truth_line.set_ydata(dd["truth_p"].values)
        pred_line.set_ydata(dd["pred_p"].values)
        ax.set_title(f"{feature} → p(fraud) — {c}")
        fig.canvas.draw_idle()
    s.on_changed(on_change); plt.show()

def optional_shap_overlay_amount(model_path: Optional[str], df_slice: pd.DataFrame, fixed: Fixed, ax):
    if not model_path or not Path(model_path).exists() or xgb is None or shap is None:
        return
    amounts = df_slice["value"].values
    cats = df_slice["category"].values
    rows = []
    for a, c in zip(amounts, cats):
        rows.append(dict(
            amount=float(a),
            merchant_category=c,
            device_trust_score=fixed.device_trust_score,
            ip_risk_score=fixed.ip_risk_score,
            acct_age_days=fixed.acct_age_days,
            txns_last_5m=fixed.txns_last_5m,
            declines_last_24h=fixed.declines_last_24h,
            chargebacks_90d=fixed.chargebacks_90d,
        ))
    Xr = pd.DataFrame([transform_raw_to_features(r) for r in rows])
    booster = xgb.Booster()
    booster.load_model(model_path)
    explainer = shap.TreeExplainer(booster, feature_perturbation="interventional", model_output="raw")
    dm = xgb.DMatrix(Xr[REALTIME_FEATURES], feature_names=REALTIME_FEATURES)
    sv = explainer.shap_values(dm)  # ndarray [n, d]
    j = REALTIME_FEATURES.index("amount")
    shap_amount = sv[:, j]
    shap_prob = 1.0 / (1.0 + np.exp(-shap_amount))
    ax.plot(df_slice["value"].values, shap_prob, linestyle=":", alpha=0.7, label="SHAP(amount)~prob")
    ax.legend()

def _resolve_grid(feature: str, grid: Tuple[float, float, int]) -> Tuple[float, float, int]:
    # If caller left the generic amount grid in place, substitute feature-aware defaults
    if grid == (0.0, 2000.0, 101) and feature in FEATURE_DEFAULT_GRID:
        return FEATURE_DEFAULT_GRID[feature]
    return grid

def run_all_graphs(
    base_url: str = DEFAULT_BASE_URL,
    feature: str = "amount",
    categories: Optional[List[str]] = None,
    grid: Tuple[float, float, int] = (0.0, 2000.0, 101),
    interactive: bool = True,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    if categories is None:
        categories = ["electronics", "luxury", "groceries"]
    start, stop, num = _resolve_grid(feature, grid)
    values = np.linspace(start, stop, int(num))
    fixed = Fixed()
    df = sweep_feature(base_url, feature, values, categories, fixed)
    if interactive:
        interactive_slider(df, feature, categories)
    else:
        for cat in categories:
            fig, ax = plt.subplots()
            plot_curves(df, feature, cat, ax=ax)
            scatter_empirical_points(ax, feature, cat)
            if feature == "amount":
                sl = df[(df["feature"] == feature) & (df["category"] == cat)]
                optional_shap_overlay_amount(model_path, sl, fixed, ax)
            plt.show()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--feature", default="amount",
                    choices=["amount","ip_risk_score","device_trust_score","acct_age_days","txns_last_5m","declines_last_24h","chargebacks_90d"])
    ap.add_argument("--categories", nargs="*", default=["electronics", "luxury", "groceries"])
    ap.add_argument("--grid", nargs=3, type=float, default=[0.0, 2000.0, 101.0])
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--model-path", default=os.getenv("MODEL_PATH", "artifacts/model_v1.xgb"))
    args = ap.parse_args()

    start, stop, num = args.grid
    run_all_graphs(
        base_url=args.base_url,
        feature=args.feature,
        categories=args.categories,
        grid=(float(start), float(stop), int(num)),
        interactive=(not args.static),
        model_path=args.model_path,
    )

if __name__ == "__main__":
    main()
