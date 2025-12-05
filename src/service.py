import json
import time
from typing import Optional, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from config import settings

app = FastAPI(title=settings.service_name)
rds = redis.Redis.from_url(settings.redis_url, decode_responses=True)

# ----------------- Metrics -----------------
REQS = Counter("fraud_requests_total", "Total requests")
CACHE_HIT = Counter("fraud_cache_hits_total", "Cache hits")
CACHE_MISS = Counter("fraud_cache_miss_total", "Cache misses")
MODEL_READY = Gauge("fraud_model_ready", "1 if model loaded")
CAL_READY = Gauge("fraud_calibrator_ready", "1 if calibrator loaded")
FEATURE_MISMATCH = Counter("fraud_feature_mismatch_total", "Feature shape mismatches")
LAT_MS = Histogram("fraud_latency_ms", "Request latency (ms)",
                   buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000))
SCORE = Histogram("fraud_scores", "Calibrated score distribution",
                  buckets=[i / 20.0 for i in range(21)])

# ----------------- Artifacts -----------------
MODEL_PATH = settings.model_path
SPEC_PATH = MODEL_PATH.replace(".xgb", ".json")
CAL_PATH_FALLBACK = MODEL_PATH.replace(".xgb", ".joblib")

try:
    with open(SPEC_PATH, "r") as f:
        SPEC = json.load(f)
except Exception:
    SPEC = {}

FEATURE_ORDER = SPEC.get("feature_order") or []
THRESHOLDS = SPEC.get("thresholds", {"decline": 0.9, "review": 0.7})
AMT_EDGES = np.array(SPEC.get("amount_bucket_edges", []), dtype=float) \
    if "amount_bucket_edges" in SPEC else None
AMT_PREFIX = SPEC.get("amount_bucket_prefix", "amt_bin_")
CAL_META = SPEC.get("calibrator", {}) or {}

# ----------------- Load model + calibrator -----------------
booster : Optional[xgb.Booster] = None
calibrator = None

try:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    MODEL_READY.set(1)
except Exception:
    booster = None
    MODEL_READY.set(0)

try:
    import os
    cal_path = CAL_META.get("path")
    if cal_path:
        cal_path = os.path.join(os.path.dirname(MODEL_PATH), cal_path)
    else:
        cal_path = CAL_PATH_FALLBACK
    calibrator = joblib.load(cal_path)
    CAL_READY.set(1)
except Exception:
    calibrator = None
    CAL_READY.set(0)

# ----------------- Schema -----------------
class Txn(BaseModel):
    request_id: str = Field(..., description="Idempotency key")
    account_id: str
    merchant_category: str
    amount: float
    device_trust_score: Optional[float] = None
    ip_risk_score: Optional[float] = None
    acct_age_days: Optional[int] = None
    txns_last_5m: Optional[int] = None
    declines_last_24h: Optional[int] = None
    chargebacks_90d: Optional[int] = None

# ----------------- Cache helpers -----------------
def cache_get(key: str):
    v = rds.get(key)
    if v is None:
        return None
    CACHE_HIT.inc()
    return json.loads(v)

def cache_set(key: str, payload: dict, ttl: int):
    rds.setex(key, ttl, json.dumps(payload))

# ----------------- Feature helpers -----------------
def _amount_bucket_ohe(amount: float) -> Dict[str, float]:
    if AMT_EDGES is None or len(AMT_EDGES) == 0:
        return {}
    s = pd.Series([max(float(amount), 0.0)], dtype=float)
    cats = pd.cut(s, AMT_EDGES, right=True, include_lowest=True)
    cols = []
    for i in range(len(AMT_EDGES) - 1):
        lo = int(AMT_EDGES[i])
        hi = "inf" if not np.isfinite(AMT_EDGES[i + 1]) else int(AMT_EDGES[i + 1])
        cols.append(f"{AMT_PREFIX}{lo}_{hi}")
    vec = dict.fromkeys(cols, 0.0)
    idx = cats.cat.codes.iloc[0]
    if idx >= 0:
        vec[cols[int(idx)]] = 1.0
    return vec

def _mc_onehots(label: str) -> Dict[str, float]:
    if not FEATURE_ORDER:
        return {}
    cat_cols = [c for c in FEATURE_ORDER if c.startswith("cat_")]
    if not cat_cols:
        return {}
    expected = {c.split("cat_", 1)[1]: c for c in cat_cols}
    vec = dict.fromkeys(cat_cols, 0.0)
    col = expected.get(label)
    if col:
        vec[col] = 1.0
    return vec

def build_vector(raw: dict) -> pd.DataFrame:
    amt = float(raw.get("amount", 0.0))
    fv = {
        "amount": amt,
        "acct_age_days": float(raw.get("acct_age_days", 0) or 0),
        "device_trust_score": float(raw.get("device_trust_score", 0.0) or 0.0),
        "ip_risk_score": float(raw.get("ip_risk_score", 0.0) or 0.0),
        "txns_last_5m": float(raw.get("txns_last_5m", 0) or 0),
        "declines_last_24h": float(raw.get("declines_last_24h", 0) or 0),
        "chargebacks_90d": float(raw.get("chargebacks_90d", 0) or 0),
    }
    fv["log_amount"] = float(np.log1p(amt))
    fv.update(_amount_bucket_ohe(amt))
    fv.update(_mc_onehots(str(raw.get("merchant_category", ""))))

    if not FEATURE_ORDER:
        raise RuntimeError("feature_order_missing")
    row = {c: 0.0 for c in FEATURE_ORDER}
    for k, v in fv.items():
        if k in row:
            row[k] = float(v)
    X = pd.DataFrame([row], columns=FEATURE_ORDER, dtype=np.float32)

    if booster is not None:
        n_model = int(booster.num_features())
        if X.shape[1] != n_model:
            FEATURE_MISMATCH.inc()
            raise RuntimeError(f"feature_count_mismatch: X={X.shape[1]} model={n_model}")
    return X

# ----------------- Rules fallback -----------------
def fallback_rules(x: dict) -> tuple[str, float, list[str]]:
    reasons = []
    score = 0.5
    if x.get("amount", 0) > 5000:
        reasons.append("rule:high_amount")
        score = max(score, 0.75)
    if float(x.get("ip_risk_score", 0) or 0) > 0.9:
        reasons.append("rule:high_ip_risk")
        score = max(score, 0.85)
    action = "review" if score >= THRESHOLDS["review"] else "allow"
    return action, score, reasons or ["rule:none"]

# ----------------- Endpoint -----------------
@app.post("/score")
def score(txn: Txn):
    REQS.inc()
    t0 = time.perf_counter()

    dkey = f"decision:{txn.request_id}"
    cached = cache_get(dkey)
    if cached:
        LAT_MS.observe((time.perf_counter() - t0) * 1000.0)
        return cached
    CACHE_MISS.inc()

    raw = txn.model_dump()
    try:
        X = build_vector(raw)
    except Exception as e:
        LAT_MS.observe((time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=500, detail=f"feature_build_failed:{type(e).__name__}")
    fkey = f"txn_raw:{txn.request_id}"
    try:
        rds.setex(fkey, settings.cache_ttl_seconds, json.dumps(raw))
    except Exception:
        pass

    if booster is None:
        action, score_val, reasons = fallback_rules(raw)
    else:
        try:
            dmat = xgb.DMatrix(X, feature_names=list(X.columns))
            if calibrator is not None:
                # Calibrate on the RAW MARGIN (matches training)
                margin = float(booster.predict(dmat, output_margin=True)[0])
                proba = float(
                    calibrator.predict_proba(np.array([[margin]], dtype=np.float64))[:, 1][0]
                )
            else:
                proba = float(booster.predict(dmat)[0])

            score_val = proba
            SCORE.observe(score_val)

            if score_val >= THRESHOLDS["decline"]:
                action = "decline"
            elif score_val >= THRESHOLDS["review"]:
                action = "review"
            else:
                action = "allow"
            reasons = [f"model:{settings.model_version}"]
        except Exception:
            LAT_MS.observe((time.perf_counter() - t0) * 1000.0)
            raise HTTPException(status_code=500, detail="scoring_failed")

    resp = {
        "request_id": txn.request_id,
        "model_version": settings.model_version if booster else "rules",
        "action": action,
        "score": round(score_val, 6),
        "latency_ms": int((time.perf_counter() - t0) * 1000.0),
        "reasons": reasons,
    }
    cache_set(dkey, resp, settings.cache_ttl_seconds)
    LAT_MS.observe((time.perf_counter() - t0) * 1000.0)
    return resp

@app.get("/healthz")
def healthz():
    try:
        redis_ok = rds.ping()
    except Exception:
        redis_ok = False
    model_loaded = booster is not None
    feature_spec = len(FEATURE_ORDER)
    model_feats = int(booster.num_features()) if model_loaded else 0
    return {
        "model_loaded": model_loaded,
        "redis_ok": redis_ok,
        "feature_order_count": feature_spec,
        "model_feature_count": model_feats,
        "calibrator": bool(calibrator),
        "thresholds": THRESHOLDS,
    }

if __name__ == "__main__":
    start_http_server(settings.prometheus_port)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
