import json, time
import xgboost as xgb
import redis
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from features import transform_raw_to_features, feature_vector_in_order, REALTIME_FEATURES
from config import settings

app = FastAPI(title=settings.service_name)
rds = redis.Redis.from_url(settings.redis_url, decode_responses=True)

REQS = Counter("fraud_requests_total", "Total requests")
LAT = Histogram("fraud_latency_ms", "latency ms")
CACHE_HIT = Counter("fraud_cache_hits_total", "Cache hits")
CACHE_MISS = Counter("fraud_cache_miss_total", "Cache misses")
MODEL_READY = Gauge("fraud_model_ready", "1 if model loaded")

try:
    booster = xgb.Booster()
    booster.load_model(settings.model_path)
    MODEL_READY.set(1)
except Exception:
    booster = None
    MODEL_READY.set(0)

try:
    with open(settings.model_path.replace(".xgb", ".json")) as f:
        spec = json.load(f)
    thresholds = spec["thresholds"]
    feature_order = spec.get("feature_order", REALTIME_FEATURES)
except Exception:
    thresholds = {"decline": 0.9, "review": 0.7}
    feature_order = REALTIME_FEATURES

class Txn(BaseModel):
    request_id: str = Field(..., description="Idempotency key")
    account_id: str
    merchant_category: str
    amount: float
    device_trust_score: float | None = None
    ip_risk_score: float | None = None
    acct_age_days: float | None = None
    txns_last_5m: float | None = None
    declines_last_24h: float | None = None
    chargebacks_90d: float | None = None

def cache_get(key: str):
    v = rds.get(key)
    if v is None:
        return None
    CACHE_HIT.inc()
    return json.loads(v)

def cache_set(key: str, payload: dict, ttl: int):
    rds.setex(key, ttl, json.dumps(payload))

def fallback_rules(x: dict) -> tuple[str, float, list[str]]:
    reasons = []
    score = 0.5
    if x["amount"] > 5000:
        reasons.append("rule:high_amount")
        score = max(score, 0.75)
    if x["ip_risk_score"] > 0.9:
        reasons.append("rule:high_ip_risk")
        score = max(score, 0.85)
    action = "review" if score >= thresholds["review"] else "allow"
    return action, score, reasons or ["rule:none"]

@app.post("/score")
def score(txn: Txn):
    REQS.inc()
    t0 = time.perf_counter()

    dkey = f"decision:{txn.request_id}"
    cached = cache_get(dkey)
    if cached:
        return cached
    CACHE_MISS.inc()

    fkey = f"features:{txn.account_id}"
    hot = cache_get(fkey)
    if hot is None:
        CACHE_MISS.inc()
        hot = {}
    merged_raw = {**txn.model_dump(), **hot}
    feats = transform_raw_to_features(merged_raw)
    cache_set(fkey, feats, settings.cache_ttl_seconds)

    fv = feature_vector_in_order(feats)

    if booster is None:
        action, score_val, reasons = fallback_rules(feats)
    else:
        d = xgb.DMatrix([fv], feature_names=feature_order)  # strict parity
        proba = float(booster.predict(d)[0])
        score_val = proba
        if score_val >= thresholds["decline"]:
            action = "decline"
        elif score_val >= thresholds["review"]:
            action = "review"
        else:
            action = "allow"
        reasons = [f"model:{settings.model_version}"]

    resp = {
        "request_id": txn.request_id,
        "model_version": settings.model_version if booster else "rules",
        "action": action,
        "score": round(score_val, 6),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "reasons": reasons,
    }
    cache_set(dkey, resp, settings.cache_ttl_seconds)
    with LAT.time():
        return resp

@app.get("/healthz")
def healthz():
    try:
        redis_ok = rds.ping()
    except Exception:
        redis_ok = False
    return {"model_loaded": booster is not None, "redis_ok": redis_ok}

if __name__ == "__main__":
    start_http_server(settings.prometheus_port)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
