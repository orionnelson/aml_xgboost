from typing import Dict, Any, List

# Fixed category vocabulary (same as your simulator)
CATEGORY_VALUES: List[str] = [
    "electronics","groceries","fuel","travel","luxury","restaurants",
    "gaming","subscriptions","services","pharmacy","ticketing","apparel"
]

BASE_FEATURES: List[str] = [
    "amount",
    "device_trust_score",
    "ip_risk_score",
    "acct_age_days",
    "txns_last_5m",
    "declines_last_24h",
    "chargebacks_90d",
]

# One-hot column names (prefix "cat_")
CAT_FEATURES: List[str] = [f"cat_{c}" for c in CATEGORY_VALUES]

# Single canonical feature list used by training AND serving
REALTIME_FEATURES: List[str] = BASE_FEATURES + CAT_FEATURES

def transform_raw_to_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic mapping → REALTIME_FEATURES."""
    out: Dict[str, float] = {}

    # continuous
    out["amount"] = float(raw.get("amount", 0.0))
    out["device_trust_score"] = float(raw.get("device_trust_score", 0.5))
    out["ip_risk_score"] = float(raw.get("ip_risk_score", 0.5))
    out["acct_age_days"] = float(raw.get("acct_age_days", 0.0))
    out["txns_last_5m"] = float(raw.get("txns_last_5m", 0.0))
    out["declines_last_24h"] = float(raw.get("declines_last_24h", 0.0))
    out["chargebacks_90d"] = float(raw.get("chargebacks_90d", 0.0))

    # one-hot category (closed vocabulary; unseen → all zeros)
    mc = str(raw.get("merchant_category", "")).strip().lower()
    for col in CAT_FEATURES:
        out[col] = 1.0 if col == f"cat_{mc}" else 0.0

    return out

def feature_vector_in_order(feats: Dict[str, float]):
    return [feats[name] for name in REALTIME_FEATURES]