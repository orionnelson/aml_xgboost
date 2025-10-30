import hashlib
from typing import Dict, Any, List

REALTIME_FEATURES: List[str] = [
    "amount",
    "merchant_category",
    "device_trust_score",
    "ip_risk_score",
    "acct_age_days",
    "txns_last_5m",
    "declines_last_24h",
    "chargebacks_90d",
]

def _hash_str(s: str) -> int:
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest(), 16)

def transform_raw_to_features(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["amount"] = float(raw.get("amount", 0.0))
    out["device_trust_score"] = float(raw.get("device_trust_score", 0.5))
    out["ip_risk_score"] = float(raw.get("ip_risk_score", 0.5))
    out["acct_age_days"] = float(raw.get("acct_age_days", 0.0))
    out["txns_last_5m"] = float(raw.get("txns_last_5m", 0.0))
    out["declines_last_24h"] = float(raw.get("declines_last_24h", 0.0))
    out["chargebacks_90d"] = float(raw.get("chargebacks_90d", 0.0))
    mc = str(raw.get("merchant_category", "UNK"))
    out["merchant_category"] = float(_hash_str(mc) % 1000)  # fixed hashing bucket
    return out

def feature_vector_in_order(feats: Dict[str, float]):
    return [feats[name] for name in REALTIME_FEATURES]
