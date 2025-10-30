from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

merchant_categories = [
    "electronics","groceries","fuel","travel","luxury","restaurants",
    "gaming","subscriptions","services","pharmacy","ticketing","apparel"
]

def simulate_row():
    mc = rng.choice(merchant_categories, p=[0.12,0.18,0.07,0.06,0.04,0.14,0.05,0.08,0.12,0.06,0.04,0.04])
    amount = np.round(rng.gamma(2.0, 60.0), 2)
    acct_age_days = rng.integers(1, 3650)
    device_trust = np.clip(rng.normal(0.65, 0.15), 0, 1)
    ip_risk = np.clip(rng.beta(1.5, 6.0), 0, 1)
    tx5m = rng.poisson(0.6)
    declines24h = rng.poisson(0.1)
    chbk90d = rng.poisson(0.02)
    logits = (
        0.003*(amount-120)
        - 0.0002*acct_age_days
        - 1.2*device_trust
        + 2.0*ip_risk
        + 0.35*tx5m
        + 0.55*declines24h
        + 0.9*chbk90d
        + (0.8 if mc in {"luxury","electronics","travel"} and amount>800 else 0.0)
        + (0.5 if mc=="gaming" and tx5m>3 else 0.0)
    )
    p = 1/(1+np.exp(-logits))
    is_fraud = rng.binomial(1, np.clip(p, 0.01, 0.7))
    return dict(
        merchant_category=mc,
        amount=float(amount),
        acct_age_days=float(acct_age_days),
        device_trust_score=float(device_trust),
        ip_risk_score=float(ip_risk),
        txns_last_5m=float(tx5m),
        declines_last_24h=float(declines24h),
        chargebacks_90d=float(chbk90d),
        is_fraud=int(is_fraud),
    )

rows = [simulate_row() for _ in range(N)]
df = pd.DataFrame(rows)

Path("data").mkdir(parents=True, exist_ok=True)
csv_path = Path("data/historical.csv")
parquet_path = Path("data/historical.parquet")
df.to_csv(csv_path, index=False)
df.to_parquet(parquet_path, index=False)
print({"rows": len(df), "positive_rate": float(df.is_fraud.mean())})
