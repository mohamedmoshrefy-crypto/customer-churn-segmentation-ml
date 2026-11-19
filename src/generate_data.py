"""Generate a synthetic e-commerce dataset (customers + orders) for the portfolio project."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 5000
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

customers = pd.DataFrame({
    "customer_id": np.arange(1, N + 1),
    "age": rng.integers(18, 70, N),
    "city_tier": rng.choice(["Tier 1", "Tier 2", "Tier 3"], N, p=[0.45, 0.35, 0.20]),
    "acquisition_channel": rng.choice(["Ads", "Organic", "Social", "Referral"], N, p=[0.35, 0.3, 0.25, 0.1]),
    "signup_days_ago": rng.integers(30, 900, N),
})
engage = rng.gamma(2.0, 1.0, N) * np.where(customers.acquisition_channel == "Referral", 1.4, 1.0)
customers["support_tickets"] = rng.poisson(np.clip(2.5 - engage * 0.5, 0.2, None))

rows = []
for cid, e, sd in zip(customers.customer_id, engage, customers.signup_days_ago):
    n = max(1, rng.poisson(e * 3))
    days = np.sort(rng.integers(0, sd, n))
    for d in days:
        rows.append((cid, int(d), round(float(rng.lognormal(3.6, 0.6)), 2), int(rng.random() < 0.08)))
orders = pd.DataFrame(rows, columns=["customer_id", "days_ago", "amount_eur", "returned"])

# churn label: no purchase in the last 90 days, made noisier by engagement and support tickets
last = orders.groupby("customer_id").days_ago.min()
p = 1 / (1 + np.exp(-(last.reindex(customers.customer_id).values / 60 - 2.2 + customers.support_tickets * 0.35 - engage * 0.3)))
customers["churned"] = (rng.random(N) < p).astype(int)

customers.to_csv(OUT / "customers.csv", index=False)
orders.to_csv(OUT / "orders.csv", index=False)
print(f"customers: {len(customers)}, orders: {len(orders)}, churn rate: {customers.churned.mean():.1%}")
