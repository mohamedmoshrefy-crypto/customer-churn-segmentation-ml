"""RFM segmentation (K-Means) + churn prediction (Logistic Regression vs Random Forest)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
cust = pd.read_csv(ROOT / "data/customers.csv")
orders = pd.read_csv(ROOT / "data/orders.csv")

# Observation window: only orders older than 90 days are used as features (no leakage from the churn window)
hist = orders[orders.days_ago >= 90]
rfm = hist.groupby("customer_id").agg(recency=("days_ago", "min"), frequency=("days_ago", "size"),
                                      monetary=("amount_eur", "sum"), return_rate=("returned", "mean"))
df = cust.merge(rfm, on="customer_id", how="left")
df[["frequency", "monetary", "return_rate"]] = df[["frequency", "monetary", "return_rate"]].fillna(0)
df["recency"] = df["recency"].fillna(df["signup_days_ago"])
df["avg_basket"] = np.where(df.frequency > 0, df.monetary / df.frequency.replace(0, 1), 0)

# ---- 1. Segmentation ----
X_seg = StandardScaler().fit_transform(np.log1p(df[["recency", "frequency", "monetary"]]))
scores = {k: silhouette_score(X_seg, KMeans(k, n_init=10, random_state=0).fit_predict(X_seg)) for k in range(2, 8)}
k = max(scores, key=scores.get)
df["segment"] = KMeans(k, n_init=10, random_state=0).fit_predict(X_seg)
seg = df.groupby("segment").agg(customers=("customer_id", "size"), recency=("recency", "median"),
                                frequency=("frequency", "median"), monetary=("monetary", "median"),
                                churn_rate=("churned", "mean")).round(2).sort_values("monetary", ascending=False)
seg.to_csv(OUT / "segments.csv")
print(f"Best k = {k} (silhouette {scores[k]:.2f})\n", seg)

fig, ax = plt.subplots(figsize=(7, 4.5))
for s, g in df.groupby("segment"):
    ax.scatter(g.recency, g.monetary, s=6, alpha=.5, label=f"Segment {s}")
ax.set(xlabel="Recency (days)", ylabel="Monetary (EUR)", yscale="log", title="RFM customer segments")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "segments.png", dpi=120)

# ---- 2. Churn prediction ----
num = ["age", "signup_days_ago", "support_tickets", "recency", "frequency", "monetary", "return_rate", "avg_basket"]
cat = ["city_tier", "acquisition_channel"]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr, Xte, ytr, yte = train_test_split(df[num + cat], df.churned, test_size=.25, stratify=df.churned, random_state=0)
models = {"Logistic Regression": LogisticRegression(max_iter=1000),
          "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=0)}
res = {}
for n, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
    res[n] = (roc_auc_score(yte, p.predict_proba(Xte)[:, 1]), p)
    print(f"{n}: ROC-AUC = {res[n][0]:.3f}")
best = max(res, key=lambda n: res[n][0]); pipe = res[best][1]
print(f"\nBest model: {best}\n", classification_report(yte, pipe.predict(Xte), digits=3))

if best == "Random Forest":
    names = num + list(pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat))
    imp = pd.Series(pipe.named_steps["m"].feature_importances_, names).sort_values()[-8:]
    fig, ax = plt.subplots(figsize=(7, 4)); imp.plot.barh(ax=ax, color="#2f5d8a")
    ax.set_title("Top churn drivers (Random Forest)"); fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=120)
pd.Series({n: round(r[0], 3) for n, r in res.items()}, name="roc_auc").to_csv(OUT / "model_scores.csv")
