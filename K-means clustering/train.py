"""K-Means Clustering: Customer segmentation with RFM.

Groups Superstore customers into segments by Recency (days since last
order), Frequency (number of orders) and Monetary value (total spend),
without any labels. Chooses k with the elbow method and silhouette
score, then names each segment from its profile.

Run:  python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "superstore.csv"
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42
RFM = ["recency_days", "frequency", "monetary"]
K_RANGE = list(range(2, 11))


def load_rfm() -> pd.DataFrame:
    """One row per customer: recency, frequency and monetary value."""
    df = pd.read_csv(DATA, encoding="latin1", parse_dates=["Order Date"])
    snapshot = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        recency_days=("Order Date", lambda d: (snapshot - d.max()).days),
        frequency=("Order ID", "nunique"),
        monetary=("Sales", "sum"),
        profit=("Profit", "sum"),
    )
    return rfm


def make_pipeline(k: int) -> Pipeline:
    """Log-transform (spend and order counts are heavily skewed), scale, then cluster."""
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
    ])


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a business name from its average R, F and M ranks.
    Lower recency is better; higher frequency and monetary are better."""
    score = (profile["recency_days"].rank(ascending=False)
             + profile["frequency"].rank() + profile["monetary"].rank())
    ordered = score.sort_values(ascending=False).index.tolist()
    labels_by_k = {
        2: ["Valuable customers", "Low-value customers"],
        3: ["Champions", "Big spenders cooling off", "Low-value / lapsing"],
        4: ["Champions", "Loyal big spenders", "Needs attention", "At risk / lapsing"],
        5: ["Champions", "Loyal big spenders", "Promising", "Needs attention", "At risk / lapsing"],
    }
    names = labels_by_k.get(len(ordered), [f"Segment {i + 1}" for i in range(len(ordered))])
    return dict(zip(ordered, names))


def train(save: bool = True, k: int | None = None) -> dict:
    rfm = load_rfm()
    X = rfm[RFM]

    # 1. Choose k: inertia (elbow) and silhouette score for k = 2..10
    selection = {"k": K_RANGE, "inertia": [], "silhouette": []}
    for kk in K_RANGE:
        pipe = make_pipeline(kk).fit(X)
        Z = pipe[:-1].transform(X)
        selection["inertia"].append(float(pipe.named_steps["kmeans"].inertia_))
        selection["silhouette"].append(float(silhouette_score(Z, pipe.named_steps["kmeans"].labels_)))

    # Silhouette tends to favour k=2 (a trivial "good vs bad" split). We pick the k
    # in 3..5 with the best silhouette, so segments are fine-grained enough to act on.
    if k is None:
        candidates = {kk: s for kk, s in zip(K_RANGE, selection["silhouette"]) if 3 <= kk <= 5}
        k = max(candidates, key=candidates.get)

    model = make_pipeline(k).fit(X)
    labels = model.named_steps["kmeans"].labels_
    rfm = rfm.assign(cluster=labels)

    profile = rfm.groupby("cluster").agg(
        customers=("frequency", "size"), recency_days=("recency_days", "median"),
        frequency=("frequency", "median"), monetary=("monetary", "median"),
        total_sales=("monetary", "sum"), total_profit=("profit", "sum"))
    names = name_segments(profile)
    profile["segment"] = profile.index.map(names)
    profile["share_of_customers"] = profile["customers"] / profile["customers"].sum()
    profile["share_of_sales"] = profile["total_sales"] / profile["total_sales"].sum()
    profile = profile.sort_values("monetary", ascending=False)

    results = {
        "dataset": {"customers": int(len(rfm)),
                    "rfm_summary": rfm[RFM].describe().round(1).to_dict()},
        "k_selection": selection,
        "chosen_k": int(k),
        "chosen_silhouette": float(selection["silhouette"][K_RANGE.index(k)]),
        "segments": profile.round(3).reset_index().to_dict("records"),
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": model, "names": names, "profile": profile, "rfm": rfm}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
        rfm.to_csv(RESULTS / "customer_segments.csv")
        make_charts(model, rfm, names, profile, selection, k)
    return results


def make_charts(model, rfm, names, profile, selection, k) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(selection["k"], selection["inertia"], "o-")
    axes[0].set_title("Elbow method: inertia (within-cluster spread)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (lower = tighter clusters)")
    axes[1].plot(selection["k"], selection["silhouette"], "o-", color="#ff7f0e")
    axes[1].set_title("Silhouette score (higher = better separated)")
    axes[1].set_xlabel("k")
    for ax in axes:
        ax.axvline(k, color="grey", linestyle="--", label=f"Chosen k = {k}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "choosing_k.png", dpi=120)
    plt.close(fig)

    # 2. Segments in 2D (PCA of the scaled RFM space) and recency vs spend
    Z = model[:-1].transform(rfm[RFM])
    pca = PCA(2, random_state=RANDOM_STATE).fit(Z)
    P = pca.transform(Z)
    C = pca.transform(model.named_steps["kmeans"].cluster_centers_)
    colors = plt.cm.tab10(np.arange(k))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for c in range(k):
        m = rfm["cluster"].values == c
        axes[0].scatter(P[m, 0], P[m, 1], s=14, alpha=0.6, color=colors[c], label=names[c])
        axes[1].scatter(rfm.loc[m, "recency_days"], rfm.loc[m, "monetary"], s=14, alpha=0.6,
                        color=colors[c], label=names[c])
    axes[0].scatter(C[:, 0], C[:, 1], marker="X", s=250, c="black", label="Centroids")
    axes[0].set_title("Customer segments (PCA view of R, F, M)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Days since last order")
    axes[1].set_ylabel("Total spend ($, log scale)")
    axes[1].set_title("Recency vs total spend")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "segments.png", dpi=110)
    plt.close(fig)

    # 3. Profile: share of customers vs share of sales
    p = profile.set_index("segment")
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, p["share_of_customers"] * 100, 0.4, label="% of customers")
    ax.bar(x + 0.2, p["share_of_sales"] * 100, 0.4, label="% of sales")
    ax.set_xticks(x)
    ax.set_xticklabels(p.index, fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Each segment's share of customers vs share of revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "segment_value.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def assign_segment(bundle: dict, recency_days: float, frequency: float, monetary: float):
    """Segment name and distance to every segment centre (in scaled RFM space)."""
    model = bundle["model"]
    row = pd.DataFrame([[recency_days, frequency, monetary]], columns=RFM)
    cluster = int(model.predict(row)[0])
    dist = model.transform(row)[0]  # KMeans.transform = distance to each centroid
    distances = {bundle["names"][c]: round(float(d), 2) for c, d in enumerate(dist)}
    return bundle["names"][cluster], distances


if __name__ == "__main__":
    res = train(save=True)
    print("Customers:", res["dataset"]["customers"])
    for kk, i, s in zip(res["k_selection"]["k"], res["k_selection"]["inertia"],
                        res["k_selection"]["silhouette"]):
        print(f"  k={kk:2d} inertia={i:8.1f} silhouette={s:.3f}")
    print("Chosen k:", res["chosen_k"])
    cols = ["segment", "customers", "recency_days", "frequency", "monetary",
            "share_of_customers", "share_of_sales", "total_profit"]
    print(pd.DataFrame(res["segments"])[cols].to_string(index=False))
