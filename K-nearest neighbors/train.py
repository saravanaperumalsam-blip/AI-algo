"""K-Nearest Neighbors: Wine cultivar classifier.

Identifies which of three grape cultivars a wine came from using 13
chemical measurements. The key lesson: KNN measures *distance*, so
features on big scales (proline ~ 1000) drown out features on small
scales (hue ~ 1) unless the data is standardised.

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
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42
CLASS_NAMES = ["Cultivar A", "Cultivar B", "Cultivar C"]
K_RANGE = list(range(1, 31))


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_wine(as_frame=True)
    X = data.data.rename(columns={"od280/od315_of_diluted_wines": "od280_od315"})
    return X, data.target


def knn(scaled: bool, **params) -> Pipeline:
    steps = [("scale", StandardScaler())] if scaled else []
    return Pipeline(steps + [("knn", KNeighborsClassifier(**params))])


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. The scaling experiment: CV accuracy for every k, with and without scaling
    k_curve = {"k": K_RANGE, "unscaled": [], "scaled": []}
    for k in K_RANGE:
        for scaled in (False, True):
            acc = cross_val_score(knn(scaled, n_neighbors=k), X_train, y_train, cv=cv).mean()
            k_curve["scaled" if scaled else "unscaled"].append(float(acc))

    # 2. Full tuning of the scaled model: k, distance metric and vote weighting
    grid = GridSearchCV(
        knn(True),
        {"knn__n_neighbors": K_RANGE,
         "knn__metric": ["euclidean", "manhattan"],
         "knn__weights": ["uniform", "distance"]},
        cv=cv, scoring="accuracy")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # 3. Final comparison on the untouched test set
    best_unscaled_k = K_RANGE[int(np.argmax(k_curve["unscaled"]))]
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    unscaled = knn(False, n_neighbors=best_unscaled_k).fit(X_train, y_train)
    test = {}
    for name, model in [("Baseline (most common cultivar)", baseline),
                        (f"KNN unscaled (best k={best_unscaled_k})", unscaled),
                        ("KNN scaled + tuned", best)]:
        pred = model.predict(X_test)
        test[name] = {"accuracy": float(accuracy_score(y_test, pred)),
                      "macro_f1": float(f1_score(y_test, pred, average="macro"))}

    results = {
        "dataset": {"rows": int(len(X)), "features": list(X.columns),
                    "class_counts": y.value_counts().sort_index().tolist(),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "feature_ranges": (X.max() - X.min()).round(2).to_dict(),
        "k_curve": k_curve,
        "best_unscaled": {"k": best_unscaled_k, "cv_accuracy": max(k_curve["unscaled"])},
        "best_params": {k.replace("knn__", ""): v for k, v in grid.best_params_.items()},
        "best_cv_accuracy": float(grid.best_score_),
        "test": test,
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": best, "features": list(X.columns),
                     "train_X": X_train, "train_y": y_train}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(X, X_train, y_train, X_test, y_test, best, k_curve, results)
    return results


def make_charts(X, X_train, y_train, X_test, y_test, best, k_curve, results) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Why scaling matters: raw feature ranges
    ranges = pd.Series(results["feature_ranges"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ranges.index, ranges.values, color="#9467bd")
    ax.set_xscale("log")
    ax.set_xlabel("Range of values (max - min), log scale")
    ax.set_title("Features live on wildly different scales")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_scales.png", dpi=120)
    plt.close(fig)

    # 2. Accuracy vs k, with and without scaling
    best_k = results["best_params"]["n_neighbors"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(K_RANGE, k_curve["unscaled"], "o-", label="Without scaling")
    ax.plot(K_RANGE, k_curve["scaled"], "o-", label="With StandardScaler")
    ax.axvline(best_k, color="grey", linestyle="--", label=f"Chosen k = {best_k}")
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_ylim(0.5, 1.01)
    ax.set_title("Scaling turns a mediocre KNN into an excellent one")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "k_curve.png", dpi=120)
    plt.close(fig)

    # 3. Decision boundaries on two features: small k = jagged (overfit), large k = smooth
    feats = ["proline", "flavanoids"]
    scaler = StandardScaler().fit(X_train[feats])
    Z_train = scaler.transform(X_train[feats])
    xx, yy = np.meshgrid(np.linspace(Z_train[:, 0].min() - 0.5, Z_train[:, 0].max() + 0.5, 300),
                         np.linspace(Z_train[:, 1].min() - 0.5, Z_train[:, 1].max() + 0.5, 300))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, k in zip(axes, [1, best_k, 50]):
        m = KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train)
        zz = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.25, cmap="viridis")
        ax.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=25)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("proline (scaled)")
        ax.set_ylabel("flavanoids (scaled)")
    fig.suptitle("Decision boundaries: k=1 overfits, very large k underfits")
    fig.tight_layout()
    fig.savefig(IMAGES / "decision_boundaries.png", dpi=110)
    plt.close(fig)

    # 4. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=CLASS_NAMES,
                                          cmap="Purples", ax=ax, colorbar=False)
    ax.set_title("Tuned KNN: test set")
    fig.tight_layout()
    fig.savefig(IMAGES / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict_with_neighbours(bundle: dict, wine: dict) -> tuple[str, dict, pd.DataFrame]:
    """Predicted cultivar, class probabilities and the k most similar training wines."""
    model = bundle["model"]
    row = pd.DataFrame([wine])[bundle["features"]]
    probs = model.predict_proba(row)[0]
    pred = CLASS_NAMES[int(np.argmax(probs))]
    dist, idx = model.named_steps["knn"].kneighbors(model.named_steps["scale"].transform(row))
    neighbours = pd.DataFrame({
        "Neighbour": range(1, len(idx[0]) + 1),
        "Distance": dist[0].round(2),
        "Cultivar": [CLASS_NAMES[c] for c in bundle["train_y"].iloc[idx[0]]],
        "Alcohol": bundle["train_X"].iloc[idx[0]]["alcohol"].values,
        "Proline": bundle["train_X"].iloc[idx[0]]["proline"].values,
    })
    return pred, dict(zip(CLASS_NAMES, probs.round(3).tolist())), neighbours


if __name__ == "__main__":
    res = train(save=True)
    print("Class counts:", res["dataset"]["class_counts"])
    print("Best unscaled:", res["best_unscaled"])
    print("Best scaled params:", res["best_params"], "CV acc:", round(res["best_cv_accuracy"], 3))
    for name, m in res["test"].items():
        print(f"{name:40s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
