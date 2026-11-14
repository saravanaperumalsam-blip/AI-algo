"""Support Vector Machine: Breast cancer diagnosis.

Classifies breast tumours as malignant or benign from 30 measurements of
cell nuclei in a fine-needle biopsy image. Compares linear, polynomial
and RBF kernels, tunes C and gamma, and visualises the maximum-margin
idea in 2D.

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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
GAMMA_GRID = [0.0001, 0.001, 0.01, 0.1, 1]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    # scikit-learn codes malignant as 0; flip so 1 = malignant (the class we must catch)
    y = (data.target == 0).astype(int).rename("malignant")
    return data.data, y


def svm(scaled: bool = True, **params) -> Pipeline:
    steps = [("scale", StandardScaler())] if scaled else []
    return Pipeline(steps + [("svm", SVC(**params))])


def evaluate(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    score = model.decision_function(X)
    return {"accuracy": float(accuracy_score(y, pred)),
            "recall_malignant": float(recall_score(y, pred)),
            "precision_malignant": float(precision_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, score)),
            "missed_cancers": int(((pred == 0) & (y == 1)).sum()),
            "false_alarms": int(((pred == 1) & (y == 0)).sum())}


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Scaling and kernel comparison (default C/gamma), 5-fold CV accuracy
    kernels = {
        "RBF, unscaled": svm(False, kernel="rbf"),
        "Linear": svm(kernel="linear"),
        "Polynomial (degree 3)": svm(kernel="poly", degree=3),
        "RBF": svm(kernel="rbf"),
    }
    kernel_cv = {n: float(cross_val_score(m, X_train, y_train, cv=cv).mean())
                 for n, m in kernels.items()}

    # 2. Tune C and gamma for the RBF kernel; keep the whole grid for a heatmap
    grid = GridSearchCV(svm(kernel="rbf"), {"svm__C": C_GRID, "svm__gamma": GAMMA_GRID},
                        cv=cv, scoring="accuracy")
    grid.fit(X_train, y_train)
    heat = (pd.DataFrame(grid.cv_results_)
            .pivot_table(index="param_svm__C", columns="param_svm__gamma", values="mean_test_score"))
    rbf = grid.best_estimator_

    # 3. Tuned linear SVM (C only) and logistic regression for comparison
    lin_grid = GridSearchCV(svm(kernel="linear"), {"svm__C": [0.001, 0.01, 0.1, 1, 10]},
                            cv=cv, scoring="accuracy").fit(X_train, y_train)
    logreg = Pipeline([("scale", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=5000))]).fit(X_train, y_train)

    test = {"SVM, RBF kernel (tuned)": evaluate(rbf, X_test, y_test),
            "SVM, linear kernel (tuned)": evaluate(lin_grid.best_estimator_, X_test, y_test)}
    pred_lr = logreg.predict(X_test)
    test["Logistic Regression"] = {
        "accuracy": float(accuracy_score(y_test, pred_lr)),
        "recall_malignant": float(recall_score(y_test, pred_lr)),
        "precision_malignant": float(precision_score(y_test, pred_lr)),
        "roc_auc": float(roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])),
        "missed_cancers": int(((pred_lr == 0) & (y_test == 1)).sum()),
        "false_alarms": int(((pred_lr == 1) & (y_test == 0)).sum())}

    n_sv = int(rbf.named_steps["svm"].n_support_.sum())
    results = {
        "dataset": {"rows": int(len(X)), "features": int(X.shape[1]),
                    "malignant_share": float(y.mean()),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "kernel_cv_accuracy": kernel_cv,
        "best_rbf_params": {k.replace("svm__", ""): v for k, v in grid.best_params_.items()},
        "best_rbf_cv_accuracy": float(grid.best_score_),
        "best_linear_C": lin_grid.best_params_["svm__C"],
        "support_vectors": {"count": n_sv, "share_of_training": n_sv / len(X_train)},
        "test": test,
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": rbf, "features": list(X.columns), "test_X": X_test,
                     "test_y": y_test}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(X_train, y_train, X_test, y_test, rbf, heat, kernel_cv)
    return results


def make_charts(X_train, y_train, X_test, y_test, rbf, heat, kernel_cv) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Kernel comparison
    names = list(kernel_cv)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(names, [kernel_cv[n] for n in names],
            color=["#bbbbbb"] + ["#17becf"] * (len(names) - 1))
    ax.invert_yaxis()
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("Cross-validated accuracy (default C and gamma)")
    ax.set_title("Kernels, and why SVMs need scaled features")
    for i, n in enumerate(names):
        ax.text(kernel_cv[n] + 0.002, i, f"{kernel_cv[n]:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(IMAGES / "kernels.png", dpi=120)
    plt.close(fig)

    # 2. C x gamma heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heat.values, cmap="viridis", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("gamma (how far each training point's influence reaches)")
    ax.set_ylabel("C (penalty for misclassified points)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="black" if v > 0.85 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="CV accuracy")
    ax.set_title("RBF SVM: tuning C and gamma")
    fig.tight_layout()
    fig.savefig(IMAGES / "c_gamma_heatmap.png", dpi=120)
    plt.close(fig)

    # 3. The margin in 2D: soft (small C) vs hard (large C)
    feats = ["worst radius", "worst concave points"]
    scaler = StandardScaler().fit(X_train[feats])
    Z = scaler.transform(X_train[feats])
    xx, yy = np.meshgrid(np.linspace(Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5, 300),
                         np.linspace(Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5, 300))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, C in zip(axes, [0.01, 100]):
        m = SVC(kernel="linear", C=C).fit(Z, y_train)
        dec = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, dec > 0, alpha=0.15, cmap="coolwarm")
        ax.contour(xx, yy, dec, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
        ax.scatter(Z[:, 0], Z[:, 1], c=y_train, cmap="coolwarm", s=18, edgecolor="k", linewidth=0.3)
        sv = m.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="k", linewidth=1,
                   label=f"{len(sv)} support vectors")
        ax.set_title(f"Linear SVM, C = {C}")
        ax.set_xlabel("worst radius (scaled)")
        ax.set_ylabel("worst concave points (scaled)")
        ax.legend(loc="upper left")
    fig.suptitle("Solid line = decision boundary, dashed = margin. "
                 "Small C: wide, tolerant margin. Large C: narrow, strict margin.")
    fig.tight_layout()
    fig.savefig(IMAGES / "margin.png", dpi=110)
    plt.close(fig)

    # 4. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(rbf, X_test, y_test, display_labels=["Benign", "Malignant"],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Tuned RBF SVM: test set")
    fig.tight_layout()
    fig.savefig(IMAGES / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def diagnose(bundle: dict, sample: dict) -> tuple[str, float]:
    """Predicted class and signed distance from the decision boundary
    (positive = malignant side; |distance| < 1 means inside the margin)."""
    row = pd.DataFrame([sample])[bundle["features"]]
    d = float(bundle["model"].decision_function(row)[0])
    return ("Malignant" if d > 0 else "Benign"), d


if __name__ == "__main__":
    res = train(save=True)
    print("Malignant share:", round(res["dataset"]["malignant_share"], 3))
    for k, v in res["kernel_cv_accuracy"].items():
        print(f"  {k:25s} CV acc {v:.3f}")
    print("Best RBF:", res["best_rbf_params"], round(res["best_rbf_cv_accuracy"], 3),
          "| best linear C:", res["best_linear_C"])
    print("Support vectors:", res["support_vectors"])
    for n, m in res["test"].items():
        print(f"{n:28s} acc={m['accuracy']:.3f} recall={m['recall_malignant']:.3f} "
              f"prec={m['precision_malignant']:.3f} auc={m['roc_auc']:.3f} "
              f"missed={m['missed_cancers']} false_alarms={m['false_alarms']}")
