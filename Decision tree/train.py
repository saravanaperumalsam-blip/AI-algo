"""Decision Tree: Loss-making order detector (Superstore).

Predicts whether a retail order line will LOSE money, before it ships,
from what is known at order time (product, customer segment, region,
shipping mode, sales value, quantity and discount).

Shows the core decision-tree trade-off: an unlimited tree memorises the
training data (overfits), while a tuned/pruned tree generalises and is
small enough to read as business rules.

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "superstore.csv"
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42

CATEGORICAL = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
NUMERIC = ["Sales", "Quantity", "Discount"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "is_loss"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA, encoding="latin1")
    # Profit is what we want to predict, so it must not be used as an input.
    y = (df["Profit"] < 0).astype(int).rename(TARGET)
    return df[FEATURES].copy(), y


def make_pipeline(**tree_params) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ], verbose_feature_names_out=False)
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_params)
    return Pipeline([("prep", pre), ("tree", tree)])


def classification_metrics(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Baseline: always predict "profitable"
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    # 2. Unlimited tree: grows until every leaf is pure -> memorises training data
    full = make_pipeline().fit(X_train, y_train)

    # 3. Tuned tree: depth, leaf size and pruning strength chosen by 5-fold CV (F1)
    grid = GridSearchCV(
        make_pipeline(),
        {
            "tree__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "tree__min_samples_leaf": [1, 5, 10, 20, 50],
            "tree__ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.005],
        },
        scoring="f1", cv=cv, n_jobs=1)
    grid.fit(X_train, y_train)
    tuned = grid.best_estimator_

    # 4. Overfitting curve: train vs CV score as the tree gets deeper
    depths = list(range(1, 21))
    curve = {"depth": depths, "train_f1": [], "cv_f1": []}
    for d in depths:
        cv_scores, tr_scores = [], []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            m = make_pipeline(max_depth=d).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            tr_scores.append(f1_score(y_train.iloc[tr_idx], m.predict(X_train.iloc[tr_idx])))
            cv_scores.append(f1_score(y_train.iloc[va_idx], m.predict(X_train.iloc[va_idx])))
        curve["train_f1"].append(float(np.mean(tr_scores)))
        curve["cv_f1"].append(float(np.mean(cv_scores)))

    feature_names = list(tuned.named_steps["prep"].get_feature_names_out())
    tree = tuned.named_steps["tree"]
    importances = (pd.Series(tree.feature_importances_, index=feature_names)
                   .sort_values(ascending=False))
    rules = export_text(tree, feature_names=feature_names, decimals=3)

    results = {
        "dataset": {"rows": int(len(X)), "loss_share": float(y.mean()),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "best_params": {k.replace("tree__", ""): v for k, v in grid.best_params_.items()},
        "best_cv_f1": float(grid.best_score_),
        "models": {
            "Baseline (always 'profitable')": {
                "test": classification_metrics(baseline, X_test, y_test)},
            "Unlimited tree": {
                "depth": int(full.named_steps["tree"].get_depth()),
                "leaves": int(full.named_steps["tree"].get_n_leaves()),
                "train": classification_metrics(full, X_train, y_train),
                "test": classification_metrics(full, X_test, y_test)},
            "Tuned tree": {
                "depth": int(tree.get_depth()),
                "leaves": int(tree.get_n_leaves()),
                "train": classification_metrics(tuned, X_train, y_train),
                "test": classification_metrics(tuned, X_test, y_test)},
        },
        "top_features": importances[importances > 0].round(3).to_dict(),
        "depth_curve": curve,
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": tuned, "feature_names": feature_names}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        (RESULTS / "tree_rules.txt").write_text(rules)
        make_charts(tuned, feature_names, importances, curve, X_test, y_test, results)
    return results


def make_charts(tuned, feature_names, importances, curve, X_test, y_test, results) -> None:
    IMAGES.mkdir(exist_ok=True)
    tree = tuned.named_steps["tree"]

    # 1. The tree itself (top 3 levels so it stays readable)
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree, feature_names=feature_names, class_names=["Profit", "Loss"],
              filled=True, rounded=True, max_depth=3, fontsize=9, impurity=False,
              proportion=True, ax=ax)
    ax.set_title("Tuned decision tree (top 3 levels)", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES / "tree.png", dpi=110)
    plt.close(fig)

    # 2. Overfitting curve
    best_depth = results["models"]["Tuned tree"]["depth"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["depth"], curve["train_f1"], "o-", label="Training F1")
    ax.plot(curve["depth"], curve["cv_f1"], "o-", label="Cross-validation F1")
    ax.axvline(best_depth, color="grey", linestyle="--", label=f"Tuned depth = {best_depth}")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("F1 score (loss orders)")
    ax.set_title("Deeper trees memorise the training data (overfitting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "overfitting_curve.png", dpi=120)
    plt.close(fig)

    # 3. Feature importance
    top = importances[importances > 0].head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top.index, top.values, color="#2ca02c")
    ax.set_xlabel("Importance (share of impurity reduction)")
    ax.set_title("What makes an order lose money?")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=120)
    plt.close(fig)

    # 4. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(tuned, X_test, y_test, display_labels=["Profit", "Loss"],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Tuned tree: test set")
    fig.tight_layout()
    fig.savefig(IMAGES / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict_with_path(bundle: dict, order: dict) -> tuple[float, list[str]]:
    """Return P(loss) and the human-readable questions the tree asked."""
    model, names = bundle["model"], bundle["feature_names"]
    row = pd.DataFrame([order])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])

    tree = model.named_steps["tree"].tree_
    x = model.named_steps["prep"].transform(row)[0]
    node, steps = 0, []
    while tree.children_left[node] != -1:
        f, t = tree.feature[node], tree.threshold[node]
        name, value = names[f], x[f]
        went_left = value <= t
        if name in NUMERIC:
            sign = "≤" if went_left else ">"
            steps.append(f"{name} = {value:g} → {name} {sign} {t:.3g}")
        else:  # one-hot column: value is 1 (is this category) or 0 (is not)
            steps.append(f"{name}? {'No' if went_left else 'Yes'}")
        node = tree.children_left[node] if went_left else tree.children_right[node]
    return proba, steps


if __name__ == "__main__":
    res = train(save=True)
    print("Loss share:", round(res["dataset"]["loss_share"], 3))
    print("Best params:", res["best_params"], "CV F1:", round(res["best_cv_f1"], 3))
    for name, m in res["models"].items():
        extra = f" depth={m['depth']} leaves={m['leaves']}" if "depth" in m else ""
        tr = f" train_f1={m['train']['f1']:.3f}" if "train" in m else ""
        t = m["test"]
        print(f"{name:32s}{extra}{tr} | test acc={t['accuracy']:.3f} prec={t['precision']:.3f} "
              f"rec={t['recall']:.3f} f1={t['f1']:.3f} auc={t['roc_auc']:.3f}")
    print("Top features:", res["top_features"])
