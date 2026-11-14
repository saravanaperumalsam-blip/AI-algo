"""Random Forest: Heart disease risk, forest vs. single tree vs. logistic regression.

Uses the Framingham Heart Study data (same as the Logistic Regression
project) to answer: does an ensemble of hundreds of trees beat a single
tree, and does it beat a simple logistic regression?

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "framingham.csv"
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42
TARGET = "TenYearCHD"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA).drop(columns=["education"])  # not a medical predictor
    return df.drop(columns=[TARGET]), df[TARGET]


def models(cv) -> dict[str, object]:
    """All models handle missing values inside the pipeline (median of training folds)
    and use balanced class weights, because only ~15% of people develop CHD."""
    impute = ("impute", SimpleImputer(strategy="median"))
    return {
        "Logistic Regression": Pipeline([
            impute, ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000))]),
        "Single Decision Tree (tuned)": GridSearchCV(
            Pipeline([impute, ("model", DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE))]),
            {"model__max_depth": [2, 3, 4, 5, 6, 8],
             "model__min_samples_leaf": [10, 25, 50, 100]},
            scoring="roc_auc", cv=cv),
        "Random Forest (default)": Pipeline([
            impute, ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_STATE))]),
        "Random Forest (tuned)": RandomizedSearchCV(
            Pipeline([impute, ("model", RandomForestClassifier(
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE))]),
            {"model__n_estimators": [200, 400, 600],
             "model__max_depth": [4, 6, 8, 10, None],
             "model__min_samples_leaf": [5, 10, 20, 40],
             "model__max_features": ["sqrt", 0.5, 0.8]},
            n_iter=25, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE),
    }


def unwrap(m):
    return m.best_estimator_ if hasattr(m, "best_estimator_") else m


def metrics(model, X, y) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y, proba)),
            "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred))}


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    comparison, fitted = {}, {}
    for name, model in models(cv).items():
        model.fit(X_train, y_train)
        best = unwrap(model)
        cv_auc = cross_val_score(best, X_train, y_train, cv=cv, scoring="roc_auc")
        fitted[name] = best
        comparison[name] = {"cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
                            "train_auc": float(roc_auc_score(y_train, best.predict_proba(X_train)[:, 1])),
                            "test": metrics(best, X_test, y_test)}
        if hasattr(model, "best_params_"):
            comparison[name]["best_params"] = {k.replace("model__", ""): v
                                               for k, v in model.best_params_.items()}

    forest = fitted["Random Forest (tuned)"]
    rf = forest.named_steps["model"]

    # How many trees are enough? Out-of-bag (OOB) score: each tree is tested on
    # the ~37% of training rows it never saw, so no separate validation set is needed.
    X_imp = forest.named_steps["impute"].transform(X_train)
    oob_curve = {"n_trees": [], "oob_auc": []}
    params = {k: v for k, v in rf.get_params().items()
              if k in ("max_depth", "min_samples_leaf", "max_features", "class_weight")}
    for n in [10, 25, 50, 100, 200, 400, 600]:
        m = RandomForestClassifier(n_estimators=n, oob_score=True, n_jobs=-1,
                                   random_state=RANDOM_STATE, **params).fit(X_imp, y_train)
        oob_curve["n_trees"].append(n)
        oob_curve["oob_auc"].append(float(roc_auc_score(y_train, m.oob_decision_function_[:, 1])))

    # Two views of feature importance
    names = list(X.columns)
    impurity = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    perm = permutation_importance(forest, X_test, y_test, scoring="roc_auc",
                                  n_repeats=20, random_state=RANDOM_STATE, n_jobs=1)
    permutation = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False)

    results = {
        "dataset": {"rows": int(len(X)), "chd_share": float(y.mean()),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "comparison": comparison,
        "oob_curve": oob_curve,
        "impurity_importance": impurity.round(4).to_dict(),
        "permutation_importance": permutation.round(4).to_dict(),
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": forest, "features": names, "train_X": X_train}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(fitted, X_test, y_test, comparison, oob_curve, impurity, permutation)
    return results


def make_charts(fitted, X_test, y_test, comparison, oob_curve, impurity, permutation) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. ROC curves on the test set
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC 0.5)")
    ax.set_title("ROC curves: test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "roc_curves.png", dpi=120)
    plt.close(fig)

    # 2. Train vs CV AUC: shows how much each model overfits
    names = list(comparison)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [comparison[n]["train_auc"] for n in names], 0.4, label="Training AUC")
    ax.bar(x + 0.2, [comparison[n]["cv_auc_mean"] for n in names], 0.4,
           yerr=[comparison[n]["cv_auc_std"] for n in names], label="Cross-validated AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Training vs cross-validated AUC (a big gap = overfitting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "train_vs_cv.png", dpi=120)
    plt.close(fig)

    # 3. OOB curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(oob_curve["n_trees"], oob_curve["oob_auc"], "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Number of trees (log scale)")
    ax.set_ylabel("Out-of-bag ROC-AUC")
    ax.set_title("More trees help, then level off")
    fig.tight_layout()
    fig.savefig(IMAGES / "oob_curve.png", dpi=120)
    plt.close(fig)

    # 4. Feature importance: impurity vs permutation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    impurity.head(10).sort_values().plot.barh(ax=axes[0], color="#2ca02c")
    axes[0].set_title("Impurity importance (from training)")
    permutation.head(10).sort_values().plot.barh(ax=axes[1], color="#1f77b4")
    axes[1].set_title("Permutation importance (AUC drop on test set)")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict_with_votes(bundle: dict, person: dict) -> tuple[float, int, int]:
    """Risk score from the forest, plus how many individual trees vote 'high risk'."""
    forest = bundle["model"]
    row = pd.DataFrame([person])[bundle["features"]]
    score = float(forest.predict_proba(row)[0, 1])
    x = forest.named_steps["impute"].transform(row)
    trees = forest.named_steps["model"].estimators_
    votes = int(sum(t.predict(x)[0] == 1 for t in trees))
    return score, votes, len(trees)


if __name__ == "__main__":
    res = train(save=True)
    print(f"CHD share: {res['dataset']['chd_share']:.3f}")
    print(f"{'Model':30s} {'CV AUC':>14s} {'Train AUC':>9s} {'Test AUC':>8s} {'Recall':>6s} {'Prec':>6s} {'F1':>6s}")
    for name, r in res["comparison"].items():
        t = r["test"]
        print(f"{name:30s} {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} {r['train_auc']:9.3f} "
              f"{t['roc_auc']:8.3f} {t['recall']:6.3f} {t['precision']:6.3f} {t['f1']:6.3f}")
        if "best_params" in r:
            print("   best params:", r["best_params"])
    print("OOB:", dict(zip(res["oob_curve"]["n_trees"], [round(v, 3) for v in res["oob_curve"]["oob_auc"]])))
    print("Permutation importance:", dict(list(res["permutation_importance"].items())[:6]))
