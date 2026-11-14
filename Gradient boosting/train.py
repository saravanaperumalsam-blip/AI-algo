"""Gradient Boosting: Superstore order profit prediction.

Predicts the profit (in $) of each order line from what is known when
the order is placed. Compares gradient boosting with linear regression,
a single tree and a random forest, shows how the learning rate and the
number of boosting rounds interact, and explains the model with
permutation importance and partial dependence.

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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "superstore.csv"
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42

CATEGORICAL = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
NUMERIC = ["Sales", "Quantity", "Discount"]
FEATURES = CATEGORICAL + NUMERIC


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA, encoding="latin1")
    return df[FEATURES].copy(), df["Profit"]


def one_hot() -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", "passthrough", NUMERIC)])


def boosting(**params) -> Pipeline:
    """HistGradientBoosting handles categories natively after ordinal encoding."""
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("num", "passthrough", NUMERIC)], verbose_feature_names_out=False)
    cat_mask = [True] * len(CATEGORICAL) + [False] * len(NUMERIC)
    gb = HistGradientBoostingRegressor(categorical_features=cat_mask, random_state=RANDOM_STATE,
                                       **params)
    return Pipeline([("prep", pre), ("gb", gb)])


def reg_metrics(y, pred) -> dict[str, float]:
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Tune gradient boosting
    search = RandomizedSearchCV(
        boosting(),
        {"gb__learning_rate": [0.03, 0.05, 0.1, 0.2],
         "gb__max_iter": [200, 400, 800],
         "gb__max_leaf_nodes": [15, 31, 63],
         "gb__min_samples_leaf": [5, 10, 20, 40],
         "gb__l2_regularization": [0.0, 0.1, 1.0]},
        n_iter=12, cv=cv, scoring="r2", random_state=RANDOM_STATE)
    search.fit(X_train, y_train)
    gb = search.best_estimator_

    # 2. Compare against simpler models (5-fold CV on the training set + test set)
    models = {
        "Baseline (average profit)": DummyRegressor(),
        "Linear Regression": Pipeline([("prep", one_hot()), ("m", LinearRegression())]),
        "Single Decision Tree": Pipeline([("prep", one_hot()), ("m", DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("prep", one_hot()), ("m", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosting (tuned)": gb,
    }
    comparison = {}
    for name, model in models.items():
        cvr = cross_validate(model, X_train, y_train, cv=cv, scoring=("r2", "neg_mean_absolute_error"))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        worst = int(np.argmax(np.abs(y_test.values - pred)))
        comparison[name] = {"cv_r2": float(cvr["test_r2"].mean()),
                            "cv_mae": float(-cvr["test_neg_mean_absolute_error"].mean()),
                            "test": reg_metrics(y_test, pred),
                            # R2 is very sensitive to single extreme orders, so record the worst one
                            "worst_error": {"sales": float(X_test.iloc[worst]["Sales"]),
                                            "actual": float(y_test.iloc[worst]),
                                            "predicted": float(pred[worst])}}

    # 3. Learning rate vs number of boosting rounds (validation curve per round)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                random_state=RANDOM_STATE)
    rounds = {}
    for lr in (0.3, 0.1, 0.03):
        m = boosting(learning_rate=lr, max_iter=800, max_leaf_nodes=31, early_stopping=False)
        m.fit(X_tr, y_tr)
        val = [float(r2_score(y_val, p)) for p in m.named_steps["gb"].staged_predict(
            m.named_steps["prep"].transform(X_val))]
        rounds[str(lr)] = val

    # 4. Explain the model
    perm = permutation_importance(gb, X_test, y_test, scoring="r2", n_repeats=10,
                                  random_state=RANDOM_STATE)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)

    results = {
        "dataset": {"rows": int(len(X)), "profit_summary": y.describe().round(2).to_dict(),
                    "loss_share": float((y < 0).mean())},
        "best_params": {k.replace("gb__", ""): v for k, v in search.best_params_.items()},
        "comparison": comparison,
        "rounds_best": {lr: {"best_round": int(np.argmax(v)) + 1, "best_r2": max(v),
                             "final_r2": v[-1]} for lr, v in rounds.items()},
        "permutation_importance": importance.round(4).to_dict(),
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": gb}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(gb, X_train, X_test, y_test, comparison, rounds, importance)
    return results


def make_charts(gb, X_train, X_test, y_test, comparison, rounds, importance) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Model comparison
    names = list(comparison)
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [comparison[n]["cv_r2"] for n in names]
    ax.barh(names, vals, color=["#bbbbbb", "#9ecae1", "#9ecae1", "#9ecae1", "#e6550d"])
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center")
    ax.set_xlim(-0.05, 1)
    ax.set_xlabel("Cross-validated R² (higher is better)")
    ax.set_title("Predicting order profit: model comparison")
    fig.tight_layout()
    fig.savefig(IMAGES / "model_comparison.png", dpi=120)
    plt.close(fig)

    # 2. Boosting rounds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lr, vals in rounds.items():
        ax.plot(range(1, len(vals) + 1), vals, label=f"learning rate {lr}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of boosting rounds (trees), log scale")
    ax.set_ylabel("Validation R²")
    ax.set_title("Smaller learning rates need more trees but are steadier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "boosting_rounds.png", dpi=120)
    plt.close(fig)

    # 3. Actual vs predicted
    pred = gb.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_test, pred, s=8, alpha=0.4)
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Actual profit ($, symmetric log scale)")
    ax.set_ylabel("Predicted profit ($)")
    ax.set_title("Gradient boosting: test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # 4. Importance + partial dependence of discount and sales
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    importance.sort_values().plot.barh(ax=axes[0], color="#e6550d")
    axes[0].set_title("Permutation importance (drop in test R²)")
    PartialDependenceDisplay.from_estimator(gb, X_train.sample(2000, random_state=RANDOM_STATE),
                                            ["Discount", "Sales"], ax=axes[1:], kind="average")
    axes[1].set_title("Effect of discount on predicted profit")
    axes[2].set_title("Effect of sales value on predicted profit")
    fig.tight_layout()
    fig.savefig(IMAGES / "explain.png", dpi=110)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict_profit(bundle: dict, order: dict) -> float:
    return float(bundle["model"].predict(pd.DataFrame([order])[FEATURES])[0])


def discount_curve(bundle: dict, order: dict, discounts=None) -> pd.DataFrame:
    """Predicted profit for the same order at different discount levels."""
    discounts = discounts if discounts is not None else np.round(np.arange(0, 0.81, 0.05), 2)
    rows = pd.DataFrame([{**order, "Discount": d} for d in discounts])[FEATURES]
    return pd.DataFrame({"Discount": discounts, "Predicted profit": bundle["model"].predict(rows)})


if __name__ == "__main__":
    res = train(save=True)
    print("Profit summary:", {k: res["dataset"]["profit_summary"][k] for k in ("mean", "50%", "min", "max")},
          "loss share:", round(res["dataset"]["loss_share"], 3))
    print("Best params:", res["best_params"])
    for n, r in res["comparison"].items():
        t = r["test"]
        print(f"{n:28s} CV R2={r['cv_r2']:.3f} CV MAE={r['cv_mae']:6.1f} | test R2={t['r2']:.3f} "
              f"MAE={t['mae']:6.1f} RMSE={t['rmse']:6.1f}")
    print("Rounds:", res["rounds_best"])
    print("Importance:", res["permutation_importance"])
