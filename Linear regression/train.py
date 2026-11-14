"""Linear Regression: Diabetes progression predictor.

Trains and compares plain Linear Regression with regularised versions
(Ridge, Lasso) and a polynomial model, picks the best one by
cross-validation, evaluates it on a held-out test set, and saves the
model, charts and metrics.

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
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42

# The raw dataset uses short codes (s1..s6) for blood measurements.
FRIENDLY_NAMES = {
    "age": "age",
    "sex": "sex",
    "bmi": "bmi",
    "bp": "blood_pressure",
    "s1": "total_cholesterol",
    "s2": "ldl",
    "s3": "hdl",
    "s4": "chol_hdl_ratio",
    "s5": "log_triglycerides",
    "s6": "blood_sugar",
}
FEATURES = list(FRIENDLY_NAMES.values())


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """442 diabetes patients: 10 baseline measurements and a disease
    progression score measured one year later (25 = mild, 346 = severe)."""
    data = load_diabetes(as_frame=True, scaled=False)
    X = data.data.rename(columns=FRIENDLY_NAMES)[FEATURES]
    y = data.target.rename("progression")
    return X, y


def candidate_models() -> dict[str, object]:
    """Each model is a pipeline, so scaling is learned on training data only."""
    alphas = {"model__alpha": np.logspace(-3, 3, 25)}
    return {
        "Baseline (predict the mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge (L2)": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            alphas, cv=5, scoring="neg_root_mean_squared_error"),
        "Lasso (L1)": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Lasso(max_iter=50_000))]),
            {"model__alpha": np.logspace(-3, 1.5, 25)}, cv=5, scoring="neg_root_mean_squared_error"),
        "Polynomial (deg 2) + Ridge": GridSearchCV(
            Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                      ("scale", StandardScaler()), ("model", Ridge())]),
            alphas, cv=5, scoring="neg_root_mean_squared_error"),
    }


def unwrap(model):
    """Return the fitted pipeline inside a GridSearchCV (or the model itself)."""
    return model.best_estimator_ if hasattr(model, "best_estimator_") else model


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    fitted = {}
    for name, model in candidate_models().items():
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        model.fit(X_train, y_train)
        fitted[name] = model
        comparison[name] = {
            "cv_rmse_mean": float(cv_rmse.mean()),
            "cv_rmse_std": float(cv_rmse.std()),
            "test": regression_metrics(y_test, model.predict(X_test)),
        }
        if hasattr(model, "best_params_"):
            comparison[name]["best_alpha"] = float(model.best_params_["model__alpha"])

    # Choose by cross-validation on the training set, never by the test set.
    best_name = min((n for n in comparison if not n.startswith("Baseline")),
                    key=lambda n: comparison[n]["cv_rmse_mean"])
    best = unwrap(fitted[best_name])
    y_pred = best.predict(X_test)

    # Coefficients of the plain linear model on standardised features:
    # "how much does the score change per 1 standard deviation of this feature".
    linear = unwrap(fitted["Linear Regression"])
    coefficients = dict(zip(FEATURES, linear.named_steps["model"].coef_.round(2).tolist()))
    lasso = unwrap(fitted["Lasso (L1)"])
    lasso_dropped = [f for f, c in zip(FEATURES, lasso.named_steps["model"].coef_) if abs(c) < 1e-6]

    results = {
        "dataset": {"rows": int(len(X)), "features": FEATURES,
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "comparison": comparison,
        "best_model": best_name,
        "best_test_metrics": regression_metrics(y_test, y_pred),
        "linear_coefficients_per_std": coefficients,
        "lasso_dropped_features": lasso_dropped,
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": best, "name": best_name, "features": FEATURES,
                     "linear": linear, "train_X": X_train}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(comparison, y_test, y_pred, coefficients, best_name)
    return results


def make_charts(comparison, y_test, y_pred, coefficients, best_name) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Model comparison (cross-validated RMSE, lower is better)
    names = list(comparison)
    rmse = [comparison[n]["cv_rmse_mean"] for n in names]
    err = [comparison[n]["cv_rmse_std"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#bbbbbb" if n.startswith("Baseline") else
              ("#1f77b4" if n == best_name else "#9ecae1") for n in names]
    ax.barh(names, rmse, xerr=err, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated RMSE (lower is better)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(IMAGES / "model_comparison.png", dpi=120)
    plt.close(fig)

    # 2. Actual vs predicted + residuals
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, y_pred, alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual progression")
    axes[0].set_ylabel("Predicted progression")
    axes[0].set_title(f"Actual vs predicted ({best_name})")
    axes[0].legend()
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted progression")
    axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residuals: no clear pattern = good fit")
    fig.tight_layout()
    fig.savefig(IMAGES / "predictions.png", dpi=120)
    plt.close(fig)

    # 3. Coefficients
    coef = pd.Series(coefficients).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(coef.index, coef.values, color=["#d62728" if v < 0 else "#2ca02c" for v in coef.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in progression score per +1 standard deviation")
    ax.set_title("What drives diabetes progression? (linear model coefficients)")
    fig.tight_layout()
    fig.savefig(IMAGES / "coefficients.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    """Load the saved model, training it first if it doesn't exist yet."""
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict(bundle: dict, patient: dict) -> float:
    row = pd.DataFrame([patient])[bundle["features"]]
    return float(bundle["model"].predict(row)[0])


if __name__ == "__main__":
    res = train(save=True)
    print(f"{'Model':30s} {'CV RMSE':>9s} {'Test R2':>8s} {'Test RMSE':>10s}")
    for name, r in res["comparison"].items():
        print(f"{name:30s} {r['cv_rmse_mean']:9.1f} {r['test']['r2']:8.3f} {r['test']['rmse']:10.1f}")
    print(f"\nBest model (chosen by CV): {res['best_model']}")
    print("Test metrics:", {k: round(v, 3) for k, v in res["best_test_metrics"].items()})
    print("Lasso dropped:", res["lasso_dropped_features"])
