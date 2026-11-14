"""PCA: Handwritten digit compression and visualisation.

Compresses 8x8 handwritten digit images (64 pixel features) into a few
principal components, then shows:
  * how much information each number of components keeps,
  * what the components ("eigen-digits") look like,
  * how well digits can be rebuilt from a compressed version,
  * a 2D map of all digits,
  * that a classifier on ~30 components is about as accurate as one on all 64 pixels.

Run:  python train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42
COMPONENT_STEPS = [2, 5, 10, 15, 20, 30, 40, 64]


def load_data():
    d = load_digits()
    return d.data, d.target  # 1,797 images, 64 pixels each (values 0-16)


def components_for(pca: PCA, share: float) -> int:
    return int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), share) + 1)


def reconstruct(pca: PCA, X: np.ndarray, n: int) -> np.ndarray:
    """Compress X to the first n components and map it back to 64 pixels."""
    Z = (X - pca.mean_) @ pca.components_[:n].T
    return Z @ pca.components_[:n] + pca.mean_


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    # 1. Full PCA on the training images. All pixels share the same 0-16 scale, so no
    #    standardising: scaling would blow up the noise in almost-empty edge pixels.
    pca = PCA(random_state=RANDOM_STATE).fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    needed = {f"{int(s * 100)}%": components_for(pca, s) for s in (0.5, 0.8, 0.9, 0.95, 0.99)}

    # 2. Reconstruction error on test images for each number of components
    recon_error = {n: float(np.mean((X_test - reconstruct(pca, X_test, n)) ** 2))
                   for n in COMPONENT_STEPS}

    # 3. Does compression hurt a classifier? SVM on n components vs all 64 pixels
    downstream = {}
    for n in COMPONENT_STEPS:
        clf = Pipeline([("pca", PCA(n_components=n, random_state=RANDOM_STATE)),
                        ("svm", SVC(C=10, gamma="scale"))])
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        downstream[n] = {"accuracy": float(accuracy_score(y_test, clf.predict(X_test))),
                         "fit_seconds": round(fit_s, 3)}

    # Final classifier: keep 95% of the variance
    final = Pipeline([("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                      ("svm", SVC(C=10, gamma="scale"))]).fit(X_train, y_train)
    final_acc = float(accuracy_score(y_test, final.predict(X_test)))
    final_n = int(final.named_steps["pca"].n_components_)

    results = {
        "dataset": {"images": int(len(X)), "pixels": int(X.shape[1]),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "variance_first_components": pca.explained_variance_ratio_[:10].round(4).tolist(),
        "components_needed": needed,
        "reconstruction_mse": recon_error,
        "classifier_by_components": downstream,
        "final_classifier": {"components": final_n, "test_accuracy": final_acc,
                             "compression": round(64 / final_n, 2)},
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"pca": pca, "classifier": final, "test_X": X_test, "test_y": y_test},
                    MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(pca, cumvar, needed, X_train, y_train, X_test, recon_error, downstream)
    return results


def make_charts(pca, cumvar, needed, X_train, y_train, X_test, recon_error, downstream) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Explained variance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, 65), pca.explained_variance_ratio_, alpha=0.5, label="Each component")
    ax.plot(range(1, 65), cumvar, "r-", label="Cumulative")
    for share, color in [(0.8, "orange"), (0.95, "green")]:
        n = needed[f"{int(share * 100)}%"]
        ax.axhline(share, color=color, linestyle=":", linewidth=1)
        ax.axvline(n, color=color, linestyle=":", linewidth=1)
        ax.text(n + 0.8, share - 0.07, f"{int(share * 100)}% with {n} components", color=color)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Share of variance explained")
    ax.set_title("How much information do the components keep?")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(IMAGES / "explained_variance.png", dpi=120)
    plt.close(fig)

    # 2. Eigen-digits: the mean image and the first 9 components
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    imgs = [pca.mean_] + list(pca.components_[:9])
    titles = ["Mean digit"] + [f"PC {i + 1} ({v:.0%})"
                               for i, v in enumerate(pca.explained_variance_ratio_[:9])]
    for ax, img, t in zip(axes.ravel(), imgs, titles):
        ax.imshow(img.reshape(8, 8), cmap="coolwarm" if t != "Mean digit" else "gray_r")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle("Eigen-digits: the building blocks PCA found (red = add ink, blue = remove ink)")
    fig.tight_layout()
    fig.savefig(IMAGES / "eigen_digits.png", dpi=120)
    plt.close(fig)

    # 3. Reconstructions with increasing numbers of components
    steps = [2, 5, 10, 20, 40, 64]
    samples = X_test[:6]
    fig, axes = plt.subplots(len(samples), len(steps) + 1, figsize=(10, 9.5))
    for r, img in enumerate(samples):
        axes[r, 0].imshow(img.reshape(8, 8), cmap="gray_r")
        for c, n in enumerate(steps, start=1):
            axes[r, c].imshow(reconstruct(pca, img[None, :], n).reshape(8, 8), cmap="gray_r",
                              vmin=0, vmax=16)
    for c, t in enumerate(["Original"] + [f"{n} comps" for n in steps]):
        axes[0, c].set_title(t, fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Rebuilding digits from compressed versions")
    fig.tight_layout()
    fig.savefig(IMAGES / "reconstructions.png", dpi=110)
    plt.close(fig)

    # 4. 2D map of all training digits
    P = pca.transform(X_train)[:, :2]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=y_train, cmap="tab10", s=10, alpha=0.7)
    for d in range(10):
        cx, cy = np.median(P[y_train == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=18, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("1,347 digits squashed from 64 dimensions to 2")
    fig.colorbar(sc, ax=ax, ticks=range(10), label="Digit")
    fig.tight_layout()
    fig.savefig(IMAGES / "digits_2d.png", dpi=120)
    plt.close(fig)

    # 5. Classifier accuracy and reconstruction error vs components
    ns = COMPONENT_STEPS
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ns, [downstream[n]["accuracy"] for n in ns], "o-", color="#1f77b4",
             label="SVM test accuracy")
    ax1.set_ylabel("Test accuracy", color="#1f77b4")
    ax1.set_xlabel("Number of components")
    ax2 = ax1.twinx()
    ax2.plot(ns, [recon_error[n] for n in ns], "s--", color="#d62728", label="Reconstruction error")
    ax2.set_ylabel("Reconstruction MSE", color="#d62728")
    ax1.set_title("Accuracy saturates long before all 64 pixels are needed")
    fig.tight_layout()
    fig.savefig(IMAGES / "accuracy_vs_components.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    res = train(save=True)
    print("Variance of first 5 PCs:", res["variance_first_components"][:5])
    print("Components needed:", res["components_needed"])
    print("Reconstruction MSE:", {k: round(v, 2) for k, v in res["reconstruction_mse"].items()})
    for n, r in res["classifier_by_components"].items():
        print(f"  {n:2d} components: acc={r['accuracy']:.3f} fit={r['fit_seconds']}s")
    print("Final classifier:", res["final_classifier"])
