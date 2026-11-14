"""Naive Bayes: SMS spam filter.

Classifies text messages as spam or ham (legitimate) using word counts
and Bayes' theorem. Compares three Naive Bayes variants and two text
representations, tunes the smoothing parameter, and benchmarks against
logistic regression.

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
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "sms_spam.csv"
MODEL_PATH = HERE / "models" / "model.joblib"
IMAGES = HERE / "images"
RESULTS = HERE / "results"
RANDOM_STATE = 42


def load_data() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(DATA, encoding="latin1")
    # A few messages contain commas and spilled into extra columns: stitch them back.
    text = df.iloc[:, 1:].fillna("").astype(str).agg(",".join, axis=1).str.rstrip(",")
    df = pd.DataFrame({"text": text, "spam": (df["v1"] == "spam").astype(int)})
    # Remove exact duplicates so the same message can't be in both train and test.
    df = df.drop_duplicates("text").reset_index(drop=True)
    return df["text"], df["spam"]


def vectorizer(kind: str):
    common = dict(lowercase=True, ngram_range=(1, 2), min_df=2)
    return TfidfVectorizer(sublinear_tf=True, **common) if kind == "tfidf" else CountVectorizer(**common)


def candidates() -> dict[str, Pipeline]:
    return {
        "MultinomialNB + word counts": Pipeline([("vec", vectorizer("count")), ("nb", MultinomialNB())]),
        "MultinomialNB + TF-IDF": Pipeline([("vec", vectorizer("tfidf")), ("nb", MultinomialNB())]),
        "ComplementNB + word counts": Pipeline([("vec", vectorizer("count")), ("nb", ComplementNB())]),
        "BernoulliNB (word present?)": Pipeline([
            ("vec", CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, binary=True)),
            ("nb", BernoulliNB())]),
    }


def scores(y, pred, proba) -> dict[str, float]:
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "avg_precision": float(average_precision_score(y, proba))}


def train(save: bool = True) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Which Naive Bayes variant fits SMS text best? (5-fold CV F1)
    variants = {}
    for name, pipe in candidates().items():
        f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        variants[name] = {"cv_f1_mean": float(f1.mean()), "cv_f1_std": float(f1.std())}
    best_variant = max(variants, key=lambda n: variants[n]["cv_f1_mean"])

    # 2. Tune the smoothing strength alpha of the best variant
    alphas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    grid = GridSearchCV(candidates()[best_variant], {"nb__alpha": alphas},
                        cv=cv, scoring="f1", return_train_score=True)
    grid.fit(X_train, y_train)
    nb = grid.best_estimator_
    alpha_curve = {"alpha": alphas,
                   "cv_f1": grid.cv_results_["mean_test_score"].round(4).tolist(),
                   "train_f1": grid.cv_results_["mean_train_score"].round(4).tolist()}

    # 3. Final comparison on the test set
    logreg = Pipeline([("vec", vectorizer("tfidf")),
                       ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))])
    logreg.fit(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    final = {}
    for name, model in [("Baseline (always ham)", baseline),
                        (f"Naive Bayes ({best_variant}, tuned)", nb),
                        ("Logistic Regression + TF-IDF", logreg)]:
        proba = model.predict_proba(X_test)[:, 1]
        final[name] = scores(y_test, (proba >= 0.5).astype(int), proba)

    # 4. The words that most strongly signal spam vs ham
    vec, clf = nb.named_steps["vec"], nb.named_steps["nb"]
    vocab = np.array(vec.get_feature_names_out())
    log_ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(log_ratio)
    spam_words = dict(zip(vocab[order[-15:]][::-1], log_ratio[order[-15:]][::-1].round(2).tolist()))
    ham_words = dict(zip(vocab[order[:15]], log_ratio[order[:15]].round(2).tolist()))

    # Error analysis: which messages fool the filter?
    proba = nb.predict_proba(X_test)[:, 1]
    errors = pd.DataFrame({"text": X_test.values, "spam": y_test.values, "p_spam": proba})
    fn = errors[(errors.spam == 1) & (errors.p_spam < 0.5)].nsmallest(5, "p_spam")
    fp = errors[(errors.spam == 0) & (errors.p_spam >= 0.5)].nlargest(5, "p_spam")

    results = {
        "dataset": {"messages_after_dedup": int(len(X)), "spam_share": float(y.mean()),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                    "vocabulary_size": int(len(vocab))},
        "variants_cv": variants,
        "best_variant": best_variant,
        "best_alpha": float(grid.best_params_["nb__alpha"]),
        "alpha_curve": alpha_curve,
        "test": final,
        "top_spam_words": spam_words,
        "top_ham_words": ham_words,
        "missed_spam_examples": fn[["text", "p_spam"]].round(3).to_dict("records"),
        "false_alarm_examples": fp[["text", "p_spam"]].round(3).to_dict("records"),
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": nb, "name": best_variant}, MODEL_PATH)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
        make_charts(nb, logreg, X_test, y_test, variants, alpha_curve, spam_words, ham_words,
                    results["best_alpha"])
    return results


def make_charts(nb, logreg, X_test, y_test, variants, alpha_curve, spam_words, ham_words,
                best_alpha) -> None:
    IMAGES.mkdir(exist_ok=True)

    # 1. Variant comparison
    names = list(variants)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(names, [variants[n]["cv_f1_mean"] for n in names],
            xerr=[variants[n]["cv_f1_std"] for n in names], color="#ff7f0e")
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("Cross-validated F1 for spam")
    ax.set_title("Naive Bayes variants on SMS text (default smoothing)")
    fig.tight_layout()
    fig.savefig(IMAGES / "variants.png", dpi=120)
    plt.close(fig)

    # 2. Smoothing curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alpha_curve["alpha"], alpha_curve["train_f1"], "o-", label="Training F1")
    ax.plot(alpha_curve["alpha"], alpha_curve["cv_f1"], "o-", label="Cross-validation F1")
    ax.axvline(best_alpha, color="grey", linestyle="--", label=f"Chosen alpha = {best_alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("Smoothing alpha (log scale)")
    ax.set_ylabel("F1 (spam)")
    ax.set_title("Laplace smoothing: too little overfits, too much underfits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "alpha_curve.png", dpi=120)
    plt.close(fig)

    # 3. Spammiest and hammiest words
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s = pd.Series(spam_words).sort_values()
    h = pd.Series(ham_words).sort_values(ascending=False)
    axes[0].barh(s.index, s.values, color="#d62728")
    axes[0].set_title("Words that signal SPAM")
    axes[0].set_xlabel("log P(word|spam) - log P(word|ham)")
    axes[1].barh(h.index, h.values, color="#2ca02c")
    axes[1].set_title("Words that signal HAM (normal messages)")
    axes[1].set_xlabel("log P(word|spam) - log P(word|ham)")
    fig.tight_layout()
    fig.savefig(IMAGES / "top_words.png", dpi=120)
    plt.close(fig)

    # 4. Confusion matrix + precision-recall curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_estimator(nb, X_test, y_test, display_labels=["Ham", "Spam"],
                                          cmap="Oranges", ax=axes[0], colorbar=False)
    axes[0].set_title("Naive Bayes: test set")
    PrecisionRecallDisplay.from_estimator(nb, X_test, y_test, name="Naive Bayes", ax=axes[1])
    PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, name="Logistic Regression", ax=axes[1])
    axes[1].set_title("Precision-recall curves")
    fig.tight_layout()
    fig.savefig(IMAGES / "evaluation.png", dpi=120)
    plt.close(fig)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def explain(bundle: dict, message: str, top: int = 8) -> tuple[float, pd.DataFrame]:
    """P(spam) and the words in this message that pushed the decision most."""
    model = bundle["model"]
    vec, clf = model.named_steps["vec"], model.named_steps["nb"]
    p_spam = float(model.predict_proba([message])[0, 1])
    counts = vec.transform([message])
    idx = counts.nonzero()[1]
    vocab = vec.get_feature_names_out()
    ratio = clf.feature_log_prob_[1, idx] - clf.feature_log_prob_[0, idx]
    weight = ratio * np.asarray(counts[0, idx].todense()).ravel()
    words = pd.DataFrame({"Word / phrase": vocab[idx], "Spam evidence": weight.round(2)})
    words = words.reindex(words["Spam evidence"].abs().sort_values(ascending=False).index)
    return p_spam, words.head(top).reset_index(drop=True)


if __name__ == "__main__":
    res = train(save=True)
    print("Messages:", res["dataset"]["messages_after_dedup"], "spam share:",
          round(res["dataset"]["spam_share"], 3), "vocab:", res["dataset"]["vocabulary_size"])
    for n, v in res["variants_cv"].items():
        print(f"  {n:32s} CV F1 {v['cv_f1_mean']:.3f} ± {v['cv_f1_std']:.3f}")
    print("Best:", res["best_variant"], "alpha:", res["best_alpha"])
    for n, m in res["test"].items():
        print(f"{n:55s} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} "
              f"acc={m['accuracy']:.3f} AP={m['avg_precision']:.3f}")
    print("Spam words:", list(res["top_spam_words"])[:10])
    print("Missed spam:", [e["text"][:60] for e in res["missed_spam_examples"]][:3])
    print("False alarms:", [e["text"][:60] for e in res["false_alarm_examples"]][:3])
