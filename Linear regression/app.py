"""Gradio app: predict diabetes progression one year ahead.

Run:  python app.py   (trains the model first if it hasn't been trained)
"""
from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd

from train import FEATURES, load_model, predict

bundle = load_model()
train_X: pd.DataFrame = bundle["train_X"]
linear = bundle["linear"]

LABELS = {
    "age": "Age (years)",
    "bmi": "Body mass index (BMI)",
    "blood_pressure": "Average blood pressure (mm Hg)",
    "total_cholesterol": "Total cholesterol (mg/dL)",
    "ldl": "LDL, 'bad' cholesterol (mg/dL)",
    "hdl": "HDL, 'good' cholesterol (mg/dL)",
    "chol_hdl_ratio": "Total cholesterol / HDL ratio",
    "log_triglycerides": "Log of triglycerides",
    "blood_sugar": "Blood sugar (mg/dL)",
}


def explain(patient: dict) -> pd.DataFrame:
    """How far each feature pushes this patient's score up or down,
    using the plain linear model: coefficient x standardised value."""
    scaler = linear.named_steps["scale"]
    coef = linear.named_steps["model"].coef_
    z = scaler.transform(pd.DataFrame([patient])[FEATURES])[0]
    contrib = pd.DataFrame({"Feature": FEATURES, "Effect on score": np.round(coef * z, 1)})
    contrib = contrib.reindex(contrib["Effect on score"].abs().sort_values(ascending=False).index)
    return contrib.head(5).reset_index(drop=True)


def run(sex, *values):
    patient = dict(zip([f for f in FEATURES if f != "sex"], values))
    patient["sex"] = int(sex)
    score = predict(bundle, patient)
    q25, q75 = np.percentile(bundle["model"].predict(train_X), [25, 75])
    if score < q25:
        band = "🟢 Lower than most patients"
    elif score > q75:
        band = "🔴 Higher than most patients"
    else:
        band = "🟡 Around the typical range"
    text = (f"### Predicted progression score: **{score:.0f}**\n\n{band} "
            f"(typical range {q25:.0f}–{q75:.0f}; scores run from about 25 to 346)\n\n"
            f"Model: {bundle['name']}")
    return text, explain(patient)


inputs = [gr.Radio([("Group 1", 1), ("Group 2", 2)], value=1,
                   label="Sex (coded 1 or 2 in the dataset)")]
for f in FEATURES:
    if f == "sex":
        continue
    col = train_X[f]
    inputs.append(gr.Slider(float(col.min()), float(col.max()), value=float(col.median()),
                            label=LABELS[f]))

demo = gr.Interface(
    fn=run,
    inputs=inputs,
    outputs=[gr.Markdown(), gr.Dataframe(label="Top 5 factors for this patient (+ raises, - lowers)")],
    title="🩺 Diabetes Progression Predictor (Linear Regression)",
    description=("Predicts a diabetes progression score one year after these baseline "
                 "measurements. Educational project only, not medical advice."),
)

if __name__ == "__main__":
    demo.launch()
