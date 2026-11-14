"""Gradio app: SVM breast tumour diagnosis on real test-set patients.

Pick a patient the model has never seen, then adjust three key
measurements to watch the prediction move across the decision boundary.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr

from train import diagnose, load_model

bundle = load_model()
TEST_X, TEST_Y = bundle["test_X"].reset_index(drop=True), bundle["test_y"].reset_index(drop=True)
KEY = ["worst radius", "worst concave points", "worst texture"]
CHOICES = [f"Patient {i + 1} (biopsy result: {'Malignant' if y else 'Benign'})"
           for i, y in enumerate(TEST_Y)]


def load_patient(choice):
    i = CHOICES.index(choice)
    return [float(TEST_X.loc[i, f]) for f in KEY]


def run(choice, radius, concave, texture):
    i = CHOICES.index(choice)
    sample = TEST_X.loc[i].to_dict()
    sample.update(dict(zip(KEY, [radius, concave, texture])))
    label, d = diagnose(bundle, sample)
    icon = "🔴" if label == "Malignant" else "🟢"
    if abs(d) < 1:
        conf = "⚠️ Inside the margin: a borderline case that should get a closer look."
    else:
        conf = "Outside the margin: a confident prediction."
    truth = "Malignant" if TEST_Y[i] else "Benign"
    return (f"### {icon} Predicted: **{label}**\n\n"
            f"Distance from the decision boundary: **{d:+.2f}** "
            f"(positive = malignant side, negative = benign side)\n\n{conf}\n\n"
            f"Actual biopsy result for this patient: **{truth}**")


with gr.Blocks(title="SVM Breast Cancer Diagnosis") as demo:
    gr.Markdown("# 🎗️ Breast Tumour Diagnosis (Support Vector Machine)\n"
                "Choose a patient from the held-out test set, then adjust three telling "
                "measurements and watch the **distance from the decision boundary** change. "
                "The model uses all 30 measurements, so changing three moves the score but "
                "often won't flip the diagnosis on its own: related measurements (mean radius, "
                "area, perimeter…) still point the same way. "
                "Educational project only, not medical advice.")
    patient = gr.Dropdown(CHOICES, value=CHOICES[0], label="Test-set patient")
    with gr.Row():
        sliders = [gr.Slider(float(TEST_X[f].min() * 0.8), float(TEST_X[f].max() * 1.1),
                             value=float(TEST_X.loc[0, f]), label=f) for f in KEY]
    btn = gr.Button("Diagnose", variant="primary")
    out = gr.Markdown()
    patient.change(load_patient, patient, sliders)
    btn.click(run, [patient] + sliders, out, api_name="diagnose")

if __name__ == "__main__":
    demo.launch()
