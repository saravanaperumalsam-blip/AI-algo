"""Gradio app: identify a wine's cultivar and show its nearest neighbours.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr

from train import load_data, load_model, predict_with_neighbours

bundle = load_model()
X, _ = load_data()
FEATURES = bundle["features"]


def run(*values):
    wine = dict(zip(FEATURES, values))
    pred, probs, neighbours = predict_with_neighbours(bundle, wine)
    k = bundle["model"].named_steps["knn"].n_neighbors
    text = (f"### 🍷 Predicted: **{pred}**\n\n"
            f"The {k} most similar wines in the training data are listed below. "
            f"Closer neighbours (smaller distance) get a bigger vote.")
    return text, probs, neighbours


def example(i):
    return [float(v) for v in X.iloc[i][FEATURES]]


demo = gr.Interface(
    fn=run,
    inputs=[gr.Slider(float(X[f].min()), float(X[f].max()), value=float(X[f].median()),
                      label=f.replace("_", " ").capitalize()) for f in FEATURES],
    outputs=[gr.Markdown(), gr.Label(label="Vote share per cultivar"),
             gr.Dataframe(label="Nearest neighbours")],
    title="🍷 Wine Cultivar Classifier (K-Nearest Neighbors)",
    description="Set 13 chemical measurements of a wine. KNN finds the most similar wines it "
                "has seen and lets them vote on the cultivar.",
    examples=[example(0), example(70), example(140)],
)

if __name__ == "__main__":
    demo.launch()
