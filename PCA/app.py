"""Gradio app: compress a handwritten digit with PCA and rebuild it.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr
import numpy as np

from train import load_model, reconstruct

bundle = load_model()
pca, clf = bundle["pca"], bundle["classifier"]
TEST_X, TEST_Y = bundle["test_X"], bundle["test_y"]
CUMVAR = np.cumsum(pca.explained_variance_ratio_)


def to_image(pixels: np.ndarray) -> np.ndarray:
    """8x8 values (0-16) -> 160x160 greyscale image, dark ink on white."""
    img = np.clip(pixels.reshape(8, 8), 0, 16) / 16
    img = (255 * (1 - img)).astype(np.uint8)
    return np.kron(img, np.ones((20, 20), dtype=np.uint8))


def run(index, n_components):
    i, n = int(index), int(n_components)
    original = TEST_X[i]
    rebuilt = reconstruct(pca, original[None, :], n)[0]
    mse = float(np.mean((original - rebuilt) ** 2))
    pred_rebuilt = int(clf.predict(rebuilt[None, :])[0])
    text = (f"### Digit **{TEST_Y[i]}** kept as **{n} numbers** instead of 64 pixels\n\n"
            f"- Compression: **{64 / n:.1f}×** smaller\n"
            f"- Information kept: **{CUMVAR[n - 1]:.1%}** of the variance in the training images\n"
            f"- Reconstruction error (MSE): **{mse:.2f}**\n"
            f"- Classifier reading the *rebuilt* image says: **{pred_rebuilt}** "
            f"{'✅' if pred_rebuilt == TEST_Y[i] else '❌'}")
    return to_image(original), to_image(rebuilt), text


demo = gr.Interface(
    fn=run,
    inputs=[gr.Slider(0, len(TEST_X) - 1, value=0, step=1, label="Test image number"),
            gr.Slider(1, 64, value=10, step=1, label="Number of principal components to keep")],
    outputs=[gr.Image(label="Original (64 pixels)"), gr.Image(label="Rebuilt from components"),
             gr.Markdown()],
    title="🔢 Digit Compression (PCA)",
    description="PCA finds the directions in which handwritten digits vary most. Keep only the "
                "first few and rebuild the image: see how few numbers you need.",
    examples=[[0, 2], [0, 10], [0, 30], [3, 5]],
)

if __name__ == "__main__":
    demo.launch()
