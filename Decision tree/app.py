"""Gradio app: will this order lose money? Shows the tree's reasoning.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr
import pandas as pd

from train import DATA, load_model, predict_with_path

bundle = load_model()
_df = pd.read_csv(DATA, encoding="latin1")
SUBCATS = (_df.groupby("Sub-Category")["Category"].first()).to_dict()


def run(ship_mode, segment, region, sub_category, sales, quantity, discount):
    order = {
        "Ship Mode": ship_mode, "Segment": segment, "Region": region,
        "Category": SUBCATS[sub_category], "Sub-Category": sub_category,
        "Sales": sales, "Quantity": quantity, "Discount": discount,
    }
    p_loss, steps = predict_with_path(bundle, order)
    verdict = ("🔴 **Likely to LOSE money**" if p_loss >= 0.5
               else "🟢 **Likely to be profitable**")
    path = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return (f"### {verdict}\n\nChance of a loss: **{p_loss:.0%}** "
            f"(category: {order['Category']})\n\n#### How the tree decided\n{path}")


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Dropdown(sorted(_df["Ship Mode"].unique()), value="Standard Class", label="Ship mode"),
        gr.Dropdown(sorted(_df["Segment"].unique()), value="Consumer", label="Customer segment"),
        gr.Dropdown(sorted(_df["Region"].unique()), value="West", label="Region"),
        gr.Dropdown(sorted(SUBCATS), value="Binders", label="Product sub-category"),
        gr.Number(value=150, label="Sales value ($)"),
        gr.Slider(1, 14, value=3, step=1, label="Quantity"),
        gr.Slider(0, 0.8, value=0.2, step=0.05, label="Discount (0.2 = 20%)"),
    ],
    outputs=gr.Markdown(),
    title="🌳 Loss-Making Order Detector (Decision Tree)",
    description="Predicts whether a Superstore order will lose money and shows every question "
                "the decision tree asked to reach its answer.",
    examples=[
        ["Standard Class", "Consumer", "Central", "Binders", 120, 3, 0.8],
        ["Second Class", "Corporate", "East", "Phones", 600, 2, 0.0],
        ["Standard Class", "Home Office", "West", "Tables", 900, 3, 0.3],
    ],
)

if __name__ == "__main__":
    demo.launch()
