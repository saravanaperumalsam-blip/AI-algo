"""Gradio app: which customer segment does a customer belong to?

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr
import pandas as pd

from train import assign_segment, load_model

bundle = load_model()
profile = bundle["profile"]

ACTIONS = {
    "Champions": "Reward them: loyalty perks, early access, ask for reviews and referrals.",
    "Big spenders cooling off": "Win them back before they lapse: personalised offers on "
                                "categories they bought, a check-in email.",
    "Low-value / lapsing": "Low-cost reactivation only (newsletters, seasonal promos); don't "
                           "spend heavily on discounts here.",
}


def run(recency_days, frequency, monetary):
    segment, distances = assign_segment(bundle, recency_days, frequency, monetary)
    row = profile.set_index("segment").loc[segment]
    text = (f"### 🎯 Segment: **{segment}**\n\n"
            f"Typical customer in this segment: last order **{row.recency_days:.0f} days** ago, "
            f"**{row.frequency:.0f} orders**, **${row.monetary:,.0f}** total spend "
            f"({int(row.customers)} customers, {row.share_of_sales:.0%} of all sales).\n\n"
            f"**Suggested action:** {ACTIONS.get(segment, '')}")
    dist = pd.DataFrame({"Segment": list(distances), "Distance to centre": list(distances.values())})
    return text, dist.sort_values("Distance to centre")


table = profile[["segment", "customers", "recency_days", "frequency", "monetary",
                 "share_of_customers", "share_of_sales"]].copy()
table["share_of_customers"] = (table["share_of_customers"] * 100).round(1)
table["share_of_sales"] = (table["share_of_sales"] * 100).round(1)
table.columns = ["Segment", "Customers", "Median days since order", "Median orders",
                 "Median spend ($)", "% customers", "% sales"]

with gr.Blocks(title="Customer Segmentation") as demo:
    gr.Markdown("# 🛒 Customer Segmentation (K-Means)\n"
                "K-Means grouped 793 Superstore customers by **Recency**, **Frequency** and "
                "**Monetary** value without being told any groups in advance.")
    gr.Dataframe(table.round(1), label="The segments K-Means found", interactive=False)
    with gr.Row():
        r = gr.Slider(1, 1200, value=60, step=1, label="Days since last order (recency)")
        f = gr.Slider(1, 17, value=6, step=1, label="Number of orders (frequency)")
        m = gr.Slider(5, 25000, value=2500, label="Total spend $ (monetary)")
    btn = gr.Button("Find segment", variant="primary")
    out, dist = gr.Markdown(), gr.Dataframe(label="Distance to each segment centre (smaller = closer)")
    btn.click(run, [r, f, m], [out, dist], api_name="segment")

if __name__ == "__main__":
    demo.launch()
