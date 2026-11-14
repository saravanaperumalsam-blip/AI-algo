"""Gradio app: predict an order's profit and see how discount changes it.

Run:  python app.py   (trains the model first if needed; tuning takes a few minutes)
"""
import gradio as gr
import pandas as pd

from train import DATA, discount_curve, load_model, predict_profit

bundle = load_model()
_df = pd.read_csv(DATA, encoding="latin1")
SUBCATS = _df.groupby("Sub-Category")["Category"].first().to_dict()


def run(ship_mode, segment, region, sub_category, sales, quantity, discount):
    order = {"Ship Mode": ship_mode, "Segment": segment, "Region": region,
             "Category": SUBCATS[sub_category], "Sub-Category": sub_category,
             "Sales": float(sales), "Quantity": int(quantity), "Discount": float(discount)}
    profit = predict_profit(bundle, order)
    curve = discount_curve(bundle, order)
    breakeven = curve.loc[curve["Predicted profit"] < 0, "Discount"]
    tip = (f"For this order, predicted profit turns **negative from a {breakeven.iloc[0]:.0%} discount**."
           if len(breakeven) else "This order stays profitable at every discount level.")
    icon = "🟢" if profit >= 0 else "🔴"
    money = f"{'-' if profit < 0 else ''}${abs(profit):,.2f}"
    text = (f"### {icon} Predicted profit: **{money}**\n\n"
            f"That's a {profit / order['Sales']:.1%} margin on ${order['Sales']:,.2f} of sales.\n\n{tip}")
    return text, curve


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Dropdown(sorted(_df["Ship Mode"].unique()), value="Standard Class", label="Ship mode"),
        gr.Dropdown(sorted(_df["Segment"].unique()), value="Consumer", label="Customer segment"),
        gr.Dropdown(sorted(_df["Region"].unique()), value="West", label="Region"),
        gr.Dropdown(sorted(SUBCATS), value="Chairs", label="Product sub-category"),
        gr.Number(value=500, label="Sales value ($)"),
        gr.Slider(1, 14, value=3, step=1, label="Quantity"),
        gr.Slider(0, 0.8, value=0.2, step=0.05, label="Discount (0.2 = 20%)"),
    ],
    outputs=[gr.Markdown(),
             gr.LinePlot(x="Discount", y="Predicted profit", title="Same order at different discounts",
                         height=300)],
    title="🚀 Order Profit Predictor (Gradient Boosting)",
    description="Gradient-boosted trees predict the profit of a Superstore order in dollars, and "
                "show how the profit changes if the discount changes.",
    examples=[["Standard Class", "Consumer", "West", "Chairs", 500, 3, 0.2],
              ["First Class", "Corporate", "East", "Copiers", 2000, 3, 0.0],
              ["Standard Class", "Consumer", "Central", "Tables", 900, 3, 0.4]],
)

if __name__ == "__main__":
    demo.launch()
