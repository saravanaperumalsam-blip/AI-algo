"""Gradio app: is this SMS spam? Shows the words that tipped the decision.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr

from train import explain, load_model

bundle = load_model()


def run(message: str):
    if not message.strip():
        return "Type a message first.", None
    p_spam, words = explain(bundle, message)
    verdict = "🚫 **SPAM**" if p_spam >= 0.5 else "✅ **Not spam (ham)**"
    note = "" if len(words) else "\n\n_None of these words appear in the training vocabulary._"
    return (f"### {verdict}\n\nSpam probability: **{p_spam:.1%}**\n\n"
            "Positive evidence pushes towards spam, negative towards a normal message." + note,
            words)


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="SMS message"),
    outputs=[gr.Markdown(), gr.Dataframe(label="Most influential words in this message")],
    title="📱 SMS Spam Filter (Naive Bayes)",
    description="Multinomial Naive Bayes trained on 5,169 real SMS messages.",
    examples=[
        ["WINNER!! You have won a £1000 prize. Call 09061701461 now to claim, T&C apply 18+"],
        ["Hey, are we still meeting for lunch tomorrow? I'll be there at 12:30"],
        ["URGENT: your mobile number has been selected for a free Nokia tone. Reply YES"],
        ["Sorry I missed your call, can you ring me back later?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
