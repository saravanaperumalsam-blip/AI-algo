"""Gradio app: heart disease risk from a Random Forest, with the trees' vote count.

Run:  python app.py   (trains the model first if needed)
"""
import gradio as gr

from train import load_model, predict_with_votes

bundle = load_model()
YES_NO = [("No", 0), ("Yes", 1)]


def run(male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
        diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    person = dict(male=male, age=age, currentSmoker=currentSmoker, cigsPerDay=cigsPerDay,
                  BPMeds=BPMeds, prevalentStroke=prevalentStroke, prevalentHyp=prevalentHyp,
                  diabetes=diabetes, totChol=totChol, sysBP=sysBP, diaBP=diaBP, BMI=BMI,
                  heartRate=heartRate, glucose=glucose)
    score, votes, n = predict_with_votes(bundle, person)
    label = "⚠️ **Higher risk**" if score >= 0.5 else "✅ **Lower risk**"
    bar = "🟥" * round(10 * votes / n) + "⬜" * (10 - round(10 * votes / n))
    return (f"### {label} of coronary heart disease within 10 years\n\n"
            f"Risk score: **{score * 100:.0f}/100**\n\n"
            f"**{votes} of {n} trees** vote 'higher risk' {bar}\n\n"
            "_Each tree saw a different random sample of patients and features; the forest "
            "averages them. Scores are relative risk scores, not calibrated probabilities._")


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Radio([("Female", 0), ("Male", 1)], value=1, label="Sex"),
        gr.Slider(30, 75, value=50, step=1, label="Age"),
        gr.Radio(YES_NO, value=0, label="Current smoker"),
        gr.Slider(0, 60, value=0, step=1, label="Cigarettes per day"),
        gr.Radio(YES_NO, value=0, label="On blood pressure medication"),
        gr.Radio(YES_NO, value=0, label="Previous stroke"),
        gr.Radio(YES_NO, value=0, label="Hypertension"),
        gr.Radio(YES_NO, value=0, label="Diabetes"),
        gr.Slider(110, 450, value=230, label="Total cholesterol (mg/dL)"),
        gr.Slider(85, 230, value=130, label="Systolic BP (mm Hg)"),
        gr.Slider(50, 140, value=82, label="Diastolic BP (mm Hg)"),
        gr.Slider(15, 50, value=25.5, label="BMI"),
        gr.Slider(45, 140, value=75, step=1, label="Heart rate (bpm)"),
        gr.Slider(40, 300, value=80, step=1, label="Glucose (mg/dL)"),
    ],
    outputs=gr.Markdown(),
    title="🌲 Heart Disease Risk (Random Forest)",
    description="A forest of decision trees votes on 10-year coronary heart disease risk. "
                "Educational project only, not medical advice.",
)

if __name__ == "__main__":
    demo.launch()
