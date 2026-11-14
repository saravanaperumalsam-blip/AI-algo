import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import gradio as gr

df=pd.read_csv('framingham.csv')

if 'education' in df.columns:
    df = df.drop(columns=['education'])

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Only ~15% of people develop CHD, so an unweighted model learns to predict
# "no CHD" for almost everyone (high accuracy, but it misses most real cases).
# class_weight="balanced" gives CHD cases more weight so the model catches them.
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate: for a screening tool, recall (CHD cases caught) matters most
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"Recall (CHD cases caught): {recall_score(y_test, y_pred) * 100:.1f}%")
print(f"Precision:                 {precision_score(y_test, y_pred) * 100:.1f}%")
print(f"F1 score:                  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:                   {roc_auc_score(y_test, y_prob):.3f}")
print(f"Accuracy:                  {accuracy_score(y_test, y_pred) * 100:.1f}%")
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, y_pred))

# NEW GRADIO ADDITIONS BELOW
# ==========================================

# 1. Prediction Function
def predict_heart_disease(male, age, currentSmoker, cigsPerDay, BPMeds, 
                          prevalentStroke, prevalentHyp, diabetes, 
                          totChol, sysBP, diaBP, BMI, heartRate, glucose):
    
    # Put input values into a DataFrame matching training features
    input_df = pd.DataFrame([[
        int(male), age, int(currentSmoker), cigsPerDay, int(BPMeds),
        int(prevalentStroke), int(prevalentHyp), int(diabetes),
        totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]], columns=X.columns)
    
    # Scale inputs using your fitted scaler
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    # With balanced class weights the score is a relative risk score,
    # not a calibrated probability, so it is labelled as a score.
    prediction = model.predict(input_scaled)[0]
    score = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        return f"⚠️ Higher risk of heart disease (risk score {score:.0f}/100)"
    else:
        return f"✅ Lower risk of heart disease (risk score {score:.0f}/100)"


# 2. Input UI Controls
inputs = [
    gr.Radio([("Female", 0), ("Male", 1)], label="Sex", value=1),
    gr.Number(label="Age", value=50),
    gr.Radio([("No", 0), ("Yes", 1)], label="Current Smoker", value=0),
    gr.Number(label="Cigarettes Per Day", value=0),
    gr.Radio([("No", 0), ("Yes", 1)], label="On BP Medication", value=0),
    gr.Radio([("No", 0), ("Yes", 1)], label="Prevalent Stroke", value=0),
    gr.Radio([("No", 0), ("Yes", 1)], label="Prevalent Hypertension", value=0),
    gr.Radio([("No", 0), ("Yes", 1)], label="Diabetes", value=0),
    gr.Number(label="Total Cholesterol", value=220),
    gr.Number(label="Systolic BP", value=120),
    gr.Number(label="Diastolic BP", value=80),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="Heart Rate", value=75),
    gr.Number(label="Glucose", value=85),
]

# 3. Create & Launch Interface
app = gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=gr.Textbox(label="Heart Disease Prediction"),
    title="Framingham Heart Disease Predictor",
    description="Estimates 10-year coronary heart disease risk. Educational project only, not medical advice."
)

if __name__ == "__main__":
    app.launch()