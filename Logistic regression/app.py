import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 8. Check how accurate the model is
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

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
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100
    
    if prediction == 1:
        return f"⚠️ High Risk of Heart Disease ({prob:.1f}% probability)"
    else:
        return f"✅ Low Risk of Heart Disease ({100 - prob:.1f}% probability)"


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
    title="Framingham Heart Disease Predictor"
)

app.launch()