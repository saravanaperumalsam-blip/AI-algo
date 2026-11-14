# 🤖 AI Algorithms

A collection of machine learning algorithms implemented in Python on real datasets, each with a small interactive app.

| Algorithm | Project | Folder |
|---|---|---|
| Logistic Regression | Heart disease (10-year CHD) risk prediction | [`Logistic regression/`](Logistic%20regression/) |

---

## ❤️ Logistic Regression: Heart Disease Risk Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Predicts whether a person is at risk of developing **coronary heart disease (CHD) within 10 years**, using the **Framingham Heart Study** dataset. A **Gradio** web app lets you enter a patient's details and returns a risk estimate with a probability.

### 📂 Dataset

`framingham.csv`: 4,238 participants and 15 features. **15.2%** of them developed CHD within 10 years (`TenYearCHD = 1`), so the classes are imbalanced.

| Group | Features |
|---|---|
| Demographics | `male`, `age`, `education` (dropped) |
| Behaviour | `currentSmoker`, `cigsPerDay` |
| Medical history | `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` |
| Measurements | `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` |
| **Target** | `TenYearCHD` |

### 🧹 Data processing (`data-processing.ipynb`)

- Checked missing values; the largest gaps are `glucose` (388), `education` (105) and `BPMeds` (53)
- Dropped `education`, which isn't a medical predictor
- Filled missing numeric values with the **median** (more robust to extreme medical readings) and `BPMeds` with the **mode**
- Removed medically impossible values (e.g. cholesterol outside 100–600, systolic BP outside 70–250, BMI outside 10–60)

### ⚙️ Model (`app.py`)

1. 80/20 train/test split (`random_state=42`)
2. Missing values filled with **training-set medians** (to avoid leaking test data)
3. Features standardised with `StandardScaler`
4. `LogisticRegression(class_weight="balanced")` trained on the 14 features, so the minority CHD class isn't ignored
5. Gradio app that returns a risk label and a 0–100 risk score

### 📈 Results (test set, 848 people)

**The accuracy trap.** Only 15% of people develop CHD, so a model that always predicts "no CHD" is already 85.4% accurate. The first version of this model reached 85.7% accuracy, but it caught only **9 of 124** real CHD cases. For a screening tool, that's the wrong trade-off: missing an at-risk person costs far more than an extra check-up.

**The fix: balanced class weights.** Weighting CHD cases more heavily during training makes the model take them seriously:

| Model | Recall (CHD caught) | Precision | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Unweighted (first version) | 7.3% (9/124) | 60.0% | 0.129 | 85.7% | 0.709 |
| **Balanced class weights (final)** | **65.3% (81/124)** | 23.8% | **0.348** | 64.3% | 0.710 |
| Balanced + tuned threshold (0.55, chosen by 5-fold CV on training data) | 55.6% (69/124) | 26.7% | 0.361 | 71.2% | 0.710 |

The final model catches **about 9× more real CHD cases** (81 vs 9). The cost is more false alarms, which is acceptable for screening because flagged people would get a follow-up test. Tuning the threshold gave a very similar F1 at lower recall, so the simpler default threshold was kept.

Final model confusion matrix:

| | Predicted no CHD | Predicted CHD |
|---|---|---|
| **Actual no CHD** | 464 | 260 |
| **Actual CHD** | 43 | 81 |

ROC-AUC stays at 0.71: re-weighting changes *where* the model draws the line, not how well it ranks people by risk. Better features or models are needed to raise that.

**Strongest risk factors** (standardised coefficients):

| Feature | Coefficient |
|---|---|
| Age | +0.57 |
| Systolic blood pressure | +0.30 |
| Cigarettes per day | +0.29 |
| Male | +0.21 |
| Glucose | +0.19 |

This matches medical knowledge: older age, high blood pressure, smoking, being male and high glucose all raise CHD risk.

### 🖥️ Gradio app

Inputs: sex, age, smoking status, cigarettes per day, BP medication, prior stroke, hypertension, diabetes, total cholesterol, systolic/diastolic BP, BMI, heart rate and glucose.

Output: **⚠️ Higher risk** or **✅ Lower risk**, with a 0–100 risk score. Balanced class weights make the score useful for ranking risk, but it isn't a calibrated probability, so it's shown as a score rather than a percentage chance.

When `app.py` starts, it prints recall, precision, F1, ROC-AUC, accuracy and the confusion matrix.

### 🚀 How to run

```bash
git clone https://github.com/saravanaperumalsam-blip/AI-algo.git
cd "AI-algo/Logistic regression"
pip install -r requirements.txt
python app.py
```

Then open the local URL shown in the terminal (usually http://127.0.0.1:7860).

### 🔧 Next improvements

- Calibrate the risk score (e.g. `CalibratedClassifierCV`) so it can be read as a real probability
- Try SMOTE and compare it with class weighting
- Use the cleaning steps from the notebook (outlier removal) inside `app.py` as well
- Compare with Random Forest and gradient boosting

### ⚠️ Disclaimer

This is an educational machine learning project. It is **not a medical device** and must not be used for diagnosis or treatment decisions.

---

## 🗂 Project structure

```
AI-algo/
├── README.md
└── Logistic regression/
    ├── app.py                   # Model training + Gradio app
    ├── data-processing.ipynb    # Data exploration and cleaning
    ├── framingham.csv           # Framingham Heart Study dataset
    └── requirements.txt
```

## 👤 Author

**Saravana Perumal**: [GitHub](https://github.com/saravanaperumalsam-blip)
