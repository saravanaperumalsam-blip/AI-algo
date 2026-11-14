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
4. `LogisticRegression` trained on the 14 features
5. Gradio app for predictions with a probability score

### 📈 Results (test set, 848 people)

| Metric | Score |
|---|---|
| Accuracy | 85.7% |
| ROC-AUC | 0.709 |
| Recall for CHD cases | 7.3% (9 of 124) |
| Precision for CHD cases | 60.0% |

Confusion matrix:

| | Predicted no CHD | Predicted CHD |
|---|---|---|
| **Actual no CHD** | 718 | 6 |
| **Actual CHD** | 115 | 9 |

**What this shows:** because 85% of people don't develop CHD, a model that always predicts "no CHD" would already be 85.4% accurate. This model's ROC-AUC of 0.71 shows it does learn real risk patterns. At the default 0.5 threshold, though, it flags very few people as high risk and misses most real cases. For a screening tool, catching at-risk people (recall) matters more than overall accuracy. That's the main thing to improve next (see below).

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

Output: **⚠️ High risk** or **✅ Low risk**, with a probability.

### 🚀 How to run

```bash
git clone https://github.com/saravanaperumalsam-blip/AI-algo.git
cd "AI-algo/Logistic regression"
pip install -r requirements.txt
python app.py
```

Then open the local URL shown in the terminal (usually http://127.0.0.1:7860).

### 🔧 Next improvements

- Handle class imbalance with `class_weight="balanced"` or SMOTE
- Lower the decision threshold to catch more at-risk patients, and report recall and F1, not just accuracy
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
