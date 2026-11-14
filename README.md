# 🤖 AI Algorithms: Machine Learning, One Project per Algorithm

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00)

This repo is a hands-on tour of the **10 core machine learning algorithms**. Each algorithm has its own folder with a real-world project: a real dataset, model tuning, evaluation charts, an interactive **Gradio** app, tests, and a README that explains the algorithm in plain English.

## ✅ Algorithm checklist

| | Algorithm | Type | Project | Folder |
|---|---|---|---|---|
| ✅ | **Logistic Regression** | Supervised: classification | Heart disease risk prediction | [`Logistic regression/`](Logistic%20regression/) |
| ✅ | **Linear Regression** | Supervised: regression | Diabetes progression predictor | [`Linear regression/`](Linear%20regression/) |
| ✅ | **Decision Tree** | Supervised: classification | Loss-making order detector (Superstore) | [`Decision tree/`](Decision%20tree/) |
| ✅ | **Random Forest** | Supervised: ensemble | Heart disease: forest vs. logistic regression | [`Random forest/`](Random%20forest/) |
| ✅ | **K-Nearest Neighbors (KNN)** | Supervised: classification | Wine cultivar classifier | [`K-nearest neighbors/`](K-nearest%20neighbors/) |
| ✅ | **Naive Bayes** | Supervised: text classification | SMS spam filter | [`Naive Bayes/`](Naive%20Bayes/) |
| ✅ | **Support Vector Machine (SVM)** | Supervised: classification | Breast cancer diagnosis | [`Support vector machine/`](Support%20vector%20machine/) |
| ⬜ | **K-Means Clustering** | Unsupervised: clustering | Customer segmentation (RFM) | `K-means clustering/` |
| ⬜ | **Principal Component Analysis (PCA)** | Unsupervised: dimensionality reduction | Handwritten digit compression | `PCA/` |
| ⬜ | **Gradient Boosting** | Supervised: ensemble | Order profit prediction (Superstore) | `Gradient boosting/` |

## 🧠 The algorithms in one line each

| Algorithm | The idea in plain English |
|---|---|
| Linear Regression | Draw the best straight line (or flat surface) through the data to predict a **number** |
| Logistic Regression | Like linear regression, but squashes the output into a 0–1 **probability** for yes/no questions |
| Decision Tree | Ask a series of yes/no questions ("Is discount > 20%?") until you reach an answer, like a flowchart |
| Random Forest | Train hundreds of slightly different decision trees and let them **vote** |
| K-Nearest Neighbors | Look at the *k* most similar past examples and copy their answer |
| Naive Bayes | Use word probabilities ("'FREE' appears in 30% of spam") and combine them with Bayes' theorem |
| Support Vector Machine | Find the boundary that separates two classes with the **widest possible gap** |
| K-Means Clustering | Group similar items together without being told the groups in advance |
| PCA | Squash many features into a few new ones that keep most of the information |
| Gradient Boosting | Build small trees one after another, each one fixing the previous trees' mistakes |

## 🗂 What every project folder contains

```
<Algorithm>/
├── README.md          # How the algorithm works, dataset, approach, results
├── train.py           # Load data → tune → evaluate → save model, charts and metrics
├── app.py             # Interactive Gradio app
├── test_project.py    # Automated tests (python -m unittest)
├── requirements.txt
├── images/            # Charts produced by train.py
└── results/           # metrics.json produced by train.py
```

## 🚀 Running any project

```bash
git clone https://github.com/saravanaperumalsam-blip/AI-algo.git
cd "AI-algo/<Algorithm folder>"
pip install -r requirements.txt
python train.py            # trains, evaluates, saves charts and metrics
python app.py              # opens the interactive app at http://127.0.0.1:7860
python -m unittest -v      # runs the tests
```

## 👤 Author

**Saravana Perumal**: [GitHub](https://github.com/saravanaperumalsam-blip)
