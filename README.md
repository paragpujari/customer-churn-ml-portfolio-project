# 📉 Customer Churn Prediction — End-to-End Machine Learning Portfolio Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **Complete end-to-end ML project** covering data cleaning → EDA → preprocessing → model training & comparison → evaluation → model persistence → interactive web deployment (Streamlit) + containerization (Docker).

Perfect for your **GitHub portfolio**, LinkedIn, or interview discussions.

---

## 🎯 Project Overview

**Business Problem**  
Telecom companies lose significant revenue when customers churn. Early identification of at-risk customers allows targeted retention campaigns (discounts, better plans, proactive support).

**Solution**  
A production-ready Random Forest classifier that predicts the probability a customer will churn, packaged as an interactive Streamlit web application.

**Dataset**  
[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (IBM sample dataset)  
- 7,043 customers  
- 20 features (demographics, services, account info, charges)  
- Target: `Churn` (Yes/No) — moderately imbalanced (~27% positive class)

---

## 🏗️ Project Structure

```
customer-churn-ml-project/
├── app/
│   └── streamlit_app.py          # Interactive web application
├── data/
│   └── Telco-Customer-Churn.csv  # Raw dataset
├── models/
│   ├── churn_pipeline.joblib     # Full trained sklearn Pipeline
│   ├── model_meta.joblib         # Metrics & feature lists
│   └── feature_importances.csv   # Feature importance ranking
├── src/
│   └── train_model.py            # Complete training + evaluation script
├── notebooks/                    # (Optional) Place your EDA notebooks here
├── Dockerfile                    # Container definition
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone / Download
```bash
git clone <your-repo-url>
cd customer-churn-ml-project
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model
```bash
python src/train_model.py
```
This will:
- Clean the data
- Split train/test (stratified)
- Compare Logistic Regression, Random Forest, Gradient Boosting
- Select best model by F1-score
- Save the full Pipeline (preprocessor + model)

### 4. Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501 in your browser.

---

## 🐳 Run with Docker

```bash
# Build
docker build -t customer-churn-app .

# Run
docker run -p 8501:8501 customer-churn-app
```
App available at http://localhost:8501

---

## ☁️ Deployment Options (Free / Easy)

### Option A: Streamlit Community Cloud (Recommended for portfolio)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Main file path: `app/streamlit_app.py`
5. Deploy → Get a public URL instantly

### Option B: Hugging Face Spaces
1. Create a new Space → Streamlit
2. Upload the project files (or connect GitHub)
3. It auto-detects and runs

### Option C: Render / Railway / Fly.io
- Use the provided `Dockerfile`
- Set start command if needed: `streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

---

## 📊 Model Performance (Test Set)

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 0.726    | 0.490     | **0.797** | 0.607   | 0.835   |
| **Random Forest** (Best) | 0.747  | 0.516     | 0.767  | **0.617** | 0.830   |
| Gradient Boosting      | **0.788**| **0.618** | 0.527  | 0.569   | **0.838**|

> We prioritized **F1-score** and **Recall** because missing a potential churner (false negative) is more costly than a false alarm.

**Top Predictive Features** (Random Forest):
1. `tenure`
2. `Contract_Month-to-month`
3. `TotalCharges`
4. `MonthlyCharges`
5. `Contract_Two year`
6. `OnlineSecurity_No`
7. `TechSupport_No`
8. `InternetService_Fiber optic`

---

## 🛠️ Technical Highlights

- **Full sklearn Pipeline**: Preprocessing (scaling + one-hot encoding) + model in a single serializable object → no data leakage, easy deployment.
- **Class imbalance handling**: `class_weight='balanced'` on tree models & Logistic Regression.
- **Stratified split** to preserve class ratio.
- **Feature importance** extraction and saving.
- **Production-ready app**: Input validation via Streamlit widgets, probability output, business recommendations, clean UI.
- **Dockerized** for reproducible deployment.
- **Modular code** ready for extension (add more models, SHAP explanations, monitoring, etc.).

---

## 📈 How to Extend This Project (Great for Interviews)

1. Add **SHAP / LIME** explanations in the Streamlit app
2. Implement **model monitoring** (data drift, prediction drift) with Evidently or NannyML
3. Create a **FastAPI** backend + separate frontend
4. Add **CI/CD** with GitHub Actions (lint + test + build Docker)
5. Experiment with **XGBoost / LightGBM / CatBoost** + Optuna hyperparameter tuning
6. Build a **batch prediction** endpoint / CLI
7. Store predictions in a database and build a simple dashboard

---

## 📦 Files Included for GitHub

- ✅ Trained model (`.joblib`) so the app works out-of-the-box
- ✅ Full training script with evaluation metrics printed
- ✅ Production Streamlit application
- ✅ Dockerfile
- ✅ Clean project layout
- ✅ Comprehensive README

Just push to GitHub and it is ready to showcase!

---

## 📜 License

MIT License — feel free to use, modify, and share for your portfolio.

---

**Made for aspiring Data Scientists & ML Engineers**  
Star ⭐ the repo if it helped you!
