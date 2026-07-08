# 🩺 Medical IT - Diabetes Prediction AI

### 인제대학교 의생명공학부 지원을 위한 AI 당뇨병 예측 프로젝트
### Inje University | GKS Undergraduate Application Project
**Applicant: Kashish | Major: Medical IT**

> AI-powered diagnostic support system to predict diabetes risk using Machine Learning - Developed for Medical IT program at Inje University, South Korea.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%2C%20Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Inje](https://img.shields.io/badge/For-Inje%20University-red)

---

## 🎯 Project Objective
To build an intelligent, data-driven system that assists early diagnosis of diabetes - aligned with Medical IT convergence of Medicine + AI + Data Science. This project demonstrates practical application of AI in healthcare, a core focus of Inje University's Medical IT program.

## 📊 Dataset
- **Source:** Pima Indians Diabetes Database (768 patients) - Kaggle
- **Features:** 8 medical parameters (Glucose, BMI, Age, Blood Pressure, Insulin, etc.)
- **Target:** Diabetes Outcome (0 = No, 1 = Yes)
- **Link:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## 🧠 Methodology - 3 Model Comparison

| Model | Accuracy | Why Used? |
| :--- | :--- | :--- |
| **Logistic Regression** | ~76% | Baseline - interpretable for medical use |
| **Random Forest** | ~77% | Handles non-linear medical data |
| **XGBoost** | **~78-80%** | **Best performance - Ensemble learning** |

> **Final Model Selected: XGBoost** for highest accuracy and robustness.

## 📈 Key Visualizations

### 1. Model Accuracy Comparison
*Compares 3 algorithms to select best clinical model - Shows XGBoost outperforms others*

### 2. Feature Importance Analysis
**Finding:** Glucose level is the strongest predictor of diabetes, followed by BMI and Age.
*This aligns with medical literature - validates model clinically.*

### 3. Age vs Glucose Correlation
*Shows correlation between age and glucose - critical for preventive care in aging population*

## 💡 Clinical Insights
- Glucose > 140 mg/dL shows high risk
- BMI and Age are secondary risk factors
- Model can be integrated into hospital EHR for early screening at Paik Hospital
- Early detection can reduce complications by 60%

## 🚀 Future Work for Inje University
- [ ] Integrate with wearable data (CGM - Continuous Glucose Monitor)
- [ ] Develop web app for doctors using Flask/FastAPI
- [ ] Add SHAP explainability for medical transparency (Doctors need to know WHY)
- [ ] Collaborate with Inje University Paik Hospital data for Korean population
- [ ] Research paper on AI in diabetes prediction

## 🛠️ Tech Stack
`Python` `Pandas` `Scikit-Learn` `XGBoost` `Matplotlib` `Seaborn` `Google Colab` `GitHub`

## 🔗 Run the Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kashish-alt0786/Medical-IT-Diabetes-AI-Project/blob/main/Diabetes_AI_Project.ipynb.ipynb)

1. Click Colab badge above
2. Runtime → Run all
3. See results instantly!

## 👩‍💻 About Me
Aspiring Medical IT student passionate about AI in healthcare. This project demonstrates my ability to apply AI to real medical problems - a core skill for Inje University's Medical IT program. I built this project to show my dedication and technical readiness for undergraduate studies in South Korea.

**For GKS 2025-2026 | Inje University | Medical IT Department**

---
⭐ If you like this project for GKS, please star the repo!
