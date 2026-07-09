## 🚀 Live Demo
**Streamlit App:** https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frffut.streamlit.app

## 🎯 GKS 2027 - Medical IT XAI Project
- **Model:** XGBoost + SHAP Explainability
- **Clinical Metrics:** Recall 67.3%, AUC-ROC 0.76
- **Focus:** Interpretable AI for Diabetes Screening
- **Tech Stack:** Python, Streamlit, SHAP, Scikit-learn
- # 🩺 Explainable AI-Based Diabetes Risk Prediction System

Independent Medical Information Technology AI Project | GKS-Undergraduate 2027 Portfolio

🚀 Live Demo: https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frffut.streamlit.app
![SHAP Demo](https://raw.githubusercontent.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project/main/demo.png)

📊 Performance:
- Accuracy: 69.5%
- AUC-ROC: 0.76
- Recall: 67.3%

👩‍💻 Developed by: Kashish

## 🎯 Problem Statement

Diabetes is a growing global healthcare challenge. Many individuals remain undiagnosed due to limited access to affordable screening.

Both India and South Korea face healthcare challenges related to early disease detection and aging populations. This project explores how explainable AI can support accessible and trustworthy diabetes risk screening.

## 🔬 Solution

I developed an XGBoost-based diabetes risk prediction system using the PIMA Diabetes dataset with SHAP explainability.

Unlike black-box models, this approach provides interpretable explanations for predictions, improving transparency and trust in healthcare AI research.

## 📈 Model Performance

| Metric | Score | Significance |
|---|---|---|
| Accuracy | 69.5% | Overall prediction performance |
| AUC-ROC | 0.76 | Ability to distinguish between risk groups |
| Recall | 67.3% | Helps reduce missed high-risk cases in screening scenarios |
| Important Features | Glucose, BMI, Age | Consistent with established diabetes risk factors |

## 🇰🇷 Connection to Medical IT Research

This project aligns with South Korea's focus on trustworthy medical AI and digital healthcare innovation.

Through further study, I aim to explore advanced approaches such as Federated Learning, enabling collaborative AI research between healthcare institutions while protecting patient privacy.

## 🔍 Model Validation and Interpretability

- Confusion Matrix: Evaluates classification performance
- Feature Importance: Examines influential model features
- SHAP Summary Plot: Provides interpretable explanations behind predictions
- Test Patient Examples: Demonstrates prediction probabilities

## 💻 Tech Stack

- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Pandas
- Matplotlib

## ⚠️ Medical Disclaimer

This project is an educational research prototype and is not intended for clinical diagnosis or medical decision-making. The model was trained on the publicly available PIMA Indians Diabetes Dataset. Always consult qualified healthcare professionals for medical decisions.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
