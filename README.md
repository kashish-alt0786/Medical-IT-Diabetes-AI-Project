# 🩺 Explainable AI-Based Diabetes Risk Prediction System

> Independent Medical Information Technology AI Project | Interpretable Machine Learning for Healthcare Screening

[[Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[[Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[[XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.ai/)
[[SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple.svg)](https://shap.readthedocs.io/)
[[Live App](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B.svg)](https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/)

**🌐 Live Demo:** [medical-it-diabetes-ai-project.streamlit.app](https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/)  
**👩‍💻 Developed by:** Kashish | Data Analytics | Medical IT  
**🌍 Languages:** English | हिन्दी | 한국어

---

## 🎯 Medical IT XAI Project Overview

- **Model:** XGBoost + SHAP Explainability
- **Model performance:** Recall 67.3%, AUC-ROC 0.76, Accuracy 69.5%
- **Focus:** Interpretable AI for Diabetes Screening
- **Tech Stack:** Python, Streamlit, SHAP, XGBoost, Scikit-learn
- **Domain:** Medical Information Technology | Digital Healthcare | Trustworthy AI

[SHAP Explainability Demo](https://raw.githubusercontent.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project/main/IMG_20260710_080138.jpg)

*SHAP force plot: Red features increase the predicted diabetes risk, while green features decrease it for the individual prediction shown.*

---

## 🎯 Problem Statement

Diabetes is a growing global healthcare challenge. Many individuals remain undiagnosed due to limited access to affordable screening.

Both India and South Korea face healthcare challenges related to early disease detection and aging populations. This project explores how explainable AI can support accessible and trustworthy diabetes risk screening in diverse healthcare settings.

---

## 🔬 Solution: Interpretable AI for Healthcare

I developed an XGBoost-based diabetes risk prediction system using the PIMA Indians Diabetes dataset with integrated SHAP explainability.

**Why Explainability Matters in Medical IT:**
Unlike black-box models, this approach provides interpretable explanations for every prediction, improving transparency and trust in healthcare AI research. SHAP force plots show exactly which clinical factors increased or decreased risk, enabling validation by healthcare researchers.

**Clinical Significance:**
- **Recall 67.3%** - Indicates the proportion of diabetic cases correctly identified by the model
- **AUC-ROC 0.76** - Good discrimination between diabetic and non-diabetic cases
- **Feature Importance** - Glucose, BMI, Age align with established diabetes risk factors

---

## 📈 Model Performance & Validation

| Metric | Score | Screening relevance |
| --- | --- | --- |
| **Accuracy** | 69.5% | Overall prediction performance on test set |
| **AUC-ROC** | 0.76 | Strong discrimination between diabetic/non-diabetic |
| **Recall** | 67.3% | Minimizes false negatives - critical for screening |
| **Important Features** | Glucose, BMI, Age | Consistent with medical literature |

### 🔍 Model Validation and Interpretability
- **Confusion Matrix:** Evaluates classification performance and error types
- **Feature Importance:** Examines influential model features using XGBoost gain
- **SHAP Summary Plot:** Provides global and local interpretable explanations
- **Test Patient Examples:** Demonstrates prediction probabilities with explanations

---

## 🌏 Connection to Medical IT Research

This project aligns with global research priorities in trustworthy medical AI and digital healthcare innovation.

**Research Directions:**
- **Explainable AI (XAI):** SHAP integration demonstrates commitment to transparent ML in healthcare
- **Digital Health Accessibility:** Multilingual interface (English/Hindi/Korean) supports diverse populations
- **Preventive Screening:** Focus on early detection aligns with public health priorities in aging societies
- **Future Work:** Through further study, I aim to explore advanced approaches such as Federated Learning, enabling collaborative AI research between healthcare institutions while protecting patient privacy

---

## 💻 Tech Stack

- **Python 3.9+** - Core programming language
- **Scikit-learn** - Data preprocessing and model evaluation
- **XGBoost** - Gradient boosting classifier with class balancing
- **SHAP** - TreeExplainer for model interpretability
- **Streamlit** - Interactive web application framework
- **Pandas** - Data manipulation and analysis
- **Matplotlib / Seaborn** - Visualization and SHAP plots

---

## 📂 Project Structure
Medical-IT-Diabetes-AI-Project/
├── LICENSE                    # MIT License
├── README.md                  # Project documentation  
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── app.py                    # Main Streamlit application
└── data/                     # Datasets
    └── diabetes.csv         # PIMA Indians  Database
    
---

## 📚 Relevant Learning

This project was developed after completing courses in:

- **Kaggle** – Intro to Machine Learning, Pandas, Data Visualization
- **OpenWHO** – Digital Health, Ethics & Governance of AI for Health

These courses strengthened my understanding of machine learning, healthcare data analysis, data visualization, and responsible AI, which I applied while developing this explainable medical screening tool.

---

## 🚀 Quick Start

**Prerequisites:** Python 3.9+

**1. Clone repository**
```bash
git clone https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project.git
cd Medical-IT-Diabetes-AI-Project
``` 


**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Run the app**
```bash
streamlit run app.py
```

**4. Open browser:** http://localhost:8501
**Or use Live Demo:** [Click here](https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/)
## 🔮 Future Research Improvements
- Federated Learning: Explore privacy-preserving collaborative training across institutions
- Multi-ethnic Datasets: Validate model on Korean National Health data for broader applicability
- FHIR Integration: Connect with educational Electronic Health Record standards
- Mobile Deployment: Progressive Web App for community health workers
- Clinical Validation: Collaborate with medical students for real-world usability testing

## ⚠️ Medical Disclaimer
This project is an educational research prototype and is not intended for clinical diagnosis or medical decision-making.

The model was trained on the publicly available PIMA Indians Diabetes Dataset and may contain demographic bias. It may not generalize to all populations, age groups, or ethnicities.

Always consult qualified healthcare professionals for medical decisions. The developer assumes no liability for any health outcomes resulting from the use of this software.

## 📧 Contact
Kashish  
GitHub: https://github.com/kashish-alt0786
Live App: https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/

## Acknowledgments
 • Dataset: National Institute of Diabetes and Digestive and Kidney Diseases
 • Courses: Kaggle Learn, OpenWHO Digital Health Program
 • Frameworks: XGBoost, SHAP, Streamlit open-source communities
 
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Open-source software for healthcare education and responsible AI research.

## Developed for Medical Information Technology Education and Explainable AI Research | July 2026  
🔍 Explainable | ⚕️ Trustworthy AI
