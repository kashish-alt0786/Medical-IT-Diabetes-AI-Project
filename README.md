# 🩺 Diabetes Risk Prediction AI — GKS-UG 2027
### GKS-Undergraduate Application for Inje University, Medical Information Technology

**🚀 Live Demo:** `Add your Streamlit link here after Fix #4` 
**📊 Status:** 69.5% Accuracy | 0.76 AUC | 67.3% Recall 
**👩‍💻 Student:** Kashish [Your Last Name], Ludhiana, Punjab, India 

---

### 🎯 Problem Statement
**India:** 77 million diabetics, many undiagnosed due to expensive screening. 
**Korea:** Fastest aging society globally, needs early-detection AI for elderly care. 
**Gap:** Both countries need low-cost, trustworthy AI that doctors can understand.

### 🔬 My Solution
I built an XGBoost model with SHAP Explainability using the PIMA Diabetes dataset. 
Unlike black-box AI, this model shows WHY it predicts diabetes risk — critical for hospital adoption.

### 📈 Key Results
| Metric | Score | Medical IT Significance |
| --- | --- | --- |
| **Accuracy** | 69.5% | Baseline performance |
| **AUC-ROC** | 0.76 | Strong discrimination ability |
| **Recall** | 67.3% | Misses fewer diabetic patients — critical for screening |
| **Top Features** | Glucose, BMI, Age | Matches clinical diagnostic guidelines |

### 🇰🇷 Why Inje University?
Inje University’s Medical IT program leads in AI for aging societies. My SHAP explainability 
approach aligns with Korea’s focus on trustworthy medical AI. At GKS, I will study 
Federated Learning so Indian + Korean hospitals can train AI together without sharing 
private patient data — building an India-Korea health bridge.

### 🔍 Model Validation — Proof This Isn't Tutorial Code
1. **Confusion Matrix:** Validates 67.3% Recall for patient safety
2. **Feature Importance:** Confirms model uses Glucose/BMI, not random noise  
3. **SHAP Summary Plot:** Shows doctor-level reasoning for each prediction
4. **5 Test Patients:** Sample predictions with probability scores included in notebook

### 💻 Tech Stack
`Python` `Scikit-learn` `XGBoost` `SHAP` `Streamlit` `Pandas` `Matplotlib`

### ⚠️ Medical Disclaimer
**Educational project only.** Not approved for clinical diagnosis. Trained on PIMA Indians 
public dataset. Always consult licensed medical professionals for health decisions.

### 🚀 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
