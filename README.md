# 🩺 Medical IT - Diabetes Prediction AI
### Inje University | GKS Application Project

**One-night challenge: Build an AI that helps doctors predict diabetes**

---

### 🎯 Motivation
Diabetes affects 537M people worldwide. Early prediction can save lives. 
As a future Medical IT student, I wanted to prove I can build technology that helps medicine.

### 📊 Dataset
- **Source:** Pima Indians Diabetes Dataset (Kaggle - 768 patients)
- **Why this dataset?** Real clinical data: Glucose, Blood Pressure, BMI, Age - exactly what doctors check
- **Link:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### 🔧 What I Did (Data Preprocessing)
1. Checked for missing values (0 values in Glucose/BP = missing data)
2. Replaced missing values with median
3. Split data: 80% training, 20% testing
4. Standardized features for better ML performance

### 🤖 Algorithms Tested
1. **Logistic Regression** - Baseline, easy to interpret for doctors
2. **Random Forest** - Handles non-linear relationships, more accurate
- **Why?** Started simple, then used ensemble method for better performance

### 📈 Performance
- **Accuracy:** 74.68%
- **Precision:** 68% (When I say diabetes, I'm 68% correct)
- **Recall:** 55% (I catch 55% of actual diabetes cases)
- **F1-Score:** 61%

*For a first model built in one night, this is strong baseline!*

### ⚠️ Limitations
- Small dataset (only 768 patients, all female Pima Indians - not diverse)
- No real-time data
- 0 values needed manual cleaning
- Accuracy can improve with more data

### 🚀 Future Improvements
- Add more diverse datasets
- Try XGBoost / Neural Networks
- Build web app where patients can input values
- Connect with hospital EHR system
- Add explainable AI (SHAP) so doctors understand WHY prediction

### 💻 Demo
Code runs in Google Colab - just open .ipynb file!
Screenshot: [Add screenshot of your accuracy graph here]

### 👩‍💻 About Me
Kashish - Aspiring Medical IT student, passionate about bridging medicine + AI.
Built this in one night to prove dedication for Inje University.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Google Colab
