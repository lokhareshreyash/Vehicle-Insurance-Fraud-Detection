# 📉 Vehicle Insurance Fraud Detection (Machine Learning Project)

This project focuses on identifying **fraudulent vehicle insurance claims** using machine learning.  
The goal is to help insurance companies **detect fraud early**, reduce financial losses, and improve claim investigation efficiency.

The project is built as a **complete ML pipeline**, starting from raw data to model evaluation and saving the final model.

---

## 🔍 What This Project Does
Insurance fraud data is highly **imbalanced**, meaning genuine claims are much more than fraud cases.  
In this project, I:
- Cleaned and prepared the data
- Handled class imbalance properly
- Built and evaluated multiple ML models
- Selected the best model based on meaningful metrics (not just accuracy)

---

## 📁 Data Used
- Vehicle insurance claims dataset  
- Target variable: **FraudFound_P** (1 = Fraud, 0 = Genuine)
- Data contains both:
  - **Numeric features** (claim amount, age, etc.)
  - **Categorical features** (policy type, vehicle category, accident area, etc.)

---

## ⚙️ Approach & Workflow

### 1️⃣ Data Preparation
- Loaded the dataset from a public source
- Separated features and target variable
- Identified numeric and categorical columns
- Handled missing values and formatting issues

### 2️⃣ Feature Processing
- Scaled numeric features using **StandardScaler**
- Converted categorical features using **One-Hot Encoding**
- Used **ColumnTransformer** to apply correct processing to each feature type

### 3️⃣ Handling Imbalanced Data
- Fraud cases were much fewer than non-fraud cases
- Applied **SMOTE** only on training data to avoid data leakage
- Ensured fair model learning for fraud detection

### 4️⃣ Model Building
- Implemented:
  - **Logistic Regression** (baseline model)
  - **Random Forest Classifier** (advanced model)
- Used pipelines to combine preprocessing, SMOTE, and model training

### 5️⃣ Model Evaluation
- Evaluated models using:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC-AUC and PR-AUC
- Focused more on **Recall and F1-score**, since missing fraud is costly

### 6️⃣ Model Selection & Saving
- Selected the best model based on **F1-score**
- Saved the complete pipeline using `joblib` for future use

---

## 🧠 Key Insights
- Accuracy alone is not reliable for fraud datasets
- Random Forest performed better than Logistic Regression
- SMOTE significantly improved fraud detection ability
- PR-AUC provided better evaluation than ROC-AUC for imbalanced data

---

## 🛠️ Tools & Technologies
- **Programming**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, imbalanced-learn  
- **Techniques**: SMOTE, Pipelines, Classification Models  

---

## 📌 Recommendations / Future Improvements
- Try advanced models like **XGBoost, LightGBM, CatBoost**
- Tune model hyperparameters for better performance
- Optimize classification threshold instead of using default 0.5
- Add feature importance and SHAP for model explainability
- Build a prediction script for real-time fraud detection

---

## 📚 What I Learned
- How to work with **imbalanced datasets**
- Importance of proper evaluation metrics in fraud problems
- Building clean and reusable ML pipelines
- Translating technical results into business-friendly insights

---
