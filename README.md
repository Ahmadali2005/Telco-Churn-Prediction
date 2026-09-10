# Telco Customer Churn Prediction & Retention System

An end-to-end Machine Learning solution designed to predict customer churn for a telecommunications provider. The project covers data cleaning, handling categorical encoding, model evaluation, feature interpretability, and a web deployment interface using **Streamlit**.

---

## Business Problem

Customer acquisition is significantly more expensive than customer retention. By identifying churn risk early, telecommunications companies can proactively offer targeted retention strategies (e.g., discounts, contract upgrades) to high-risk customers, preserving long-term revenue.

---

## Tech Stack & Tools

* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn (`ColumnTransformer`, `OneHotEncoder`, `LogisticRegression`, `RandomForestClassifier`)
* **Web Framework & UI:** Streamlit
* **Model Persistence:** Joblib

---

## Machine Learning Pipeline

1. **Data Cleaning & Handling Missing Values:**
   * Handled missing values in `TotalCharges` by coercing types and imputing appropriately.
   * Dropped identifier features (`customerID`) to eliminate noise.

2. **Preprocessing & Feature Encoding:**
   * Applied `OneHotEncoder` via `ColumnTransformer` to categorical features.
   * Passed numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) safely without leakage.
   * Prevented **Data Leakage** by executing `fit_transform` strictly on `X_train` and applying `transform` on `X_test`.

3. **Model Training & Evaluation:**
   * Trained baseline models including **Logistic Regression** and **Random Forest**.
   * Evaluated models using `classification_report` and `confusion_matrix` with a focus on **Recall** for Class 1 (Churners) to minimize false negatives.

---

## Model Performance

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82%** | **0.70** | **0.57** | **0.63** |
| Random Forest | 80% | 0.66 | 0.47 | 0.55 |

> **Key Finding:** Logistic Regression outperformed Random Forest across all critical metrics, making it the selected model for deployment.

---

## Key Feature Insights

Analysis of model coefficients revealed the primary drivers for customer churn:

* **Top Churn Drivers (+ Coefficients):**
  * Short-term contracts (`Month-to-month`)
  * Payment method via `Electronic check`
  * Lack of `TechSupport` and `OnlineSecurity`

* **Top Retention Factors (- Coefficients):**
  * Long-term contracts (`Two year`)
  * Subscriptions with `DSL` Internet Service
  * Active add-on services (`OnlineSecurity_Yes`, `TechSupport_Yes`)

---

## Web Application Demo

The project includes an interactive web application built with **Streamlit** that allows users to input customer details and view instant churn probability predictions alongside actionable business recommendations.

### Running the App Locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Ahmadali2005/telco churn prediction.git]
   cd telco churn prediction


## Install dependencies:
    
 pip install -r requirements.txt

## Run the Streamlit application:

 streamlit run app.py

## Repository Structure:

 app.py                  # Streamlit web application interface
 preprocessor.pkl        # Saved ColumnTransformer preprocessor
 model1.pkl              # Trained Logistic Regression model
 Telco_Churn.ipynb       # Complete Jupyter Notebook (EDA, Training, Evaluation)
 README.md               # Project documentation

    
