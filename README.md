# Customer Churn Classification

## Project Overview

Customer retention is a critical challenge for subscription-based businesses. This project focuses on predicting customer churn for a telecommunications company using supervised machine learning techniques.

The objective is to identify customers who are likely to leave the service, enabling businesses to take proactive retention measures and improve customer satisfaction.

---

## Business Problem

Customer churn directly impacts revenue and growth. By analyzing customer demographics, service usage patterns, contract details, and billing information, this project builds a predictive model that classifies whether a customer is likely to churn.

---

## Project Objectives

- Perform exploratory data analysis (EDA) to understand churn behavior.
- Identify key factors influencing customer attrition.
- Build and evaluate multiple classification models.
- Improve model performance through feature engineering and hyperparameter tuning.
- Generate actionable business insights for customer retention strategies.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/ML-Classification-red)

- Python
- Pandas, NumPy
- Scikit-learn (6 classifiers, preprocessing, model selection)
- XGBoost
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn, Plotly
- SciPy, StatsModels
- Joblib
- OpenPyXL
- Jupyter Notebook
- VS Code

---

## Dataset

The dataset contains customer demographic information, account details, subscribed services, billing information, and churn status.

### Target Variable

| Variable | Description                                            |
| -------- | ------------------------------------------------------ |
| Churn    | Indicates whether a customer left the service (Yes/No) |

### Key Features

- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Internet service
- Online security
- Tech support
- Payment method
- Partner and dependent status

---

## Project Workflow

### 1. Data Understanding

- Examine dataset structure and feature distributions.
- Identifie missing values and data quality issues.

### 2. Data Cleaning & Preprocessing

- Handle missing and inconsistent values.
- Convert categorical variables into numerical representations.
- Scale numerical features where required.

### 3. Exploratory Data Analysis (EDA)

- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Churn distribution analysis

### 4. Handling Class Imbalance

- Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

### 5. Feature Engineering

- Encode categorical variables.
- Selecte relevant features for model training.

### 6. Model Building

Traine and evaluate 6 classification models:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

### 7. Hyperparameter Tuning

- Optimize model performance using parameter tuning techniques.

### 8. Model Evaluation

Evaluate models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## How to Run

### Clone Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

---
