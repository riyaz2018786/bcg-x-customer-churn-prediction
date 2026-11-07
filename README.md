# BCG X Customer Churn Prediction & Price Sensitivity Analysis

## Project Overview
This project was completed as part of the **BCG X Data Science Job Simulation**. It investigates customer churn for **PowerCo**, a major gas and electricity utility provider servicing small and medium-sized enterprises (SMEs). 

The primary business objective is to diagnose the drivers of churn, test whether price sensitivity is the primary churn catalyst, and build a predictive classifier to identify at-risk customers.

---

## Repository Structure
```text
├── data/          # Raw & cleaned customer and pricing datasets
├── notebooks/     # Chronological execution pipelines (EDA, Feature Engineering, Modeling)
├── reports/       # Executive summary presentation deck for client leadership
├── src/           # Modular Python production scripts
├── requirements.txt
└── README.md
```

---

## Workflow & Key Insights

1. **Exploratory Data Analysis (`01_exploratory_data_analysis.ipynb`)**
   - Churn rate across SME customers was approximately ~9.7%.
   - Price sensitivity was evaluated across fixed vs. variable rates for electricity and gas.
   - Handled skewness in energy consumption metrics via log transformations.

2. **Feature Engineering (`02_feature_engineering.ipynb`)**
   - Computed price changes across off-peak, peak, and mid-peak periods.
   - Built spread and differential features between historical peak and off-peak prices.
   - Calculated customer tenure, net margins, and contract duration metrics.

3. **Predictive Modeling (`03_modeling_and_evaluation.ipynb`)**
   - Trained a **Random Forest Classifier** with hyperparameter tuning to handle class imbalance.
   - Evaluated model using **AUC-ROC**, Precision, Recall, and F1-score.
   - **Key Finding:** Net margin, consumption history, and tenure are significantly stronger predictors of churn than price changes alone.

---

## Business Recommendations
- **Targeted Retention:** Rather than applying broad price cuts, retention discounts (up to 20%) should be targeted strictly at high-margin customers predicted to churn.
- **Contract Incentives:** Offer extended contract loyalty incentives to customers reaching critical tenure thresholds.

---

## Setup & Reproduction
```bash
git clone [https://github.com/riyaz2018786/bcg-x-customer-churn-prediction.git](https://github.com/riyaz2018786/bcg-x-customer-churn-prediction.git)
cd bcg-x-customer-churn-prediction
pip install -r requirements.txt
```
