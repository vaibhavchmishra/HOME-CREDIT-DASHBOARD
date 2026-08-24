# Home Credit Default Risk — Streamlit Analytics Dashboard

Complete Streamlit EDA and risk-analysis project for the Kaggle Home Credit Default Risk dataset.

## Data
Place these real competition files in `data/`:
- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

The project does not fabricate Kaggle records. Placeholder files are included only to preserve the structure and should be replaced with the real CSVs.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages
20 Streamlit pages cover executive overview, data quality, missing values, outliers, demographics, income, employment, family/housing, loan applications, affordability, default risk, risk factors, bureau history, bureau balance, previous applications, POS/CASH, installments, credit cards, customer segmentation, and executive recommendations.
