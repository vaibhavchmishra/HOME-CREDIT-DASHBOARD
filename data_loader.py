from pathlib import Path
import pandas as pd
import streamlit as st
DATA_DIR=Path(__file__).resolve().parents[1]/'data'
DATASETS={'application_train':'application_train.csv','bureau':'bureau.csv','bureau_balance':'bureau_balance.csv','previous_application':'previous_application.csv','pos_cash':'POS_CASH_balance.csv','installments':'installments_payments.csv','credit_card':'credit_card_balance.csv'}
def available_datasets(): return {k:(DATA_DIR/v).exists() and (DATA_DIR/v).stat().st_size>100 for k,v in DATASETS.items()}
@st.cache_data(show_spinner=False)
def load_csv(name):
    p=DATA_DIR/DATASETS[name]
    if not p.exists() or p.stat().st_size<=100: return pd.DataFrame()
    return pd.read_csv(p)
def require_dataset(name):
    df=load_csv(name)
    if df.empty:
        st.error(f'Missing required dataset: data/{DATASETS[name]}')
        st.stop()
    return df
def load_application_train(): return load_csv('application_train')
def load_bureau(): return load_csv('bureau')
def load_bureau_balance(): return load_csv('bureau_balance')
def load_previous_application(): return load_csv('previous_application')
def load_pos_cash(): return load_csv('pos_cash')
def load_installments(): return load_csv('installments')
def load_credit_card(): return load_csv('credit_card')
