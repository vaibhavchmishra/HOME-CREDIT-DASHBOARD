import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
st.title('Credit Card Analysis'); df=require_dataset('credit_card'); st.dataframe(df.head(100),use_container_width=True)
for c in ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_PAYMENT_TOTAL_CURRENT']:
    if c in df: st.plotly_chart(px.histogram(df,x=c,nbins=40),use_container_width=True)
