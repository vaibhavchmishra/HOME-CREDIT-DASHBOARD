import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
st.title('POS CASH Analysis'); df=require_dataset('pos_cash'); st.dataframe(df.head(100),use_container_width=True)
for c in ['MONTHS_BALANCE','SK_DPD','CNT_INSTALMENT']:
    if c in df: st.plotly_chart(px.histogram(df,x=c,nbins=40),use_container_width=True)
