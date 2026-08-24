import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
st.title('Installment Payment Analysis'); df=require_dataset('installments'); st.dataframe(df.head(100),use_container_width=True)
for c in ['AMT_INSTALMENT','AMT_PAYMENT','DAYS_ENTRY_PAYMENT']:
    if c in df: st.plotly_chart(px.histogram(df,x=c,nbins=40),use_container_width=True)
