import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
from utils.preprocessing import missing_summary
st.title('Missing Value Analysis'); df=require_dataset('application_train'); m=missing_summary(df).reset_index().rename(columns={'index':'column'}); st.dataframe(m,use_container_width=True)
if len(m): st.plotly_chart(px.bar(m.head(30),x='missing_pct',y='column',orientation='h'),use_container_width=True)





