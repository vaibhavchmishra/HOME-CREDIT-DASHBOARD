import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
st.title('Previous Applications'); df=require_dataset('previous_application'); st.dataframe(df.head(100),use_container_width=True)
if 'NAME_CONTRACT_STATUS' in df: st.plotly_chart(px.bar(df.NAME_CONTRACT_STATUS.value_counts().reset_index(),x='NAME_CONTRACT_STATUS',y='count'),use_container_width=True)
