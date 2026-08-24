import streamlit as st, plotly.express as px
from utils.data_loader import require_dataset
st.title('Bureau Balance Analysis'); df=require_dataset('bureau_balance'); st.dataframe(df.head(100),use_container_width=True)
if 'STATUS' in df: st.plotly_chart(px.bar(df.STATUS.value_counts(dropna=False).reset_index(),x='STATUS',y='count'),use_container_width=True)
