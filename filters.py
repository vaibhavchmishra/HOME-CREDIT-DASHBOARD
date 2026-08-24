import streamlit as st
def sidebar_filters(df):
    f={}
    if 'TARGET' in df: f['TARGET']=st.sidebar.multiselect('Default status',[0,1],default=[0,1])
    return f
def apply_filters(df,filters):
    out=df
    for c,v in filters.items():
        if v: out=out[out[c].isin(v)]
    return out
