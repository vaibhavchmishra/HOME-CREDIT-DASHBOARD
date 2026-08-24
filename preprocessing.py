import pandas as pd
import numpy as np
def numeric_columns(df): return df.select_dtypes(include=np.number).columns.tolist()
def categorical_columns(df): return df.select_dtypes(include=['object','category','bool']).columns.tolist()
def missing_summary(df):
    s=df.isna().sum().sort_values(ascending=False); out=pd.DataFrame({'missing_count':s,'missing_pct':(s/len(df)*100).round(2)})
    return out[out.missing_count>0]
def cap_outliers_iqr(series,factor=1.5):
    q1,q3=series.quantile([.25,.75]); iqr=q3-q1; return series.clip(q1-factor*iqr,q3+factor*iqr)
