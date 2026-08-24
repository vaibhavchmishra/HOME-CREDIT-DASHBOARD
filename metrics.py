def application_kpis(df):
    n=len(df); d=int(df.TARGET.sum()) if 'TARGET' in df else 0
    return {'Applications':f'{n:,}','Defaults':f'{d:,}','Default Rate':f'{d/n*100:.2f}%' if n else '0.00%','Median Income':f'{df.AMT_INCOME_TOTAL.median():,.0f}' if 'AMT_INCOME_TOTAL' in df else 'N/A','Median Credit':f'{df.AMT_CREDIT.median():,.0f}' if 'AMT_CREDIT' in df else 'N/A'}
def default_rate(df): return float(df.TARGET.mean()*100) if 'TARGET' in df and len(df) else 0.0
