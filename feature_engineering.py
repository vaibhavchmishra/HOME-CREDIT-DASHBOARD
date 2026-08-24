import numpy as np
def add_application_features(df):
    out=df.copy()
    if 'DAYS_BIRTH' in out: out['AGE_YEARS']=(-out['DAYS_BIRTH']/365.25).round(1)
    if 'DAYS_EMPLOYED' in out: out['EMPLOYED_YEARS']=(-out['DAYS_EMPLOYED'].replace(365243,np.nan)/365.25).round(1)
    if {'AMT_CREDIT','AMT_INCOME_TOTAL'}<=set(out.columns): out['CREDIT_INCOME_RATIO']=out.AMT_CREDIT/out.AMT_INCOME_TOTAL.replace(0,np.nan)
    if {'AMT_ANNUITY','AMT_INCOME_TOTAL'}<=set(out.columns): out['ANNUITY_INCOME_RATIO']=out.AMT_ANNUITY/out.AMT_INCOME_TOTAL.replace(0,np.nan)
    if {'AMT_CREDIT','AMT_GOODS_PRICE'}<=set(out.columns): out['CREDIT_GOODS_RATIO']=out.AMT_CREDIT/out.AMT_GOODS_PRICE.replace(0,np.nan)
    return out
