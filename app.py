import streamlit as st
from utils.data_loader import load_application_train
from utils.feature_engineering import add_application_features
from utils.metrics import application_kpis
st.set_page_config(page_title='Home Credit Streamlit Dashboard',page_icon='🏦',layout='wide')
st.title('🏦 Home Credit Streamlit')



#......................................................
# PAGE CONFIGURATION:
#......................................................

pages = {

    " app": [
        st.Page(
            "pages/01_Executive_Overview.py",
            title="Executive Overview",
            icon="📊"
        ),
        st.Page(
            "pages/02_Data_Quality.py",
            title="Data Quality",
            icon="🔍"
        ),
         st.Page(
        "pages/03_Missing_Value_Analysis.py",
         title="Missing Value Analysis",
        icon="🧩"
),
        st.Page(
            "pages/04_Outlier_Analysis.py",
             title="Outlier Analysis",
             icon="📦"
        ),
         st.Page(
            "pages/05_Customer_Demographics.py",
            title="Customer Demographics",
            icon="👥"
        ),
        st.Page(
            "pages/06_Income_Analysis.py",
            title="Income Analysis",
            icon="💰"
        ),
        st.Page(
            "pages/07_Employment_Analysis.py",
            title="Employment Analysis",
            icon="🔗"
        ),
        st.Page(
            "pages/08_Family_Housing_Analysis.py",
            title="Famliy Housing Analysis",
            icon="👨‍👩‍👧"
        ),
        st.Page(
            "pages/09_Loan_Application_Analysis.py",
            title="Loan Application Analysis",
            icon="💼"
        ),
        st.Page(
            "pages/10_Credit_Affordability.py",
            title="Credit Affordability",
            icon="⭐"
        ),
        st.Page(
            "pages/11_Default_Risk_EDA.py",
            title="Default Risk EDA",
            icon="🎓"
        ),
        st.Page(
            "pages/12_Risk_Factor_Analysis.py",
            title="Risk Factor Analysis",
            icon="🏦"
        ),
        st.Page(
            "pages/13_Bureau_Credit_History.py",
            title="Bureau Credit History",
            icon="📈"
        ),
        st.Page(
            "pages/14_Bureau_Balance_Analysis.py",
            title="Bureau Balance Analysis",
            icon="🏠"
        ),
        st.Page(
            "pages/15_Previous_Applications.py",
            title="Previous Applications",
            icon="👤"
        ),
        st.Page(
            "pages/16_POS_CASH_Analysis.py",
            title="POS CASH Analysis",
            icon="🌍"
        ),
        st.Page(
            "pages/17_Installment_Payment_Analysis.py",
            title="Installment Payment Analysis",
            icon="📝"
        ),

        st.Page(
            "pages/18_Credit_Card_Analysis.py",
            title="Credit Card Analysis",
            icon="💳"
        ),
        st.Page(
            "pages/19_Customer_Risk_Segmentation.py",
            title="Customer Risk Segmentation",
            icon="🗂️"
        ),
         st.Page(
            "pages/20_Executive_Insights_Recommendations.py",
            title="Executive Insights Recommendations",
            icon="🗂️"
          ),
    ],
}

pg = st.navigation(
    pages,
    position="sidebar"
)
pg.run()