import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Bureau Credit History'); df=require_dataset('bureau'); st.metric('Rows',f'{len(df):,}'); st.dataframe(df.head(100),use_container_width=True); nums=df.select_dtypes('number').columns.tolist();
if nums:
    c=st.selectbox('Numeric variable',nums); st.plotly_chart(px.histogram(df,x=c,nbins=40),use_container_width=True)



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bureau Credit Analysis | Home Credit",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bureau Credit Analysis")

st.markdown(
    """
    Analyze customers' historical credit accounts, previous credit
    exposure, debt, overdue amounts, credit status, credit types,
    and credit utilization using the Home Credit bureau dataset.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_bureau():

    return pd.read_csv(
        "data/bureau.csv"
    )


df = load_bureau()


# ============================================================
# BASIC INFORMATION
# ============================================================

st.sidebar.header("🔎 Bureau Credit Filters")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Credit age
# ------------------------------------------------------------

df["CREDIT_AGE_YEARS"] = (
    -df["DAYS_CREDIT"] / 365.25
)


# ------------------------------------------------------------
# Debt-to-credit ratio
# ------------------------------------------------------------

credit_sum = df[
    "AMT_CREDIT_SUM"
].replace(0, np.nan)


df["DEBT_TO_CREDIT"] = (
    df["AMT_CREDIT_SUM_DEBT"] /
    credit_sum
)


# ------------------------------------------------------------
# Overdue-to-credit ratio
# ------------------------------------------------------------

df["OVERDUE_TO_CREDIT"] = (
    df["AMT_CREDIT_SUM_OVERDUE"] /
    credit_sum
)


# ------------------------------------------------------------
# Credit utilization
#
# Debt / credit limit
# ------------------------------------------------------------

credit_limit = df[
    "AMT_CREDIT_SUM_LIMIT"
].replace(0, np.nan)


df["CREDIT_UTILIZATION"] = (
    df["AMT_CREDIT_SUM_DEBT"] /
    credit_limit
)


# ============================================================
# SIDEBAR FILTER - CREDIT ACTIVE
# ============================================================

if "CREDIT_ACTIVE" in df.columns:

    active_values = sorted(
        df["CREDIT_ACTIVE"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_active = st.sidebar.multiselect(
        "Credit Status",
        active_values,
        default=active_values
    )

    if selected_active:

        df = df[
            df["CREDIT_ACTIVE"]
            .astype(str)
            .isin(selected_active)
        ]


# ============================================================
# SIDEBAR FILTER - CREDIT TYPE
# ============================================================

if "CREDIT_TYPE" in df.columns:

    credit_types = sorted(
        df["CREDIT_TYPE"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_types = st.sidebar.multiselect(
        "Credit Type",
        credit_types,
        default=credit_types
    )

    if selected_types:

        df = df[
            df["CREDIT_TYPE"]
            .astype(str)
            .isin(selected_types)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

bureau_accounts = len(df)

unique_customers = df[
    "SK_ID_CURR"
].nunique()

total_credit = df[
    "AMT_CREDIT_SUM"
].sum()

total_debt = df[
    "AMT_CREDIT_SUM_DEBT"
].sum()

total_overdue = df[
    "AMT_CREDIT_SUM_OVERDUE"
].sum()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Bureau Credit Overview")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Bureau Accounts",
        f"{bureau_accounts:,}"
    )


with col2:

    st.metric(
        "Unique Customers",
        f"{unique_customers:,}"
    )


with col3:

    st.metric(
        "Total Credit Exposure",
        f"₹{total_credit:,.0f}"
    )


with col4:

    st.metric(
        "Total Debt",
        f"₹{total_debt:,.0f}"
    )


with col5:

    st.metric(
        "Total Overdue",
        f"₹{total_overdue:,.0f}"
    )


st.divider()


# ============================================================
# CREDIT ACCOUNT OVERVIEW
# ============================================================

st.subheader("📁 Credit Account Overview")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit Active Status
# ------------------------------------------------------------

with col1:

    if "CREDIT_ACTIVE" in df.columns:

        active_summary = (
            df["CREDIT_ACTIVE"]
            .value_counts()
            .reset_index()
        )

        active_summary.columns = [
            "Credit Status",
            "Accounts"
        ]

        fig_active = px.pie(
            active_summary,
            names="Credit Status",
            values="Accounts",
            hole=0.45,
            title="Credit Account Status"
        )

        st.plotly_chart(
            fig_active,
            use_container_width=True
        )


# ------------------------------------------------------------
# Credit Type
# ------------------------------------------------------------

with col2:

    type_summary = (
        df["CREDIT_TYPE"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    type_summary.columns = [
        "Credit Type",
        "Accounts"
    ]

    fig_type = px.bar(
        type_summary,
        x="Accounts",
        y="Credit Type",
        orientation="h",
        text="Accounts",
        title="Top Credit Types"
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True
    )


# ============================================================
# CREDIT CURRENCY
# ============================================================

if "CREDIT_CURRENCY" in df.columns:

    currency_summary = (
        df["CREDIT_CURRENCY"]
        .value_counts()
        .reset_index()
    )

    currency_summary.columns = [
        "Currency",
        "Accounts"
    ]

    fig_currency = px.bar(
        currency_summary,
        x="Currency",
        y="Accounts",
        text="Accounts",
        title="Credit Accounts by Currency"
    )

    st.plotly_chart(
        fig_currency,
        use_container_width=True
    )


# ============================================================
# CREDIT ACCOUNTS PER CUSTOMER
# ============================================================

st.subheader("👤 Customer Credit Account Count")


customer_credit_count = (
    df.groupby("SK_ID_CURR")
    .size()
    .reset_index(
        name="CREDIT_ACCOUNT_COUNT"
    )
)


fig_customer_count = px.histogram(
    customer_credit_count,
    x="CREDIT_ACCOUNT_COUNT",
    nbins=30,
    title="Number of Bureau Credit Accounts per Customer",
    labels={
        "CREDIT_ACCOUNT_COUNT":
            "Number of Credit Accounts"
    }
)


st.plotly_chart(
    fig_customer_count,
    use_container_width=True
)


# ============================================================
# CREDIT HISTORY
# ============================================================

st.subheader("📅 Credit History Analysis")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit Age
# ------------------------------------------------------------

with col1:

    age_data = df[
        "CREDIT_AGE_YEARS"
    ].dropna()

    age_data = age_data[
        age_data.between(0, 30)
    ]

    fig_age = px.histogram(
        x=age_data,
        nbins=40,
        title="Previous Credit Age Distribution",
        labels={
            "x":
                "Credit Age (Years)"
        }
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


# ------------------------------------------------------------
# Average Credit by Type
# ------------------------------------------------------------

with col2:

    avg_credit_type = (
        df.groupby(
            "CREDIT_TYPE"
        )["AMT_CREDIT_SUM"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    avg_credit_type.columns = [
        "Credit Type",
        "Average Credit"
    ]

    fig_avg_credit = px.bar(
        avg_credit_type,
        x="Average Credit",
        y="Credit Type",
        orientation="h",
        text="Average Credit",
        title="Average Credit Amount by Credit Type"
    )

    st.plotly_chart(
        fig_avg_credit,
        use_container_width=True
    )


# ============================================================
# CREDIT EXPOSURE BY TYPE
# ============================================================

col1, col2 = st.columns(2)


with col1:

    exposure_type = (
        df.groupby(
            "CREDIT_TYPE"
        )["AMT_CREDIT_SUM"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    exposure_type.columns = [
        "Credit Type",
        "Total Exposure"
    ]

    fig_exposure = px.bar(
        exposure_type,
        x="Total Exposure",
        y="Credit Type",
        orientation="h",
        title="Total Credit Exposure by Credit Type"
    )

    st.plotly_chart(
        fig_exposure,
        use_container_width=True
    )


with col2:

    debt_type = (
        df.groupby(
            "CREDIT_TYPE"
        )["AMT_CREDIT_SUM_DEBT"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    debt_type.columns = [
        "Credit Type",
        "Average Debt"
    ]

    fig_debt_type = px.bar(
        debt_type,
        x="Average Debt",
        y="Credit Type",
        orientation="h",
        title="Average Debt by Credit Type"
    )

    st.plotly_chart(
        fig_debt_type,
        use_container_width=True
    )


# ============================================================
# DEBT ANALYSIS
# ============================================================

st.subheader("💰 Debt & Overdue Analysis")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Debt Distribution
# ------------------------------------------------------------

with col1:

    debt_data = df[
        "AMT_CREDIT_SUM_DEBT"
    ].dropna()

    debt_limit = debt_data.quantile(
        0.99
    )

    debt_data = debt_data[
        debt_data <= debt_limit
    ]

    fig_debt = px.histogram(
        x=debt_data,
        nbins=40,
        title="Current Debt Distribution",
        labels={
            "x":
                "Current Debt"
        }
    )

    st.plotly_chart(
        fig_debt,
        use_container_width=True
    )


# ------------------------------------------------------------
# Overdue Distribution
# ------------------------------------------------------------

with col2:

    overdue_data = df[
        "AMT_CREDIT_SUM_OVERDUE"
    ].dropna()

    overdue_data = overdue_data[
        overdue_data > 0
    ]

    overdue_limit = (
        overdue_data.quantile(0.99)
        if len(overdue_data) > 0
        else 0
    )

    overdue_data = overdue_data[
        overdue_data <= overdue_limit
    ]

    fig_overdue = px.histogram(
        x=overdue_data,
        nbins=40,
        title="Overdue Amount Distribution",
        labels={
            "x":
                "Overdue Amount"
        }
    )

    st.plotly_chart(
        fig_overdue,
        use_container_width=True
    )


# ============================================================
# DEBT VS CREDIT
# ============================================================

st.subheader("📈 Debt vs Credit Exposure")


scatter_data = df[
    [
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "CREDIT_ACTIVE"
    ]
].dropna()


scatter_data = scatter_data[
    (scatter_data["AMT_CREDIT_SUM"] > 0) &
    (scatter_data["AMT_CREDIT_SUM_DEBT"] >= 0)
]


credit_limit = scatter_data[
    "AMT_CREDIT_SUM"
].quantile(0.99)


debt_limit = scatter_data[
    "AMT_CREDIT_SUM_DEBT"
].quantile(0.99)


scatter_data = scatter_data[
    (scatter_data["AMT_CREDIT_SUM"]
     <= credit_limit) &
    (scatter_data["AMT_CREDIT_SUM_DEBT"]
     <= debt_limit)
]


fig_debt_credit = px.scatter(
    scatter_data,
    x="AMT_CREDIT_SUM",
    y="AMT_CREDIT_SUM_DEBT",
    color="CREDIT_ACTIVE",
    opacity=0.5,
    title="Credit Amount vs Current Debt",
    labels={
        "AMT_CREDIT_SUM":
            "Credit Amount",
        "AMT_CREDIT_SUM_DEBT":
            "Current Debt",
        "CREDIT_ACTIVE":
            "Credit Status"
    }
)


st.plotly_chart(
    fig_debt_credit,
    use_container_width=True
)


# ============================================================
# OVERDUE VS CREDIT
# ============================================================

st.subheader("⚠️ Overdue vs Credit")


overdue_scatter = df[
    [
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_ACTIVE"
    ]
].dropna()


overdue_scatter = overdue_scatter[
    overdue_scatter["AMT_CREDIT_SUM"] > 0
]


credit_limit = overdue_scatter[
    "AMT_CREDIT_SUM"
].quantile(0.99)


overdue_limit = overdue_scatter[
    "AMT_CREDIT_SUM_OVERDUE"
].quantile(0.99)


overdue_scatter = overdue_scatter[
    (overdue_scatter["AMT_CREDIT_SUM"]
     <= credit_limit) &
    (overdue_scatter["AMT_CREDIT_SUM_OVERDUE"]
     <= overdue_limit)
]


fig_overdue_credit = px.scatter(
    overdue_scatter,
    x="AMT_CREDIT_SUM",
    y="AMT_CREDIT_SUM_OVERDUE",
    color="CREDIT_ACTIVE",
    opacity=0.5,
    title="Credit Amount vs Overdue Amount",
    labels={
        "AMT_CREDIT_SUM":
            "Credit Amount",
        "AMT_CREDIT_SUM_OVERDUE":
            "Overdue Amount"
    }
)


st.plotly_chart(
    fig_overdue_credit,
    use_container_width=True
)


# ============================================================
# CREDIT UTILIZATION
# ============================================================

st.subheader("📊 Credit Utilization")


utilization_data = df[
    "CREDIT_UTILIZATION"
].dropna()


utilization_data = utilization_data[
    utilization_data.between(0, 2)
]


fig_utilization = px.histogram(
    x=utilization_data,
    nbins=40,
    title="Credit Utilization Distribution",
    labels={
        "x":
            "Debt / Credit Limit"
    }
)


st.plotly_chart(
    fig_utilization,
    use_container_width=True
)


# ============================================================
# UTILIZATION BY CREDIT TYPE
# ============================================================

utilization_type = df[
    [
        "CREDIT_TYPE",
        "CREDIT_UTILIZATION"
    ]
].dropna()


utilization_type = utilization_type[
    utilization_type["CREDIT_UTILIZATION"]
    .between(0, 2)
]


top_types = (
    utilization_type["CREDIT_TYPE"]
    .value_counts()
    .head(10)
    .index
)


utilization_type = utilization_type[
    utilization_type["CREDIT_TYPE"]
    .isin(top_types)
]


fig_utilization_type = px.box(
    utilization_type,
    x="CREDIT_TYPE",
    y="CREDIT_UTILIZATION",
    points=False,
    title="Credit Utilization by Credit Type",
    labels={
        "CREDIT_TYPE":
            "Credit Type",
        "CREDIT_UTILIZATION":
            "Debt / Credit Limit"
    }
)


st.plotly_chart(
    fig_utilization_type,
    use_container_width=True
)


# ============================================================
# TOP CUSTOMERS
# ============================================================

st.subheader("👤 Top Customers by Previous Credit Exposure")


customer_exposure = (
    df.groupby("SK_ID_CURR")
    .agg(
        Total_Credit_Exposure=(
            "AMT_CREDIT_SUM",
            "sum"
        ),
        Total_Debt=(
            "AMT_CREDIT_SUM_DEBT",
            "sum"
        ),
        Total_Overdue=(
            "AMT_CREDIT_SUM_OVERDUE",
            "sum"
        ),
        Credit_Accounts=(
            "SK_ID_BUREAU",
            "count"
        )
    )
    .reset_index()
)


top_customers = (
    customer_exposure
    .sort_values(
        "Total_Credit_Exposure",
        ascending=False
    )
    .head(20)
)


fig_top_customers = px.bar(
    top_customers,
    x="Total_Credit_Exposure",
    y="SK_ID_CURR",
    orientation="h",
    text="Total_Credit_Exposure",
    title="Top 20 Customers by Previous Credit Exposure",
    labels={
        "SK_ID_CURR":
            "Customer ID",
        "Total_Credit_Exposure":
            "Total Credit Exposure"
    }
)


st.plotly_chart(
    fig_top_customers,
    use_container_width=True
)


# ============================================================
# CUSTOMER CREDIT SUMMARY
# ============================================================

st.subheader("📋 Customer-Level Bureau Summary")


st.dataframe(
    customer_exposure.sort_values(
        "Total_Credit_Exposure",
        ascending=False
    ),
    use_container_width=True,
    height=400
)


# ============================================================
# RAW BUREAU DATA
# ============================================================

st.subheader("🔍 Bureau Credit Data")


st.dataframe(
    df,
    use_container_width=True,
    height=400
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = (
    df.to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Filtered Bureau Data",
    data=csv_data,
    file_name="bureau_credit_filtered.csv",
    mime="text/csv"
)