import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
st.title('Credit & Affordability'); df=add_application_features(require_dataset('application_train'))
for c in ['CREDIT_INCOME_RATIO','ANNUITY_INCOME_RATIO']:
    if c in df: st.plotly_chart(px.histogram(df,x=c,color='TARGET',nbins=40),use_container_width=True)



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Affordability | Home Credit",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Affordability Analysis")

st.markdown(
    """
    Analyze whether customers can reasonably afford their requested
    credit based on income, credit amount, annuity, goods price,
    and repayment difficulty.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/application_train.csv"
    )


df = load_data()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "NAME_FAMILY_STATUS",
    "TARGET"
]

available_required = [
    col
    for col in required_columns
    if col in df.columns
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.warning(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Avoid division by zero
income_safe = df["AMT_INCOME_TOTAL"].replace(
    0, np.nan
)

credit_safe = df["AMT_CREDIT"].replace(
    0, np.nan
)

goods_safe = df["AMT_GOODS_PRICE"].replace(
    0, np.nan
)


# ------------------------------------------------------------
# Credit-to-Income Ratio
# ------------------------------------------------------------

df["CREDIT_TO_INCOME"] = (
    df["AMT_CREDIT"] / income_safe
)


# ------------------------------------------------------------
# Annuity-to-Income Ratio
# ------------------------------------------------------------

df["ANNUITY_TO_INCOME"] = (
    df["AMT_ANNUITY"] / income_safe
)


# ------------------------------------------------------------
# Credit-to-Goods-Price Ratio
# ------------------------------------------------------------

df["CREDIT_TO_GOODS"] = (
    df["AMT_CREDIT"] / goods_safe
)


# ------------------------------------------------------------
# Income-to-Credit Ratio
# ------------------------------------------------------------

df["INCOME_TO_CREDIT"] = (
    df["AMT_INCOME_TOTAL"] / credit_safe
)


# ------------------------------------------------------------
# Currency conversion for visualization
# ------------------------------------------------------------

df["INCOME_LAKHS"] = (
    df["AMT_INCOME_TOTAL"] / 100000
)

df["CREDIT_LAKHS"] = (
    df["AMT_CREDIT"] / 100000
)

df["ANNUITY_LAKHS"] = (
    df["AMT_ANNUITY"] / 100000
)

df["GOODS_PRICE_LAKHS"] = (
    df["AMT_GOODS_PRICE"] / 100000
)


# ============================================================
# TARGET LABEL
# ============================================================

df["TARGET_LABEL"] = df["TARGET"].map(
    {
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    }
)


# ============================================================
# AFFORDABILITY RISK GROUP
# ============================================================

def affordability_group(value):

    if pd.isna(value):
        return "Unknown"

    if value <= 1:
        return "Low Burden"

    elif value <= 2:
        return "Moderate Burden"

    elif value <= 4:
        return "High Burden"

    else:
        return "Very High Burden"


df["AFFORDABILITY_GROUP"] = (
    df["CREDIT_TO_INCOME"]
    .apply(affordability_group)
)


# ============================================================
# ANNUITY BURDEN GROUP
# ============================================================

def annuity_group(value):

    if pd.isna(value):
        return "Unknown"

    if value <= 0.10:
        return "Low Burden"

    elif value <= 0.20:
        return "Moderate Burden"

    elif value <= 0.40:
        return "High Burden"

    else:
        return "Very High Burden"


df["ANNUITY_BURDEN_GROUP"] = (
    df["ANNUITY_TO_INCOME"]
    .apply(annuity_group)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Affordability Filters")

filtered_df = df.copy()


# ============================================================
# INCOME TYPE
# ============================================================

income_types = sorted(
    df["NAME_INCOME_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_income_types = st.sidebar.multiselect(
    "Income Type",
    income_types,
    default=list(income_types)
)

if selected_income_types:

    filtered_df = filtered_df[
        filtered_df["NAME_INCOME_TYPE"]
        .astype(str)
        .isin(selected_income_types)
    ]


# ============================================================
# CONTRACT TYPE
# ============================================================

contract_types = sorted(
    df["NAME_CONTRACT_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_contract_types = st.sidebar.multiselect(
    "Contract Type",
    contract_types,
    default=list(contract_types)
)

if selected_contract_types:

    filtered_df = filtered_df[
        filtered_df["NAME_CONTRACT_TYPE"]
        .astype(str)
        .isin(selected_contract_types)
    ]


# ============================================================
# HOUSING TYPE
# ============================================================

housing_types = sorted(
    df["NAME_HOUSING_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_housing_types = st.sidebar.multiselect(
    "Housing Type",
    housing_types,
    default=list(housing_types)
)

if selected_housing_types:

    filtered_df = filtered_df[
        filtered_df["NAME_HOUSING_TYPE"]
        .astype(str)
        .isin(selected_housing_types)
    ]


# ============================================================
# FAMILY STATUS
# ============================================================

family_statuses = sorted(
    df["NAME_FAMILY_STATUS"]
    .dropna()
    .astype(str)
    .unique()
)

selected_family_statuses = st.sidebar.multiselect(
    "Family Status",
    family_statuses,
    default=list(family_statuses)
)

if selected_family_statuses:

    filtered_df = filtered_df[
        filtered_df["NAME_FAMILY_STATUS"]
        .astype(str)
        .isin(selected_family_statuses)
    ]


# ============================================================
# TARGET
# ============================================================

target_options = {
    0: "No Payment Difficulties",
    1: "Payment Difficulties"
}

selected_targets = st.sidebar.multiselect(
    "Repayment Status",
    options=list(target_options.keys()),
    default=list(target_options.keys()),
    format_func=lambda x: target_options[x]
)

if selected_targets:

    filtered_df = filtered_df[
        filtered_df["TARGET"].isin(
            selected_targets
        )
    ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

avg_income = filtered_df[
    "AMT_INCOME_TOTAL"
].mean()

avg_credit = filtered_df[
    "AMT_CREDIT"
].mean()

avg_credit_income = filtered_df[
    "CREDIT_TO_INCOME"
].mean()

avg_annuity_income = filtered_df[
    "ANNUITY_TO_INCOME"
].mean()

payment_difficulty_rate = (
    filtered_df["TARGET"].mean() * 100
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Affordability Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "Average Income",
        f"₹{avg_income:,.0f}"
    )

with col3:

    st.metric(
        "Average Credit",
        f"₹{avg_credit:,.0f}"
    )

with col4:

    st.metric(
        "Credit / Income",
        f"{avg_credit_income:.2f}x"
    )

with col5:

    st.metric(
        "Annuity / Income",
        f"{avg_annuity_income:.2%}"
    )


st.divider()


# ============================================================
# INCOME & CREDIT DISTRIBUTION
# ============================================================

st.subheader("💰 Income & Credit Distribution")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Income Distribution
# ------------------------------------------------------------

with col1:

    income_data = filtered_df[
        filtered_df["INCOME_LAKHS"] > 0
    ].copy()

    income_limit = income_data[
        "INCOME_LAKHS"
    ].quantile(0.99)

    income_data = income_data[
        income_data["INCOME_LAKHS"]
        <= income_limit
    ]

    fig_income = px.histogram(
        income_data,
        x="INCOME_LAKHS",
        nbins=40,
        title="Annual Income Distribution",
        labels={
            "INCOME_LAKHS":
                "Annual Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True
    )


# ------------------------------------------------------------
# Credit Distribution
# ------------------------------------------------------------

with col2:

    credit_data = filtered_df[
        filtered_df["CREDIT_LAKHS"] > 0
    ].copy()

    credit_limit = credit_data[
        "CREDIT_LAKHS"
    ].quantile(0.99)

    credit_data = credit_data[
        credit_data["CREDIT_LAKHS"]
        <= credit_limit
    ]

    fig_credit = px.histogram(
        credit_data,
        x="CREDIT_LAKHS",
        nbins=40,
        title="Credit Amount Distribution",
        labels={
            "CREDIT_LAKHS":
                "Credit Amount (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_credit,
        use_container_width=True
    )


# ============================================================
# AFFORDABILITY RATIOS
# ============================================================

st.subheader("📐 Affordability Ratio Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit-to-Income
# ------------------------------------------------------------

with col1:

    ratio_data = filtered_df[
        filtered_df["CREDIT_TO_INCOME"]
        .between(0, 10)
    ]

    fig_credit_income_ratio = px.histogram(
        ratio_data,
        x="CREDIT_TO_INCOME",
        nbins=40,
        title="Credit-to-Income Ratio",
        labels={
            "CREDIT_TO_INCOME":
                "Credit / Annual Income"
        }
    )

    st.plotly_chart(
        fig_credit_income_ratio,
        use_container_width=True
    )


# ------------------------------------------------------------
# Annuity-to-Income
# ------------------------------------------------------------

with col2:

    annuity_ratio_data = filtered_df[
        filtered_df["ANNUITY_TO_INCOME"]
        .between(0, 1)
    ]

    fig_annuity_income_ratio = px.histogram(
        annuity_ratio_data,
        x="ANNUITY_TO_INCOME",
        nbins=40,
        title="Annuity-to-Income Ratio",
        labels={
            "ANNUITY_TO_INCOME":
                "Annuity / Annual Income"
        }
    )

    st.plotly_chart(
        fig_annuity_income_ratio,
        use_container_width=True
    )


# ============================================================
# INCOME VS CREDIT
# ============================================================

st.subheader("📈 Income vs Credit")

scatter_data = filtered_df[
    [
        "INCOME_LAKHS",
        "CREDIT_LAKHS",
        "TARGET_LABEL"
    ]
].dropna()

scatter_data = scatter_data[
    (scatter_data["INCOME_LAKHS"] > 0) &
    (scatter_data["CREDIT_LAKHS"] > 0)
]

income_limit = scatter_data[
    "INCOME_LAKHS"
].quantile(0.99)

credit_limit = scatter_data[
    "CREDIT_LAKHS"
].quantile(0.99)

scatter_data = scatter_data[
    (scatter_data["INCOME_LAKHS"] <= income_limit) &
    (scatter_data["CREDIT_LAKHS"] <= credit_limit)
]


fig_income_credit = px.scatter(
    scatter_data,
    x="INCOME_LAKHS",
    y="CREDIT_LAKHS",
    color="TARGET_LABEL",
    opacity=0.55,
    title="Annual Income vs Credit Amount",
    labels={
        "INCOME_LAKHS":
            "Annual Income (₹ Lakhs)",
        "CREDIT_LAKHS":
            "Credit Amount (₹ Lakhs)",
        "TARGET_LABEL":
            "Repayment Status"
    }
)

st.plotly_chart(
    fig_income_credit,
    use_container_width=True
)


# ============================================================
# INCOME VS ANNUITY
# ============================================================

st.subheader("💵 Income vs Annuity")

income_annuity = filtered_df[
    [
        "INCOME_LAKHS",
        "ANNUITY_LAKHS",
        "TARGET_LABEL"
    ]
].dropna()

income_annuity = income_annuity[
    (income_annuity["INCOME_LAKHS"] > 0) &
    (income_annuity["ANNUITY_LAKHS"] > 0)
]

income_limit = income_annuity[
    "INCOME_LAKHS"
].quantile(0.99)

annuity_limit = income_annuity[
    "ANNUITY_LAKHS"
].quantile(0.99)

income_annuity = income_annuity[
    (income_annuity["INCOME_LAKHS"] <= income_limit) &
    (income_annuity["ANNUITY_LAKHS"] <= annuity_limit)
]


fig_income_annuity = px.scatter(
    income_annuity,
    x="INCOME_LAKHS",
    y="ANNUITY_LAKHS",
    color="TARGET_LABEL",
    opacity=0.55,
    title="Annual Income vs Loan Annuity",
    labels={
        "INCOME_LAKHS":
            "Annual Income (₹ Lakhs)",
        "ANNUITY_LAKHS":
            "Annuity (₹ Lakhs)",
        "TARGET_LABEL":
            "Repayment Status"
    }
)

st.plotly_chart(
    fig_income_annuity,
    use_container_width=True
)


# ============================================================
# AFFORDABILITY BY INCOME TYPE
# ============================================================

st.subheader("👔 Affordability by Income Type")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit-to-Income by Income Type
# ------------------------------------------------------------

with col1:

    income_type_credit = (
        filtered_df
        .groupby("NAME_INCOME_TYPE")[
            "CREDIT_TO_INCOME"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    income_type_credit.columns = [
        "Income Type",
        "Average Credit-to-Income"
    ]

    fig_income_type_credit = px.bar(
        income_type_credit,
        x="Average Credit-to-Income",
        y="Income Type",
        orientation="h",
        text="Average Credit-to-Income",
        title="Credit-to-Income by Income Type"
    )

    st.plotly_chart(
        fig_income_type_credit,
        use_container_width=True
    )


# ------------------------------------------------------------
# Annuity-to-Income by Income Type
# ------------------------------------------------------------

with col2:

    income_type_annuity = (
        filtered_df
        .groupby("NAME_INCOME_TYPE")[
            "ANNUITY_TO_INCOME"
        ]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    income_type_annuity.columns = [
        "Income Type",
        "Annuity-to-Income (%)"
    ]

    fig_income_type_annuity = px.bar(
        income_type_annuity,
        x="Annuity-to-Income (%)",
        y="Income Type",
        orientation="h",
        text="Annuity-to-Income (%)",
        title="Annuity Burden by Income Type"
    )

    st.plotly_chart(
        fig_income_type_annuity,
        use_container_width=True
    )


# ============================================================
# AFFORDABILITY VS TARGET
# ============================================================

st.subheader("⚠️ Affordability vs Payment Difficulty")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit-to-Income Box Plot
# ------------------------------------------------------------

with col1:

    box_data = filtered_df[
        filtered_df["CREDIT_TO_INCOME"]
        .between(0, 10)
    ].copy()

    fig_box_credit = px.box(
        box_data,
        x="TARGET_LABEL",
        y="CREDIT_TO_INCOME",
        points=False,
        title="Credit-to-Income by Repayment Status",
        labels={
            "TARGET_LABEL":
                "Repayment Status",
            "CREDIT_TO_INCOME":
                "Credit / Annual Income"
        }
    )

    st.plotly_chart(
        fig_box_credit,
        use_container_width=True
    )


# ------------------------------------------------------------
# Annuity-to-Income Box Plot
# ------------------------------------------------------------

with col2:

    box_annuity = filtered_df[
        filtered_df["ANNUITY_TO_INCOME"]
        .between(0, 1)
    ].copy()

    fig_box_annuity = px.box(
        box_annuity,
        x="TARGET_LABEL",
        y="ANNUITY_TO_INCOME",
        points=False,
        title="Annuity-to-Income by Repayment Status",
        labels={
            "TARGET_LABEL":
                "Repayment Status",
            "ANNUITY_TO_INCOME":
                "Annuity / Annual Income"
        }
    )

    st.plotly_chart(
        fig_box_annuity,
        use_container_width=True
    )


# ============================================================
# AFFORDABILITY BY HOUSING
# ============================================================

st.subheader("🏠 Affordability by Housing Type")

housing_affordability = (
    filtered_df
    .groupby("NAME_HOUSING_TYPE")[
        "CREDIT_TO_INCOME"
    ]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

housing_affordability.columns = [
    "Housing Type",
    "Average Credit-to-Income"
]


fig_housing_affordability = px.bar(
    housing_affordability,
    x="Average Credit-to-Income",
    y="Housing Type",
    orientation="h",
    text="Average Credit-to-Income",
    title="Average Credit-to-Income by Housing Type"
)

st.plotly_chart(
    fig_housing_affordability,
    use_container_width=True
)


# ============================================================
# AFFORDABILITY BY FAMILY STATUS
# ============================================================

st.subheader("👨‍👩‍👧 Affordability by Family Status")

family_affordability = (
    filtered_df
    .groupby("NAME_FAMILY_STATUS")[
        "CREDIT_TO_INCOME"
    ]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

family_affordability.columns = [
    "Family Status",
    "Average Credit-to-Income"
]


fig_family_affordability = px.bar(
    family_affordability,
    x="Average Credit-to-Income",
    y="Family Status",
    orientation="h",
    text="Average Credit-to-Income",
    title="Average Credit-to-Income by Family Status"
)

st.plotly_chart(
    fig_family_affordability,
    use_container_width=True
)


# ============================================================
# CREDIT-TO-GOODS RATIO
# ============================================================

st.subheader("🛒 Credit-to-Goods-Price Analysis")

goods_ratio = filtered_df[
    filtered_df["CREDIT_TO_GOODS"]
    .between(0, 5)
]

fig_goods_ratio = px.histogram(
    goods_ratio,
    x="CREDIT_TO_GOODS",
    nbins=40,
    title="Credit-to-Goods-Price Ratio",
    labels={
        "CREDIT_TO_GOODS":
            "Credit / Goods Price"
    }
)

st.plotly_chart(
    fig_goods_ratio,
    use_container_width=True
)


# ============================================================
# AFFORDABILITY RISK GROUP
# ============================================================

st.subheader("🚦 Affordability Risk Groups")

risk_order = [
    "Low Burden",
    "Moderate Burden",
    "High Burden",
    "Very High Burden"
]

risk_data = (
    filtered_df
    .groupby(
        "AFFORDABILITY_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .mul(100)
    .reindex(risk_order)
    .reset_index()
)

risk_data.columns = [
    "Affordability Group",
    "Payment Difficulty Rate"
]


fig_risk = px.bar(
    risk_data,
    x="Affordability Group",
    y="Payment Difficulty Rate",
    text="Payment Difficulty Rate",
    title="Payment Difficulty Rate by Credit Affordability Group",
    labels={
        "Payment Difficulty Rate":
            "Payment Difficulty Rate (%)"
    }
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# DATA TABLE
# ============================================================

st.subheader("📋 Credit Affordability Data")

display_columns = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_GOODS",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "NAME_FAMILY_STATUS",
    "TARGET"
]

available_columns = [
    col
    for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    height=400
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = (
    filtered_df[available_columns]
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇️ Download Credit Affordability Data",
    data=csv_data,
    file_name="credit_affordability_filtered.csv",
    mime="text/csv"
)