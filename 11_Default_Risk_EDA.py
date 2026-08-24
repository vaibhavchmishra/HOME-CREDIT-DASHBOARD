import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Default Risk EDA'); df=require_dataset('application_train'); rate=df.TARGET.mean()*100; st.metric('Default Rate',f'{rate:.2f}%')
if 'NAME_CONTRACT_TYPE' in df: st.plotly_chart(px.bar(df.groupby('NAME_CONTRACT_TYPE').TARGET.mean().mul(100).reset_index(),x='NAME_CONTRACT_TYPE',y='TARGET'),use_container_width=True)



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Default Risk Analysis",
    page_icon="⚠️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚠️ Default-Risk Analysis")

st.markdown("""
Analyze historical payment difficulties across customer demographics,
income, employment, housing, loan characteristics, and external
credit scores.
""")


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
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Target labels
# ------------------------------------------------------------

df["TARGET_LABEL"] = df["TARGET"].map({
    0: "No Payment Difficulties",
    1: "Payment Difficulties"
})


# ------------------------------------------------------------
# Age in years
# ------------------------------------------------------------

df["AGE_YEARS"] = (
    -df["DAYS_BIRTH"] / 365.25
)


# ------------------------------------------------------------
# Age groups
# ------------------------------------------------------------

df["AGE_GROUP"] = pd.cut(
    df["AGE_YEARS"],
    bins=[
        0,
        25,
        30,
        35,
        40,
        45,
        50,
        60,
        100
    ],
    labels=[
        "<25",
        "25-30",
        "30-35",
        "35-40",
        "40-45",
        "45-50",
        "50-60",
        "60+"
    ]
)


# ------------------------------------------------------------
# Employment years
# ------------------------------------------------------------

# Home Credit contains DAYS_EMPLOYED as negative days.
# 365243 is commonly used as a special/anomalous value.

df["EMPLOYMENT_YEARS"] = np.where(
    df["DAYS_EMPLOYED"] == 365243,
    np.nan,
    -df["DAYS_EMPLOYED"] / 365.25
)


# ------------------------------------------------------------
# Employment groups
# ------------------------------------------------------------

df["EMPLOYMENT_GROUP"] = pd.cut(
    df["EMPLOYMENT_YEARS"],
    bins=[
        -0.01,
        1,
        3,
        5,
        10,
        20,
        np.inf
    ],
    labels=[
        "<1 year",
        "1-3 years",
        "3-5 years",
        "5-10 years",
        "10-20 years",
        "20+ years"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Default Risk Filters")

filtered_df = df.copy()


# ============================================================
# CONTRACT FILTER
# ============================================================

contract_types = sorted(
    df["NAME_CONTRACT_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_contract = st.sidebar.multiselect(
    "Contract Type",
    contract_types,
    default=contract_types
)

if selected_contract:

    filtered_df = filtered_df[
        filtered_df["NAME_CONTRACT_TYPE"]
        .astype(str)
        .isin(selected_contract)
    ]


# ============================================================
# GENDER FILTER
# ============================================================

gender_values = sorted(
    df["CODE_GENDER"]
    .dropna()
    .astype(str)
    .unique()
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_values,
    default=gender_values
)

if selected_gender:

    filtered_df = filtered_df[
        filtered_df["CODE_GENDER"]
        .astype(str)
        .isin(selected_gender)
    ]


# ============================================================
# EDUCATION FILTER
# ============================================================

education_values = sorted(
    df["NAME_EDUCATION_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_education = st.sidebar.multiselect(
    "Education",
    education_values,
    default=education_values
)

if selected_education:

    filtered_df = filtered_df[
        filtered_df["NAME_EDUCATION_TYPE"]
        .astype(str)
        .isin(selected_education)
    ]


# ============================================================
# HOUSING FILTER
# ============================================================

housing_values = sorted(
    df["NAME_HOUSING_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)

selected_housing = st.sidebar.multiselect(
    "Housing Type",
    housing_values,
    default=housing_values
)

if selected_housing:

    filtered_df = filtered_df[
        filtered_df["NAME_HOUSING_TYPE"]
        .astype(str)
        .isin(selected_housing)
    ]


# ============================================================
# TARGET FILTER
# ============================================================

target_options = {
    0: "No Payment Difficulties",
    1: "Payment Difficulties"
}

selected_target = st.sidebar.multiselect(
    "Repayment Status",
    options=list(target_options.keys()),
    default=list(target_options.keys()),
    format_func=lambda x: target_options[x]
)

if selected_target:

    filtered_df = filtered_df[
        filtered_df["TARGET"].isin(
            selected_target
        )
    ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_applications = len(filtered_df)

payment_difficulty = (
    filtered_df["TARGET"] == 1
).sum()

no_payment_difficulty = (
    filtered_df["TARGET"] == 0
).sum()

target_rate = (
    filtered_df["TARGET"].mean() * 100
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Default Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Applications",
        f"{total_applications:,}"
    )

with col2:

    st.metric(
        "No Payment Difficulties",
        f"{no_payment_difficulty:,}"
    )

with col3:

    st.metric(
        "Payment Difficulties",
        f"{payment_difficulty:,}"
    )

with col4:

    st.metric(
        "Payment-Difficulty Rate",
        f"{target_rate:.2f}%"
    )


st.divider()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

st.subheader("🎯 Overall Default-Risk Distribution")

target_data = (
    filtered_df["TARGET_LABEL"]
    .value_counts()
    .reset_index()
)

target_data.columns = [
    "Status",
    "Applications"
]


fig_target = px.pie(
    target_data,
    names="Status",
    values="Applications",
    hole=0.45,
    title="Payment Difficulty Distribution"
)

st.plotly_chart(
    fig_target,
    use_container_width=True
)


# ============================================================
# CONTRACT RISK
# ============================================================

st.subheader("📄 Contract Risk Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Contract vs Target
# ------------------------------------------------------------

with col1:

    contract_target = (
        filtered_df
        .groupby(
            [
                "NAME_CONTRACT_TYPE",
                "TARGET_LABEL"
            ]
        )
        .size()
        .reset_index(name="Applications")
    )

    fig_contract_target = px.bar(
        contract_target,
        x="NAME_CONTRACT_TYPE",
        y="Applications",
        color="TARGET_LABEL",
        barmode="stack",
        title="Contract Type vs Payment Difficulty"
    )

    st.plotly_chart(
        fig_contract_target,
        use_container_width=True
    )


# ------------------------------------------------------------
# Default Rate by Contract
# ------------------------------------------------------------

with col2:

    contract_rate = (
        filtered_df
        .groupby("NAME_CONTRACT_TYPE")[
            "TARGET"
        ]
        .mean()
        .mul(100)
        .reset_index()
    )

    contract_rate.columns = [
        "Contract Type",
        "Payment Difficulty Rate"
    ]

    fig_contract_rate = px.bar(
        contract_rate,
        x="Contract Type",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Contract Type"
    )

    st.plotly_chart(
        fig_contract_rate,
        use_container_width=True
    )


# ============================================================
# DEMOGRAPHIC RISK
# ============================================================

st.subheader("👥 Customer Demographic Risk")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Gender
# ------------------------------------------------------------

with col1:

    gender_rate = (
        filtered_df
        .groupby("CODE_GENDER")[
            "TARGET"
        ]
        .mean()
        .mul(100)
        .reset_index()
    )

    gender_rate.columns = [
        "Gender",
        "Payment Difficulty Rate"
    ]

    fig_gender = px.bar(
        gender_rate,
        x="Gender",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Gender"
    )

    st.plotly_chart(
        fig_gender,
        use_container_width=True
    )


# ------------------------------------------------------------
# Education
# ------------------------------------------------------------

with col2:

    education_rate = (
        filtered_df
        .groupby("NAME_EDUCATION_TYPE")[
            "TARGET"
        ]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    education_rate.columns = [
        "Education",
        "Payment Difficulty Rate"
    ]

    fig_education = px.bar(
        education_rate,
        x="Payment Difficulty Rate",
        y="Education",
        orientation="h",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Education"
    )

    st.plotly_chart(
        fig_education,
        use_container_width=True
    )


# ============================================================
# FAMILY & HOUSING RISK
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Family Status
# ------------------------------------------------------------

with col1:

    family_rate = (
        filtered_df
        .groupby("NAME_FAMILY_STATUS")[
            "TARGET"
        ]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    family_rate.columns = [
        "Family Status",
        "Payment Difficulty Rate"
    ]

    fig_family = px.bar(
        family_rate,
        x="Payment Difficulty Rate",
        y="Family Status",
        orientation="h",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Family Status"
    )

    st.plotly_chart(
        fig_family,
        use_container_width=True
    )


# ------------------------------------------------------------
# Housing
# ------------------------------------------------------------

with col2:

    housing_rate = (
        filtered_df
        .groupby("NAME_HOUSING_TYPE")[
            "TARGET"
        ]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    housing_rate.columns = [
        "Housing Type",
        "Payment Difficulty Rate"
    ]

    fig_housing = px.bar(
        housing_rate,
        x="Payment Difficulty Rate",
        y="Housing Type",
        orientation="h",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Housing Type"
    )

    st.plotly_chart(
        fig_housing,
        use_container_width=True
    )


# ============================================================
# INCOME TYPE
# ============================================================

st.subheader("💼 Income Type Risk")

income_rate = (
    filtered_df
    .groupby("NAME_INCOME_TYPE")[
        "TARGET"
    ]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

income_rate.columns = [
    "Income Type",
    "Payment Difficulty Rate"
]


fig_income = px.bar(
    income_rate,
    x="Payment Difficulty Rate",
    y="Income Type",
    orientation="h",
    text="Payment Difficulty Rate",
    title="Payment-Difficulty Rate by Income Type"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)


# ============================================================
# AGE RISK
# ============================================================

st.subheader("🎂 Age & Employment Risk")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Age Group
# ------------------------------------------------------------

with col1:

    age_rate = (
        filtered_df
        .dropna(subset=["AGE_GROUP"])
        .groupby(
            "AGE_GROUP",
            observed=True
        )["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    age_rate.columns = [
        "Age Group",
        "Payment Difficulty Rate"
    ]

    fig_age = px.bar(
        age_rate,
        x="Age Group",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Age Group"
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


# ------------------------------------------------------------
# Employment Group
# ------------------------------------------------------------

with col2:

    employment_rate = (
        filtered_df
        .dropna(subset=["EMPLOYMENT_GROUP"])
        .groupby(
            "EMPLOYMENT_GROUP",
            observed=True
        )["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    employment_rate.columns = [
        "Employment Group",
        "Payment Difficulty Rate"
    ]

    fig_employment = px.bar(
        employment_rate,
        x="Employment Group",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment-Difficulty Rate by Employment Duration"
    )

    st.plotly_chart(
        fig_employment,
        use_container_width=True
    )


# ============================================================
# FINANCIAL RISK
# ============================================================

st.subheader("💰 Financial Risk")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Income
# ------------------------------------------------------------

with col1:

    income_box = filtered_df[
        ["TARGET_LABEL", "AMT_INCOME_TOTAL"]
    ].dropna()

    income_limit = income_box[
        "AMT_INCOME_TOTAL"
    ].quantile(0.99)

    income_box = income_box[
        income_box["AMT_INCOME_TOTAL"]
        <= income_limit
    ]

    fig_income_box = px.box(
        income_box,
        x="TARGET_LABEL",
        y="AMT_INCOME_TOTAL",
        points=False,
        title="Income by Payment Status",
        labels={
            "TARGET_LABEL":
                "Payment Status",
            "AMT_INCOME_TOTAL":
                "Annual Income"
        }
    )

    st.plotly_chart(
        fig_income_box,
        use_container_width=True
    )


# ------------------------------------------------------------
# Credit
# ------------------------------------------------------------

with col2:

    credit_box = filtered_df[
        ["TARGET_LABEL", "AMT_CREDIT"]
    ].dropna()

    credit_limit = credit_box[
        "AMT_CREDIT"
    ].quantile(0.99)

    credit_box = credit_box[
        credit_box["AMT_CREDIT"]
        <= credit_limit
    ]

    fig_credit_box = px.box(
        credit_box,
        x="TARGET_LABEL",
        y="AMT_CREDIT",
        points=False,
        title="Credit Amount by Payment Status",
        labels={
            "TARGET_LABEL":
                "Payment Status",
            "AMT_CREDIT":
                "Credit Amount"
        }
    )

    st.plotly_chart(
        fig_credit_box,
        use_container_width=True
    )


# ------------------------------------------------------------
# Annuity
# ------------------------------------------------------------

with col3:

    annuity_box = filtered_df[
        ["TARGET_LABEL", "AMT_ANNUITY"]
    ].dropna()

    annuity_limit = annuity_box[
        "AMT_ANNUITY"
    ].quantile(0.99)

    annuity_box = annuity_box[
        annuity_box["AMT_ANNUITY"]
        <= annuity_limit
    ]

    fig_annuity_box = px.box(
        annuity_box,
        x="TARGET_LABEL",
        y="AMT_ANNUITY",
        points=False,
        title="Annuity by Payment Status",
        labels={
            "TARGET_LABEL":
                "Payment Status",
            "AMT_ANNUITY":
                "Annuity"
        }
    )

    st.plotly_chart(
        fig_annuity_box,
        use_container_width=True
    )


# ============================================================
# EXTERNAL SCORE ANALYSIS
# ============================================================

st.subheader("📊 External Score Risk Analysis")

score_columns = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

score_available = [
    col
    for col in score_columns
    if col in filtered_df.columns
]

score_cols = st.columns(
    len(score_available)
)


for container, score_column in zip(
    score_cols,
    score_available
):

    with container:

        score_data = filtered_df[
            [
                "TARGET_LABEL",
                score_column
            ]
        ].dropna()

        fig_score = px.box(
            score_data,
            x="TARGET_LABEL",
            y=score_column,
            points=False,
            title=f"{score_column} vs Payment Status"
        )

        st.plotly_chart(
            fig_score,
            use_container_width=True
        )


# ============================================================
# EXTERNAL SCORE AVERAGES
# ============================================================

if score_available:

    st.subheader("📈 Average External Scores")

    score_summary = (
        filtered_df
        .groupby("TARGET_LABEL")[
            score_available
        ]
        .mean()
        .T
        .reset_index()
    )

    score_summary.columns = (
        ["External Score"]
        + [
            str(col)
            for col in score_summary.columns[1:]
        ]
    )

    fig_score_avg = px.bar(
        score_summary,
        x="External Score",
        y=score_summary.columns[1:],
        barmode="group",
        title="Average External Scores by Payment Status"
    )

    st.plotly_chart(
        fig_score_avg,
        use_container_width=True
    )


# ============================================================
# RISK SUMMARY TABLE
# ============================================================

st.subheader("📋 Default-Risk Summary")

risk_summary = pd.DataFrame({
    "Metric": [
        "Total Applications",
        "No Payment Difficulties",
        "Payment Difficulties",
        "Payment-Difficulty Rate"
    ],
    "Value": [
        total_applications,
        no_payment_difficulty,
        payment_difficulty,
        f"{target_rate:.2f}%"
    ]
})

st.dataframe(
    risk_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA EXPLORER
# ============================================================

st.subheader("🔍 Default-Risk Data")

display_columns = [
    "SK_ID_CURR",
    "TARGET",
    "TARGET_LABEL",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

available_display_columns = [
    col
    for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[
        available_display_columns
    ],
    use_container_width=True,
    height=400
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = (
    filtered_df[
        available_display_columns
    ]
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇️ Download Filtered Default-Risk Data",
    data=csv_data,
    file_name="default_risk_filtered.csv",
    mime="text/csv"
)