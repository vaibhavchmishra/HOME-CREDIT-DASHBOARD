import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
from utils.metrics import application_kpis

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.title("Executive Overview")
st.title("Home Credit Executive Overview")
st.write(
    "This dashboard provides an overview of loan applications, "
    "customer characteristics, credit risk, and housing information."
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: black;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "application_train.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "application_train.csv was not found. "
        "Please place it inside the data/ folder."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

data = df.copy()


# Convert negative DAYS_BIRTH into age
if "DAYS_BIRTH" in data.columns:

    data["AGE_YEARS"] = (
        -data["DAYS_BIRTH"] / 365.25
    )


# Convert negative employment days
if "DAYS_EMPLOYED" in data.columns:

    data["EMPLOYED_YEARS"] = np.where(
        data["DAYS_EMPLOYED"] < 0,
        -data["DAYS_EMPLOYED"] / 365.25,
        np.nan
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")


# Gender filter
if "CODE_GENDER" in data.columns:

    gender_options = sorted(
        data["CODE_GENDER"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options
    )

    if selected_gender:
        data = data[
            data["CODE_GENDER"].isin(selected_gender)
        ]


# Contract type filter
if "NAME_CONTRACT_TYPE" in data.columns:

    contract_options = sorted(
        data["NAME_CONTRACT_TYPE"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_contract = st.sidebar.multiselect(
        "Contract Type",
        options=contract_options,
        default=contract_options
    )

    if selected_contract:
        data = data[
            data["NAME_CONTRACT_TYPE"].isin(
                selected_contract
            )
        ]


# Income type filter
if "NAME_INCOME_TYPE" in data.columns:

    income_options = sorted(
        data["NAME_INCOME_TYPE"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_income = st.sidebar.multiselect(
        "Income Type",
        options=income_options,
        default=income_options
    )

    if selected_income:
        data = data[
            data["NAME_INCOME_TYPE"].isin(
                selected_income
            )
        ]


# Education filter
if "NAME_EDUCATION_TYPE" in data.columns:

    education_options = sorted(
        data["NAME_EDUCATION_TYPE"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_education = st.sidebar.multiselect(
        "Education",
        options=education_options,
        default=education_options
    )

    if selected_education:
        data = data[
            data["NAME_EDUCATION_TYPE"].isin(
                selected_education
            )
        ]


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_mean(column):

    if column in data.columns:

        return data[column].mean()

    return 0


def safe_sum(column):

    if column in data.columns:

        return data[column].sum()

    return 0


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Portfolio Summary")


total_applications = len(data)

total_credit = safe_sum("AMT_CREDIT")

average_credit = safe_mean("AMT_CREDIT")

average_income = safe_mean("AMT_INCOME_TOTAL")

average_age = safe_mean("AGE_YEARS")

if "TARGET" in data.columns:

    default_rate = data["TARGET"].mean() * 100

else:

    default_rate = 0


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Applications",
        f"{total_applications:,}"
    )

with col2:

    st.metric(
        "Total Credit",
        f"{total_credit:,.0f}"
    )

with col3:

    st.metric(
        "Average Credit",
        f"{average_credit:,.0f}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Average Income",
        f"{average_income:,.0f}"
    )

with col5:

    st.metric(
        "Average Age",
        f"{average_age:.1f} years"
    )

with col6:

    st.metric(
        "Payment Difficulty Rate",
        f"{default_rate:.2f}%"
    )


st.divider()


# ============================================================
# CHART 1 — TARGET DISTRIBUTION
# ============================================================

st.subheader("1️⃣ Loan Payment Status")

if "TARGET" in data.columns:

    target_data = (
        data["TARGET"]
        .value_counts()
        .reset_index()
    )

    target_data.columns = [
        "TARGET",
        "COUNT"
    ]

    target_data["STATUS"] = target_data["TARGET"].map({
        0: "Repaid / No Payment Difficulty",
        1: "Payment Difficulty"
    })

    fig_target = px.pie(
        target_data,
        names="STATUS",
        values="COUNT",
        hole=0.55,
        title="Loan Application Status"
    )

    fig_target.update_traces(
        textinfo="percent+label"
    )

    fig_target.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_target,
        use_container_width=True
    )


# ============================================================
# CHART 2 & 3
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Applications by Contract Type
# ------------------------------------------------------------

with col1:

    if "NAME_CONTRACT_TYPE" in data.columns:

        contract_data = (
            data["NAME_CONTRACT_TYPE"]
            .value_counts()
            .reset_index()
        )

        contract_data.columns = [
            "Contract Type",
            "Applications"
        ]

        fig_contract = px.bar(
            contract_data,
            x="Contract Type",
            y="Applications",
            text="Applications",
            title="Applications by Contract Type"
        )

        fig_contract.update_traces(
            textposition="outside"
        )

        fig_contract.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_contract,
            use_container_width=True
        )


# ------------------------------------------------------------
# Applications by Income Type
# ------------------------------------------------------------

with col2:

    if "NAME_INCOME_TYPE" in data.columns:

        income_data = (
            data["NAME_INCOME_TYPE"]
            .value_counts()
            .reset_index()
        )

        income_data.columns = [
            "Income Type",
            "Applications"
        ]

        income_data = income_data.sort_values(
            "Applications",
            ascending=True
        )

        fig_income = px.bar(
            income_data,
            x="Applications",
            y="Income Type",
            orientation="h",
            text="Applications",
            title="Applications by Income Type"
        )

        fig_income.update_traces(
            textposition="outside"
        )

        fig_income.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_income,
            use_container_width=True
        )


# ============================================================
# CHART 4 — EDUCATION
# ============================================================

st.subheader("2️⃣ Customer Education Profile")

if "NAME_EDUCATION_TYPE" in data.columns:

    education_data = (
        data["NAME_EDUCATION_TYPE"]
        .value_counts()
        .reset_index()
    )

    education_data.columns = [
        "Education",
        "Applications"
    ]

    education_data = education_data.sort_values(
        "Applications",
        ascending=True
    )

    fig_education = px.bar(
        education_data,
        x="Applications",
        y="Education",
        orientation="h",
        text="Applications",
        title="Applications by Education Type"
    )

    fig_education.update_traces(
        textposition="outside"
    )

    fig_education.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_education,
        use_container_width=True
    )


# ============================================================
# CHART 5 & 6 — DISTRIBUTIONS
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Age Distribution
# ------------------------------------------------------------

with col1:

    if "AGE_YEARS" in data.columns:

        age_data = data[
            data["AGE_YEARS"].between(18, 100)
        ]

        fig_age = px.histogram(
            age_data,
            x="AGE_YEARS",
            nbins=30,
            title="Customer Age Distribution"
        )

        fig_age.update_layout(
            xaxis_title="Age (Years)",
            yaxis_title="Number of Applications",
            height=450
        )

        st.plotly_chart(
            fig_age,
            use_container_width=True
        )


# ------------------------------------------------------------
# Income Distribution
# ------------------------------------------------------------

with col2:

    if "AMT_INCOME_TOTAL" in data.columns:

        income_values = data[
            data["AMT_INCOME_TOTAL"] > 0
        ]

        # Limit extreme values for visualization
        income_values = income_values[
            income_values["AMT_INCOME_TOTAL"]
            <= income_values["AMT_INCOME_TOTAL"].quantile(0.99)
        ]

        fig_income_dist = px.histogram(
            income_values,
            x="AMT_INCOME_TOTAL",
            nbins=40,
            title="Annual Income Distribution"
        )

        fig_income_dist.update_layout(
            xaxis_title="Annual Income",
            yaxis_title="Number of Applications",
            height=450
        )

        st.plotly_chart(
            fig_income_dist,
            use_container_width=True
        )


# ============================================================
# CHART 7 — CREDIT DISTRIBUTION
# ============================================================

st.subheader("3️⃣ Loan Amount Analysis")

if "AMT_CREDIT" in data.columns:

    credit_values = data[
        data["AMT_CREDIT"] > 0
    ]

    credit_values = credit_values[
        credit_values["AMT_CREDIT"]
        <= credit_values["AMT_CREDIT"].quantile(0.99)
    ]

    fig_credit = px.histogram(
        credit_values,
        x="AMT_CREDIT",
        nbins=40,
        title="Credit Amount Distribution"
    )

    fig_credit.update_layout(
        xaxis_title="Credit Amount",
        yaxis_title="Number of Applications",
        height=450
    )

    st.plotly_chart(
        fig_credit,
        use_container_width=True
    )


# ============================================================
# CHART 8 — INCOME VS CREDIT
# ============================================================

st.subheader("4️⃣ Income vs Credit Relationship")

if (
    "AMT_INCOME_TOTAL" in data.columns
    and "AMT_CREDIT" in data.columns
):

    scatter_data = data[
        [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT"
        ]
    ].dropna()

    scatter_data = scatter_data[
        scatter_data["AMT_INCOME_TOTAL"] > 0
    ]

    # Remove extreme values for better visualization
    income_limit = scatter_data[
        "AMT_INCOME_TOTAL"
    ].quantile(0.99)

    credit_limit = scatter_data[
        "AMT_CREDIT"
    ].quantile(0.99)

    scatter_data = scatter_data[
        (scatter_data["AMT_INCOME_TOTAL"] <= income_limit)
        &
        (scatter_data["AMT_CREDIT"] <= credit_limit)
    ]

    fig_scatter = px.scatter(
        scatter_data,
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        opacity=0.45,
        title="Annual Income vs Credit Amount",
        labels={
            "AMT_INCOME_TOTAL": "Annual Income",
            "AMT_CREDIT": "Credit Amount"
        }
    )

    fig_scatter.update_layout(
        height=550
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# ============================================================
# CHART 9 — DEFAULT RATE BY INCOME TYPE
# ============================================================

st.subheader("5️⃣ Payment Difficulty Analysis")

if (
    "NAME_INCOME_TYPE" in data.columns
    and "TARGET" in data.columns
):

    default_income = (
        data
        .groupby("NAME_INCOME_TYPE")["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    default_income.columns = [
        "Income Type",
        "Payment Difficulty Rate"
    ]

    default_income = default_income.sort_values(
        "Payment Difficulty Rate",
        ascending=True
    )

    fig_default_income = px.bar(
        default_income,
        x="Payment Difficulty Rate",
        y="Income Type",
        orientation="h",
        text="Payment Difficulty Rate",
        title="Payment Difficulty Rate by Income Type"
    )

    fig_default_income.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_default_income.update_layout(
        xaxis_title="Payment Difficulty Rate (%)",
        yaxis_title="Income Type",
        height=500
    )

    st.plotly_chart(
        fig_default_income,
        use_container_width=True
    )


# ============================================================
# CHART 10 — DEFAULT RATE BY EDUCATION
# ============================================================

if (
    "NAME_EDUCATION_TYPE" in data.columns
    and "TARGET" in data.columns
):

    default_education = (
        data
        .groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    default_education.columns = [
        "Education",
        "Payment Difficulty Rate"
    ]

    default_education = default_education.sort_values(
        "Payment Difficulty Rate",
        ascending=True
    )

    fig_default_education = px.bar(
        default_education,
        x="Payment Difficulty Rate",
        y="Education",
        orientation="h",
        text="Payment Difficulty Rate",
        title="Payment Difficulty Rate by Education"
    )

    fig_default_education.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_default_education.update_layout(
        xaxis_title="Payment Difficulty Rate (%)",
        yaxis_title="Education",
        height=500
    )

    st.plotly_chart(
        fig_default_education,
        use_container_width=True
    )


# ============================================================
# CHART 11 — DEFAULT RATE BY GENDER
# ============================================================

if (
    "CODE_GENDER" in data.columns
    and "TARGET" in data.columns
):

    default_gender = (
        data
        .groupby("CODE_GENDER")["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    default_gender.columns = [
        "Gender",
        "Payment Difficulty Rate"
    ]

    fig_gender = px.bar(
        default_gender,
        x="Gender",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment Difficulty Rate by Gender"
    )

    fig_gender.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_gender.update_layout(
        yaxis_title="Payment Difficulty Rate (%)",
        height=450
    )

    st.plotly_chart(
        fig_gender,
        use_container_width=True
    )


# ============================================================
# CHART 12 — EXTERNAL CREDIT SCORES
# ============================================================

st.subheader("6️⃣ External Credit Score Analysis")

available_scores = [
    column
    for column in [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]
    if column in data.columns
]


if available_scores:

    score_data = data[
        available_scores
    ].melt(
        var_name="Credit Score",
        value_name="Score"
    ).dropna()

    fig_scores = px.histogram(
        score_data,
        x="Score",
        color="Credit Score",
        barmode="overlay",
        opacity=0.65,
        nbins=30,
        title="External Credit Score Distribution"
    )

    fig_scores.update_layout(
        xaxis_title="External Credit Score",
        yaxis_title="Number of Applications",
        height=500
    )

    st.plotly_chart(
        fig_scores,
        use_container_width=True
    )


# ============================================================
# DATA SUMMARY
# ============================================================

st.subheader("📋 Executive Overview Data Summary")

summary = pd.DataFrame({
    "Metric": [
        "Applications",
        "Total Credit",
        "Average Credit",
        "Average Income",
        "Average Age",
        "Payment Difficulty Rate"
    ],
    "Value": [
        f"{total_applications:,}",
        f"{total_credit:,.0f}",
        f"{average_credit:,.0f}",
        f"{average_income:,.0f}",
        f"{average_age:.1f} years",
        f"{default_rate:.2f}%"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

st.subheader("⬇️ Download")

csv_data = data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Data",
    data=csv_data,
    file_name="home_credit_executive_filtered.csv",
    mime="text/csv"
)