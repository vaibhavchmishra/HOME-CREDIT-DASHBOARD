import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
st.title('Employment Analysis'); df=add_application_features(require_dataset('application_train'))
if 'EMPLOYED_YEARS' in df: st.plotly_chart(px.histogram(df,x='EMPLOYED_YEARS',color='TARGET',nbins=40),use_container_width=True)



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employment Analysis | Home Credit",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💼 Employment Analysis")

st.markdown("""
Analyze customer employment characteristics, including employment
duration, income type, occupation, organization, and their relationship
with income, credit, age, and payment difficulties.
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
# DATA PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# DAYS_EMPLOYED
# ------------------------------------------------------------

if "DAYS_EMPLOYED" in df.columns:

    # Replace Home Credit anomaly / special value
    df["DAYS_EMPLOYED_CLEAN"] = df[
        "DAYS_EMPLOYED"
    ].replace(
        365243,
        np.nan
    )

    # Convert negative days to employment years
    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED_CLEAN"] / 365.25
    ).round(2)


# ------------------------------------------------------------
# DAYS_BIRTH TO AGE
# ------------------------------------------------------------

if "DAYS_BIRTH" in df.columns:

    df["AGE"] = (
        -df["DAYS_BIRTH"] / 365.25
    ).round(1)


# ------------------------------------------------------------
# INCOME TO LAKHS
# ------------------------------------------------------------

if "AMT_INCOME_TOTAL" in df.columns:

    df["INCOME_LAKHS"] = (
        df["AMT_INCOME_TOTAL"] / 100000
    )


# ------------------------------------------------------------
# CREDIT TO LAKHS
# ------------------------------------------------------------

if "AMT_CREDIT" in df.columns:

    df["CREDIT_LAKHS"] = (
        df["AMT_CREDIT"] / 100000
    )


# ------------------------------------------------------------
# TARGET LABEL
# ------------------------------------------------------------

if "TARGET" in df.columns:

    df["TARGET_LABEL"] = df["TARGET"].map({
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    })


# ============================================================
# CREATE EMPLOYMENT GROUPS
# ============================================================

if "EMPLOYMENT_YEARS" in df.columns:

    df["EMPLOYMENT_GROUP"] = pd.cut(
        df["EMPLOYMENT_YEARS"],
        bins=[
            -0.01,
            1,
            3,
            5,
            10,
            np.inf
        ],
        labels=[
            "0–1 Years",
            "1–3 Years",
            "3–5 Years",
            "5–10 Years",
            "10+ Years"
        ]
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Employment Filters")

filtered_df = df.copy()


# ------------------------------------------------------------
# Income Type
# ------------------------------------------------------------

if "NAME_INCOME_TYPE" in filtered_df.columns:

    income_types = sorted(
        filtered_df["NAME_INCOME_TYPE"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_income_types = st.sidebar.multiselect(
        "Income Type",
        income_types,
        default=income_types
    )

    if selected_income_types:

        filtered_df = filtered_df[
            filtered_df["NAME_INCOME_TYPE"]
            .astype(str)
            .isin(selected_income_types)
        ]


# ------------------------------------------------------------
# Occupation
# ------------------------------------------------------------

if "OCCUPATION_TYPE" in filtered_df.columns:

    occupations = sorted(
        filtered_df["OCCUPATION_TYPE"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_occupations = st.sidebar.multiselect(
        "Occupation",
        occupations,
        default=occupations
    )

    if selected_occupations:

        filtered_df = filtered_df[
            filtered_df["OCCUPATION_TYPE"]
            .astype(str)
            .isin(selected_occupations)
        ]


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

if "TARGET" in filtered_df.columns:

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
            filtered_df["TARGET"]
            .isin(selected_target)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

employed_customers = (
    filtered_df["EMPLOYMENT_YEARS"]
    .notna()
    .sum()
    if "EMPLOYMENT_YEARS" in filtered_df.columns
    else 0
)

average_employment_years = (
    filtered_df["EMPLOYMENT_YEARS"].mean()
    if "EMPLOYMENT_YEARS" in filtered_df.columns
    else np.nan
)

median_employment_years = (
    filtered_df["EMPLOYMENT_YEARS"].median()
    if "EMPLOYMENT_YEARS" in filtered_df.columns
    else np.nan
)

payment_difficulty_rate = (
    filtered_df["TARGET"].mean() * 100
    if "TARGET" in filtered_df.columns
    else np.nan
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Employment Overview")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:
    st.metric(
        "Employment Records",
        f"{employed_customers:,}"
    )


with col3:
    st.metric(
        "Avg Employment",
        f"{average_employment_years:.2f} Years"
        if not pd.isna(average_employment_years)
        else "N/A"
    )


with col4:
    st.metric(
        "Median Employment",
        f"{median_employment_years:.2f} Years"
        if not pd.isna(median_employment_years)
        else "N/A"
    )


with col5:
    st.metric(
        "Payment Difficulty Rate",
        f"{payment_difficulty_rate:.2f}%"
        if not pd.isna(payment_difficulty_rate)
        else "N/A"
    )


st.divider()


# ============================================================
# EMPLOYMENT DURATION ANALYSIS
# ============================================================

st.subheader("📈 Employment Duration Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Employment Histogram
# ------------------------------------------------------------

with col1:

    employment_data = filtered_df.dropna(
        subset=["EMPLOYMENT_YEARS"]
    )

    # Limit extreme values for better visualization
    employment_data = employment_data[
        employment_data["EMPLOYMENT_YEARS"] <=
        employment_data["EMPLOYMENT_YEARS"].quantile(0.99)
    ]

    fig_employment_hist = px.histogram(
        employment_data,
        x="EMPLOYMENT_YEARS",
        nbins=30,
        title="Employment Duration Distribution",
        labels={
            "EMPLOYMENT_YEARS": "Employment Duration (Years)"
        }
    )

    st.plotly_chart(
        fig_employment_hist,
        use_container_width=True
    )


# ------------------------------------------------------------
# Employment Box Plot
# ------------------------------------------------------------

with col2:

    fig_employment_box = px.box(
        employment_data,
        y="EMPLOYMENT_YEARS",
        points=False,
        title="Employment Duration Box Plot",
        labels={
            "EMPLOYMENT_YEARS": "Employment Duration (Years)"
        }
    )

    st.plotly_chart(
        fig_employment_box,
        use_container_width=True
    )


# ============================================================
# EMPLOYMENT VS TARGET
# ============================================================

st.subheader("🎯 Employment Duration by Repayment Status")

if "TARGET_LABEL" in employment_data.columns:

    fig_employment_target = px.box(
        employment_data,
        x="TARGET_LABEL",
        y="EMPLOYMENT_YEARS",
        points=False,
        title="Employment Duration by Repayment Status",
        labels={
            "TARGET_LABEL": "Repayment Status",
            "EMPLOYMENT_YEARS": "Employment Duration (Years)"
        }
    )

    st.plotly_chart(
        fig_employment_target,
        use_container_width=True
    )


# ============================================================
# INCOME TYPE ANALYSIS
# ============================================================

st.subheader("💼 Income Type Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Income Type Count
# ------------------------------------------------------------

with col1:

    income_type_data = (
        filtered_df["NAME_INCOME_TYPE"]
        .value_counts()
        .reset_index()
    )

    income_type_data.columns = [
        "Income Type",
        "Customers"
    ]

    fig_income_type = px.bar(
        income_type_data,
        x="Customers",
        y="Income Type",
        orientation="h",
        text="Customers",
        title="Customers by Income Type"
    )

    st.plotly_chart(
        fig_income_type,
        use_container_width=True
    )


# ------------------------------------------------------------
# Average Employment by Income Type
# ------------------------------------------------------------

with col2:

    employment_by_income_type = (
        filtered_df
        .groupby("NAME_INCOME_TYPE")[
            "EMPLOYMENT_YEARS"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    employment_by_income_type.columns = [
        "Income Type",
        "Average Employment Years"
    ]

    fig_avg_employment = px.bar(
        employment_by_income_type,
        x="Average Employment Years",
        y="Income Type",
        orientation="h",
        text="Average Employment Years",
        title="Average Employment Duration by Income Type"
    )

    st.plotly_chart(
        fig_avg_employment,
        use_container_width=True
    )


# ============================================================
# OCCUPATION ANALYSIS
# ============================================================

st.subheader("👨‍💼 Occupation Analysis")

if "OCCUPATION_TYPE" in filtered_df.columns:

    occupation_data = (
        filtered_df["OCCUPATION_TYPE"]
        .value_counts()
        .head(15)
        .sort_values()
        .reset_index()
    )

    occupation_data.columns = [
        "Occupation",
        "Customers"
    ]

    fig_occupation = px.bar(
        occupation_data,
        x="Customers",
        y="Occupation",
        orientation="h",
        text="Customers",
        title="Top 15 Occupations by Customer Count"
    )

    st.plotly_chart(
        fig_occupation,
        use_container_width=True
    )


# ============================================================
# ORGANIZATION ANALYSIS
# ============================================================

st.subheader("🏢 Organization Analysis")

if "ORGANIZATION_TYPE" in filtered_df.columns:

    organization_data = (
        filtered_df["ORGANIZATION_TYPE"]
        .value_counts()
        .head(15)
        .sort_values()
        .reset_index()
    )

    organization_data.columns = [
        "Organization Type",
        "Customers"
    ]

    fig_organization = px.bar(
        organization_data,
        x="Customers",
        y="Organization Type",
        orientation="h",
        text="Customers",
        title="Top 15 Organization Types"
    )

    st.plotly_chart(
        fig_organization,
        use_container_width=True
    )


# ============================================================
# EMPLOYMENT VS INCOME
# ============================================================

st.subheader("💰 Employment Duration vs Income")

employment_income = filtered_df.dropna(
    subset=[
        "EMPLOYMENT_YEARS",
        "INCOME_LAKHS"
    ]
).copy()

employment_income = employment_income[
    employment_income["INCOME_LAKHS"] > 0
]


if not employment_income.empty:

    income_limit = (
        employment_income["INCOME_LAKHS"]
        .quantile(0.99)
    )

    employment_income = employment_income[
        employment_income["INCOME_LAKHS"] <= income_limit
    ]


fig_income_scatter = px.scatter(
    employment_income,
    x="EMPLOYMENT_YEARS",
    y="INCOME_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in employment_income.columns
    else None,
    opacity=0.55,
    title="Employment Duration vs Annual Income",
    labels={
        "EMPLOYMENT_YEARS": "Employment Duration (Years)",
        "INCOME_LAKHS": "Annual Income (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_income_scatter,
    use_container_width=True
)


# ============================================================
# EMPLOYMENT VS CREDIT
# ============================================================

st.subheader("💳 Employment Duration vs Credit")

employment_credit = filtered_df.dropna(
    subset=[
        "EMPLOYMENT_YEARS",
        "CREDIT_LAKHS"
    ]
).copy()

employment_credit = employment_credit[
    employment_credit["CREDIT_LAKHS"] > 0
]


if not employment_credit.empty:

    credit_limit = (
        employment_credit["CREDIT_LAKHS"]
        .quantile(0.99)
    )

    employment_credit = employment_credit[
        employment_credit["CREDIT_LAKHS"] <= credit_limit
    ]


fig_credit_scatter = px.scatter(
    employment_credit,
    x="EMPLOYMENT_YEARS",
    y="CREDIT_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in employment_credit.columns
    else None,
    opacity=0.55,
    title="Employment Duration vs Credit Amount",
    labels={
        "EMPLOYMENT_YEARS": "Employment Duration (Years)",
        "CREDIT_LAKHS": "Credit Amount (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_credit_scatter,
    use_container_width=True
)


# ============================================================
# AGE VS EMPLOYMENT
# ============================================================

st.subheader("🎂 Age vs Employment Duration")

age_employment = filtered_df.dropna(
    subset=[
        "AGE",
        "EMPLOYMENT_YEARS"
    ]
).copy()

age_employment = age_employment[
    age_employment["AGE"].between(18, 100)
]


fig_age_employment = px.scatter(
    age_employment,
    x="AGE",
    y="EMPLOYMENT_YEARS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in age_employment.columns
    else None,
    opacity=0.55,
    title="Customer Age vs Employment Duration",
    labels={
        "AGE": "Age (Years)",
        "EMPLOYMENT_YEARS": "Employment Duration (Years)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_age_employment,
    use_container_width=True
)


# ============================================================
# EMPLOYMENT GROUP ANALYSIS
# ============================================================

st.subheader("📊 Employment Group Analysis")

group_data = filtered_df.dropna(
    subset=["EMPLOYMENT_GROUP"]
).copy()


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Customer Count by Group
# ------------------------------------------------------------

with col1:

    employment_group_count = (
        group_data["EMPLOYMENT_GROUP"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    employment_group_count.columns = [
        "Employment Group",
        "Customers"
    ]

    fig_group_count = px.bar(
        employment_group_count,
        x="Employment Group",
        y="Customers",
        text="Customers",
        title="Customers by Employment Group"
    )

    st.plotly_chart(
        fig_group_count,
        use_container_width=True
    )


# ------------------------------------------------------------
# Payment Difficulty Rate
# ------------------------------------------------------------

with col2:

    if "TARGET" in group_data.columns:

        employment_group_target = (
            group_data
            .groupby(
                "EMPLOYMENT_GROUP",
                observed=True
            )["TARGET"]
            .mean()
            .mul(100)
            .reset_index()
        )

        employment_group_target.columns = [
            "Employment Group",
            "Payment Difficulty Rate"
        ]

        fig_group_target = px.bar(
            employment_group_target,
            x="Employment Group",
            y="Payment Difficulty Rate",
            text="Payment Difficulty Rate",
            title="Payment Difficulty Rate by Employment Group",
            labels={
                "Payment Difficulty Rate":
                    "Payment Difficulty Rate (%)"
            }
        )

        st.plotly_chart(
            fig_group_target,
            use_container_width=True
        )


# ============================================================
# DATA TABLE
# ============================================================

st.subheader("📋 Employment Analysis Data")


display_columns = [
    "SK_ID_CURR",
    "DAYS_EMPLOYED",
    "EMPLOYMENT_YEARS",
    "EMPLOYMENT_GROUP",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "AGE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
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
# DOWNLOAD FILTERED DATA
# ============================================================

csv_data = (
    filtered_df[available_columns]
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Employment Analysis Data",
    data=csv_data,
    file_name="employment_analysis_filtered.csv",
    mime="text/csv"
)