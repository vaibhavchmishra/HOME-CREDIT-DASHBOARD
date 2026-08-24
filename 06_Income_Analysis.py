import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Income Analysis'); df=require_dataset('application_train')
if 'AMT_INCOME_TOTAL' in df: st.plotly_chart(px.histogram(df,x='AMT_INCOME_TOTAL',color='TARGET' if 'TARGET' in df else None,nbins=50),use_container_width=True)
if 'NAME_INCOME_TYPE' in df and 'TARGET' in df: st.plotly_chart(px.bar(df.groupby('NAME_INCOME_TYPE').TARGET.mean().mul(100).sort_values(ascending=False).reset_index(),x='NAME_INCOME_TYPE',y='TARGET'),use_container_width=True)





# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Income Analysis | Home Credit",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("💰 Income Analysis")

st.markdown(
    """
    Analyze customer income, income sources, occupations,
    loan amounts, annuity, goods price, and repayment difficulty.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = "data/application_train.csv"

    data = pd.read_csv(file_path)

    return data


df = load_data()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Convert DAYS_BIRTH to age
if "DAYS_BIRTH" in df.columns:

    df["AGE"] = (
        -df["DAYS_BIRTH"] / 365.25
    ).round(1)

else:

    df["AGE"] = np.nan


# Convert income to lakhs
if "AMT_INCOME_TOTAL" in df.columns:

    df["INCOME_LAKHS"] = (
        df["AMT_INCOME_TOTAL"] / 100000
    )

else:

    df["INCOME_LAKHS"] = np.nan


# Convert credit amount to lakhs
if "AMT_CREDIT" in df.columns:

    df["CREDIT_LAKHS"] = (
        df["AMT_CREDIT"] / 100000
    )

else:

    df["CREDIT_LAKHS"] = np.nan


# Convert annuity to lakhs
if "AMT_ANNUITY" in df.columns:

    df["ANNUITY_LAKHS"] = (
        df["AMT_ANNUITY"] / 100000
    )

else:

    df["ANNUITY_LAKHS"] = np.nan


# Convert goods price to lakhs
if "AMT_GOODS_PRICE" in df.columns:

    df["GOODS_PRICE_LAKHS"] = (
        df["AMT_GOODS_PRICE"] / 100000
    )

else:

    df["GOODS_PRICE_LAKHS"] = np.nan


# Target labels
if "TARGET" in df.columns:

    df["TARGET_LABEL"] = df["TARGET"].map(
        {
            0: "No Payment Difficulties",
            1: "Payment Difficulties"
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Income Filters")


filtered_df = df.copy()


# Income Type
if "NAME_INCOME_TYPE" in df.columns:

    income_types = sorted(
        df["NAME_INCOME_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_income_type = st.sidebar.multiselect(
        "Income Type",
        income_types,
        default=income_types
    )

    if selected_income_type:

        filtered_df = filtered_df[
            filtered_df["NAME_INCOME_TYPE"]
            .astype(str)
            .isin(selected_income_type)
        ]


# Education
if "NAME_EDUCATION_TYPE" in df.columns:

    education_types = sorted(
        df["NAME_EDUCATION_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_education = st.sidebar.multiselect(
        "Education",
        education_types,
        default=education_types
    )

    if selected_education:

        filtered_df = filtered_df[
            filtered_df["NAME_EDUCATION_TYPE"]
            .astype(str)
            .isin(selected_education)
        ]


# Gender
if "CODE_GENDER" in df.columns:

    genders = sorted(
        df["CODE_GENDER"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        genders,
        default=genders
    )

    if selected_gender:

        filtered_df = filtered_df[
            filtered_df["CODE_GENDER"]
            .astype(str)
            .isin(selected_gender)
        ]


# Target
if "TARGET" in df.columns:

    target_labels = {
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    }

    selected_target = st.sidebar.multiselect(
        "Repayment Status",
        options=list(target_labels.keys()),
        default=list(target_labels.keys()),
        format_func=lambda x: target_labels[x]
    )

    if selected_target:

        filtered_df = filtered_df[
            filtered_df["TARGET"].isin(selected_target)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

average_income = (
    filtered_df["AMT_INCOME_TOTAL"].mean()
    if "AMT_INCOME_TOTAL" in filtered_df.columns
    else np.nan
)

median_income = (
    filtered_df["AMT_INCOME_TOTAL"].median()
    if "AMT_INCOME_TOTAL" in filtered_df.columns
    else np.nan
)

maximum_income = (
    filtered_df["AMT_INCOME_TOTAL"].max()
    if "AMT_INCOME_TOTAL" in filtered_df.columns
    else np.nan
)

average_credit = (
    filtered_df["AMT_CREDIT"].mean()
    if "AMT_CREDIT" in filtered_df.columns
    else np.nan
)

default_rate = (
    filtered_df["TARGET"].mean() * 100
    if "TARGET" in filtered_df.columns
    else np.nan
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Income Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "Average Income",
        f"₹{average_income:,.0f}"
        if not pd.isna(average_income)
        else "N/A"
    )

with col3:

    st.metric(
        "Median Income",
        f"₹{median_income:,.0f}"
        if not pd.isna(median_income)
        else "N/A"
    )

with col4:

    st.metric(
        "Maximum Income",
        f"₹{maximum_income:,.0f}"
        if not pd.isna(maximum_income)
        else "N/A"
    )

with col5:

    st.metric(
        "Payment Difficulty Rate",
        f"{default_rate:.2f}%"
        if not pd.isna(default_rate)
        else "N/A"
    )


st.divider()


# ============================================================
# INCOME DISTRIBUTION
# ============================================================

st.subheader("📈 Income Distribution")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------

with col1:

    income_data = filtered_df[
        filtered_df["INCOME_LAKHS"] > 0
    ].copy()

    # Remove extreme values only for visualization
    if not income_data.empty:

        upper_limit = income_data[
            "INCOME_LAKHS"
        ].quantile(0.99)

        income_data = income_data[
            income_data["INCOME_LAKHS"] <= upper_limit
        ]

    fig_income = px.histogram(
        income_data,
        x="INCOME_LAKHS",
        nbins=40,
        title="Annual Income Distribution",
        labels={
            "INCOME_LAKHS": "Annual Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True
    )


# ------------------------------------------------------------
# Box Plot
# ------------------------------------------------------------

with col2:

    fig_income_box = px.box(
        income_data,
        y="INCOME_LAKHS",
        points=False,
        title="Income Box Plot",
        labels={
            "INCOME_LAKHS": "Annual Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_income_box,
        use_container_width=True
    )


# ============================================================
# INCOME VS TARGET
# ============================================================

st.subheader("🎯 Income by Repayment Status")

if "TARGET_LABEL" in filtered_df.columns:

    target_income = filtered_df[
        filtered_df["INCOME_LAKHS"] > 0
    ].copy()

    if not target_income.empty:

        upper_limit = target_income[
            "INCOME_LAKHS"
        ].quantile(0.99)

        target_income = target_income[
            target_income["INCOME_LAKHS"] <= upper_limit
        ]

    fig_target_income = px.box(
        target_income,
        x="TARGET_LABEL",
        y="INCOME_LAKHS",
        points=False,
        title="Income Distribution by Repayment Status",
        labels={
            "TARGET_LABEL": "Repayment Status",
            "INCOME_LAKHS": "Annual Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_target_income,
        use_container_width=True
    )


# ============================================================
# INCOME TYPE
# ============================================================

st.subheader("💼 Income Type Analysis")

col1, col2 = st.columns(2)


with col1:

    if "NAME_INCOME_TYPE" in filtered_df.columns:

        income_type_count = (
            filtered_df["NAME_INCOME_TYPE"]
            .value_counts()
            .reset_index()
        )

        income_type_count.columns = [
            "Income Type",
            "Customers"
        ]

        fig_income_type = px.bar(
            income_type_count,
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


with col2:

    if "NAME_INCOME_TYPE" in filtered_df.columns:

        income_type_avg = (
            filtered_df
            .groupby("NAME_INCOME_TYPE")[
                "AMT_INCOME_TOTAL"
            ]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        income_type_avg.columns = [
            "Income Type",
            "Average Income"
        ]

        income_type_avg[
            "Average Income Lakhs"
        ] = (
            income_type_avg["Average Income"] / 100000
        )

        fig_income_type_avg = px.bar(
            income_type_avg,
            x="Average Income Lakhs",
            y="Income Type",
            orientation="h",
            text="Average Income Lakhs",
            title="Average Income by Income Type",
            labels={
                "Average Income Lakhs":
                    "Average Income (₹ Lakhs)"
            }
        )

        st.plotly_chart(
            fig_income_type_avg,
            use_container_width=True
        )


# ============================================================
# EDUCATION VS INCOME
# ============================================================

st.subheader("🎓 Education vs Income")

if "NAME_EDUCATION_TYPE" in filtered_df.columns:

    education_income = (
        filtered_df
        .groupby("NAME_EDUCATION_TYPE")[
            "AMT_INCOME_TOTAL"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    education_income[
        "Average Income Lakhs"
    ] = (
        education_income["AMT_INCOME_TOTAL"] / 100000
    )

    fig_education_income = px.bar(
        education_income,
        x="Average Income Lakhs",
        y="NAME_EDUCATION_TYPE",
        orientation="h",
        text="Average Income Lakhs",
        title="Average Income by Education Level",
        labels={
            "NAME_EDUCATION_TYPE": "Education",
            "Average Income Lakhs":
                "Average Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_education_income,
        use_container_width=True
    )


# ============================================================
# OCCUPATION VS INCOME
# ============================================================

st.subheader("👨‍💼 Occupation vs Income")

if "OCCUPATION_TYPE" in filtered_df.columns:

    occupation_income = (
        filtered_df
        .dropna(subset=["OCCUPATION_TYPE"])
        .groupby("OCCUPATION_TYPE")[
            "AMT_INCOME_TOTAL"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    occupation_income[
        "Average Income Lakhs"
    ] = (
        occupation_income["AMT_INCOME_TOTAL"] / 100000
    )

    fig_occupation = px.bar(
        occupation_income,
        x="Average Income Lakhs",
        y="OCCUPATION_TYPE",
        orientation="h",
        text="Average Income Lakhs",
        title="Average Income by Occupation",
        labels={
            "OCCUPATION_TYPE": "Occupation",
            "Average Income Lakhs":
                "Average Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_occupation,
        use_container_width=True
    )


# ============================================================
# INCOME VS CREDIT
# ============================================================

st.subheader("💳 Income vs Credit Amount")

credit_data = filtered_df[
    (filtered_df["INCOME_LAKHS"] > 0) &
    (filtered_df["CREDIT_LAKHS"] > 0)
].copy()


# Limit extreme observations for readability
if not credit_data.empty:

    income_limit = credit_data[
        "INCOME_LAKHS"
    ].quantile(0.99)

    credit_limit = credit_data[
        "CREDIT_LAKHS"
    ].quantile(0.99)

    credit_data = credit_data[
        (credit_data["INCOME_LAKHS"] <= income_limit) &
        (credit_data["CREDIT_LAKHS"] <= credit_limit)
    ]


fig_credit = px.scatter(
    credit_data,
    x="INCOME_LAKHS",
    y="CREDIT_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in credit_data.columns
    else None,
    opacity=0.55,
    title="Annual Income vs Credit Amount",
    labels={
        "INCOME_LAKHS": "Income (₹ Lakhs)",
        "CREDIT_LAKHS": "Credit Amount (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_credit,
    use_container_width=True
)


# ============================================================
# INCOME VS ANNUITY
# ============================================================

st.subheader("💰 Income vs Annuity")

annuity_data = filtered_df[
    (filtered_df["INCOME_LAKHS"] > 0) &
    (filtered_df["ANNUITY_LAKHS"] > 0)
].copy()


if not annuity_data.empty:

    income_limit = annuity_data[
        "INCOME_LAKHS"
    ].quantile(0.99)

    annuity_limit = annuity_data[
        "ANNUITY_LAKHS"
    ].quantile(0.99)

    annuity_data = annuity_data[
        (annuity_data["INCOME_LAKHS"] <= income_limit) &
        (annuity_data["ANNUITY_LAKHS"] <= annuity_limit)
    ]


fig_annuity = px.scatter(
    annuity_data,
    x="INCOME_LAKHS",
    y="ANNUITY_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in annuity_data.columns
    else None,
    opacity=0.55,
    title="Annual Income vs Loan Annuity",
    labels={
        "INCOME_LAKHS": "Income (₹ Lakhs)",
        "ANNUITY_LAKHS": "Annuity (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_annuity,
    use_container_width=True
)


# ============================================================
# INCOME VS GOODS PRICE
# ============================================================

st.subheader("🛒 Income vs Goods Price")

goods_data = filtered_df[
    (filtered_df["INCOME_LAKHS"] > 0) &
    (filtered_df["GOODS_PRICE_LAKHS"] > 0)
].copy()


if not goods_data.empty:

    income_limit = goods_data[
        "INCOME_LAKHS"
    ].quantile(0.99)

    goods_limit = goods_data[
        "GOODS_PRICE_LAKHS"
    ].quantile(0.99)

    goods_data = goods_data[
        (goods_data["INCOME_LAKHS"] <= income_limit) &
        (goods_data["GOODS_PRICE_LAKHS"] <= goods_limit)
    ]


fig_goods = px.scatter(
    goods_data,
    x="INCOME_LAKHS",
    y="GOODS_PRICE_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in goods_data.columns
    else None,
    opacity=0.55,
    title="Annual Income vs Goods Price",
    labels={
        "INCOME_LAKHS": "Income (₹ Lakhs)",
        "GOODS_PRICE_LAKHS": "Goods Price (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_goods,
    use_container_width=True
)


# ============================================================
# INCOME VS AGE
# ============================================================

st.subheader("🎂 Income vs Age")

age_income = filtered_df[
    (filtered_df["AGE"].between(18, 100)) &
    (filtered_df["INCOME_LAKHS"] > 0)
].copy()


if not age_income.empty:

    income_limit = age_income[
        "INCOME_LAKHS"
    ].quantile(0.99)

    age_income = age_income[
        age_income["INCOME_LAKHS"] <= income_limit
    ]


fig_age_income = px.scatter(
    age_income,
    x="AGE",
    y="INCOME_LAKHS",
    color="TARGET_LABEL"
    if "TARGET_LABEL" in age_income.columns
    else None,
    opacity=0.55,
    title="Customer Age vs Annual Income",
    labels={
        "AGE": "Age",
        "INCOME_LAKHS": "Income (₹ Lakhs)",
        "TARGET_LABEL": "Repayment Status"
    }
)

st.plotly_chart(
    fig_age_income,
    use_container_width=True
)


# ============================================================
# INCOME QUANTILE ANALYSIS
# ============================================================

st.subheader("📊 Income Quantile Analysis")

quantile_data = filtered_df[
    filtered_df["AMT_INCOME_TOTAL"] > 0
].copy()


if len(quantile_data) >= 5:

    quantile_data["INCOME_GROUP"] = pd.qcut(
        quantile_data["AMT_INCOME_TOTAL"],
        q=5,
        labels=[
            "Lowest Income",
            "Lower-Middle Income",
            "Middle Income",
            "Upper-Middle Income",
            "Highest Income"
        ],
        duplicates="drop"
    )


    income_group_summary = (
        quantile_data
        .groupby(
            "INCOME_GROUP",
            observed=True
        )
        .agg(
            Customers=("SK_ID_CURR", "count"),
            Average_Income=("AMT_INCOME_TOTAL", "mean"),
            Payment_Difficulty_Rate=("TARGET", "mean")
        )
        .reset_index()
    )


    income_group_summary[
        "Payment_Difficulty_Rate"
    ] *= 100

    income_group_summary[
        "Average_Income_Lakhs"
    ] = (
        income_group_summary["Average_Income"] / 100000
    )


    col1, col2 = st.columns(2)


    with col1:

        fig_quantile_count = px.bar(
            income_group_summary,
            x="INCOME_GROUP",
            y="Customers",
            text="Customers",
            title="Customers by Income Group",
            labels={
                "INCOME_GROUP": "Income Group",
                "Customers": "Customers"
            }
        )

        st.plotly_chart(
            fig_quantile_count,
            use_container_width=True
        )


    with col2:

        fig_quantile_target = px.bar(
            income_group_summary,
            x="INCOME_GROUP",
            y="Payment_Difficulty_Rate",
            text="Payment_Difficulty_Rate",
            title="Payment Difficulty Rate by Income Group",
            labels={
                "INCOME_GROUP": "Income Group",
                "Payment_Difficulty_Rate":
                    "Payment Difficulty Rate (%)"
            }
        )

        st.plotly_chart(
            fig_quantile_target,
            use_container_width=True
        )


# ============================================================
# INCOME DATA TABLE
# ============================================================

st.subheader("📋 Income Analysis Data")


display_columns = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "CODE_GENDER",
    "TARGET"
]


available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
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
    label="⬇️ Download Income Analysis Data",
    data=csv_data,
    file_name="income_analysis_filtered.csv",
    mime="text/csv"
)