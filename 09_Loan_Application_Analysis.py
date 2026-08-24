import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Loan Application Analysis'); df=require_dataset('application_train')
for x,y in [('AMT_CREDIT','AMT_ANNUITY'),('AMT_GOODS_PRICE','AMT_CREDIT')]:
    if x in df and y in df: st.plotly_chart(px.scatter(df,x=x,y=y,color='TARGET' if 'TARGET' in df else None,opacity=.5),use_container_width=True)



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Application & Housing Analysis",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Application & Housing Analysis")

st.markdown("""
Analyze loan applications, credit amounts, annuity, goods prices,
contract types, housing characteristics, property ownership,
and payment difficulties.
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
# Income in Lakhs
# ------------------------------------------------------------

if "AMT_INCOME_TOTAL" in df.columns:

    df["INCOME_LAKHS"] = (
        df["AMT_INCOME_TOTAL"] / 100000
    )


# ------------------------------------------------------------
# Credit in Lakhs
# ------------------------------------------------------------

if "AMT_CREDIT" in df.columns:

    df["CREDIT_LAKHS"] = (
        df["AMT_CREDIT"] / 100000
    )


# ------------------------------------------------------------
# Annuity in Lakhs
# ------------------------------------------------------------

if "AMT_ANNUITY" in df.columns:

    df["ANNUITY_LAKHS"] = (
        df["AMT_ANNUITY"] / 100000
    )


# ------------------------------------------------------------
# Goods Price in Lakhs
# ------------------------------------------------------------

if "AMT_GOODS_PRICE" in df.columns:

    df["GOODS_PRICE_LAKHS"] = (
        df["AMT_GOODS_PRICE"] / 100000
    )


# ------------------------------------------------------------
# TARGET Label
# ------------------------------------------------------------

if "TARGET" in df.columns:

    df["TARGET_LABEL"] = df["TARGET"].map({
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    })


# ------------------------------------------------------------
# Property Label
# ------------------------------------------------------------

if "FLAG_OWN_REALTY" in df.columns:

    df["REALTY_LABEL"] = df[
        "FLAG_OWN_REALTY"
    ].map({
        "Y": "Own Property",
        "N": "Do Not Own Property"
    })


# ------------------------------------------------------------
# Credit Groups
# ------------------------------------------------------------

if "CREDIT_LAKHS" in df.columns:

    df["CREDIT_GROUP"] = pd.cut(
        df["CREDIT_LAKHS"],
        bins=[
            -np.inf,
            2.5,
            5,
            10,
            20,
            50,
            np.inf
        ],
        labels=[
            "< ₹2.5L",
            "₹2.5L–₹5L",
            "₹5L–₹10L",
            "₹10L–₹20L",
            "₹20L–₹50L",
            "> ₹50L"
        ]
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Loan & Housing Filters")

filtered_df = df.copy()


# ============================================================
# CONTRACT TYPE FILTER
# ============================================================

if "NAME_CONTRACT_TYPE" in df.columns:

    contract_types = sorted(
        df["NAME_CONTRACT_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
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
# HOUSING TYPE FILTER
# ============================================================

if "NAME_HOUSING_TYPE" in df.columns:

    housing_types = sorted(
        df["NAME_HOUSING_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_housing = st.sidebar.multiselect(
        "Housing Type",
        housing_types,
        default=housing_types
    )

    if selected_housing:

        filtered_df = filtered_df[
            filtered_df["NAME_HOUSING_TYPE"]
            .astype(str)
            .isin(selected_housing)
        ]


# ============================================================
# PROPERTY FILTER
# ============================================================

if "FLAG_OWN_REALTY" in df.columns:

    property_options = ["Y", "N"]

    selected_property = st.sidebar.multiselect(
        "Property Ownership",
        property_options,
        default=property_options,
        format_func=lambda x:
            "Own Property" if x == "Y"
            else "Do Not Own Property"
    )

    if selected_property:

        filtered_df = filtered_df[
            filtered_df["FLAG_OWN_REALTY"]
            .isin(selected_property)
        ]


# ============================================================
# TARGET FILTER
# ============================================================

if "TARGET" in df.columns:

    target_options = {
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    }

    selected_target = st.sidebar.multiselect(
        "Repayment Status",
        options=list(target_options.keys()),
        default=list(target_options.keys()),
        format_func=lambda x:
            target_options[x]
    )

    if selected_target:

        filtered_df = filtered_df[
            filtered_df["TARGET"]
            .isin(selected_target)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_applications = len(filtered_df)

average_credit = (
    filtered_df["AMT_CREDIT"].mean()
    if "AMT_CREDIT" in filtered_df.columns
    else np.nan
)

average_annuity = (
    filtered_df["AMT_ANNUITY"].mean()
    if "AMT_ANNUITY" in filtered_df.columns
    else np.nan
)

average_goods_price = (
    filtered_df["AMT_GOODS_PRICE"].mean()
    if "AMT_GOODS_PRICE" in filtered_df.columns
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

st.subheader("📊 Loan Application Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Applications",
        f"{total_applications:,}"
    )

with col2:

    st.metric(
        "Avg Credit",
        f"₹{average_credit:,.0f}"
        if not pd.isna(average_credit)
        else "N/A"
    )

with col3:

    st.metric(
        "Avg Annuity",
        f"₹{average_annuity:,.0f}"
        if not pd.isna(average_annuity)
        else "N/A"
    )

with col4:

    st.metric(
        "Avg Goods Price",
        f"₹{average_goods_price:,.0f}"
        if not pd.isna(average_goods_price)
        else "N/A"
    )

with col5:

    st.metric(
        "Payment Difficulty",
        f"{payment_difficulty_rate:.2f}%"
        if not pd.isna(payment_difficulty_rate)
        else "N/A"
    )


st.divider()


# ============================================================
# CONTRACT TYPE
# ============================================================

st.subheader("📄 Loan Contract Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Contract Type Distribution
# ------------------------------------------------------------

with col1:

    contract_data = (
        filtered_df["NAME_CONTRACT_TYPE"]
        .value_counts()
        .reset_index()
    )

    contract_data.columns = [
        "Contract Type",
        "Applications"
    ]

    fig_contract = px.pie(
        contract_data,
        names="Contract Type",
        values="Applications",
        hole=0.45,
        title="Loan Contract Type Distribution"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )


# ------------------------------------------------------------
# Contract Type vs TARGET
# ------------------------------------------------------------

with col2:

    contract_target = (
        filtered_df
        .groupby(
            ["NAME_CONTRACT_TYPE", "TARGET_LABEL"],
            dropna=False
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
        title="Contract Type by Repayment Status",
        labels={
            "NAME_CONTRACT_TYPE": "Contract Type",
            "TARGET_LABEL": "Repayment Status"
        }
    )

    st.plotly_chart(
        fig_contract_target,
        use_container_width=True
    )


# ============================================================
# CREDIT DISTRIBUTION
# ============================================================

st.subheader("💳 Credit Amount Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit Histogram
# ------------------------------------------------------------

with col1:

    credit_data = filtered_df[
        filtered_df["CREDIT_LAKHS"] > 0
    ].copy()

    if not credit_data.empty:

        credit_limit = credit_data[
            "CREDIT_LAKHS"
        ].quantile(0.99)

        credit_data = credit_data[
            credit_data["CREDIT_LAKHS"] <= credit_limit
        ]

    fig_credit_hist = px.histogram(
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
        fig_credit_hist,
        use_container_width=True
    )


# ------------------------------------------------------------
# Credit Box Plot
# ------------------------------------------------------------

with col2:

    fig_credit_box = px.box(
        credit_data,
        y="CREDIT_LAKHS",
        points=False,
        title="Credit Amount Box Plot",
        labels={
            "CREDIT_LAKHS":
                "Credit Amount (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_credit_box,
        use_container_width=True
    )


# ============================================================
# ANNUITY & GOODS PRICE
# ============================================================

st.subheader("💰 Annuity & Goods Price Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Annuity
# ------------------------------------------------------------

with col1:

    annuity_data = filtered_df[
        filtered_df["ANNUITY_LAKHS"] > 0
    ]

    fig_annuity = px.histogram(
        annuity_data,
        x="ANNUITY_LAKHS",
        nbins=40,
        title="Loan Annuity Distribution",
        labels={
            "ANNUITY_LAKHS":
                "Annuity (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_annuity,
        use_container_width=True
    )


# ------------------------------------------------------------
# Goods Price
# ------------------------------------------------------------

with col2:

    goods_data = filtered_df[
        filtered_df["GOODS_PRICE_LAKHS"] > 0
    ]

    fig_goods = px.histogram(
        goods_data,
        x="GOODS_PRICE_LAKHS",
        nbins=40,
        title="Goods Price Distribution",
        labels={
            "GOODS_PRICE_LAKHS":
                "Goods Price (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_goods,
        use_container_width=True
    )


# ============================================================
# CREDIT VS INCOME
# ============================================================

st.subheader("📈 Credit vs Income")

scatter_credit_income = filtered_df[
    [
        "INCOME_LAKHS",
        "CREDIT_LAKHS",
        "TARGET_LABEL"
    ]
].dropna()

scatter_credit_income = scatter_credit_income[
    (scatter_credit_income["INCOME_LAKHS"] > 0) &
    (scatter_credit_income["CREDIT_LAKHS"] > 0)
]


# Remove extreme values for visualization

if not scatter_credit_income.empty:

    income_limit = (
        scatter_credit_income[
            "INCOME_LAKHS"
        ].quantile(0.99)
    )

    credit_limit = (
        scatter_credit_income[
            "CREDIT_LAKHS"
        ].quantile(0.99)
    )

    scatter_credit_income = (
        scatter_credit_income[
            (scatter_credit_income["INCOME_LAKHS"]
             <= income_limit)
            &
            (scatter_credit_income["CREDIT_LAKHS"]
             <= credit_limit)
        ]
    )


fig_credit_income = px.scatter(
    scatter_credit_income,
    x="INCOME_LAKHS",
    y="CREDIT_LAKHS",
    color="TARGET_LABEL",
    opacity=0.55,
    title="Credit Amount vs Annual Income",
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
    fig_credit_income,
    use_container_width=True
)


# ============================================================
# CREDIT VS ANNUITY
# ============================================================

st.subheader("💳 Credit Amount vs Annuity")

credit_annuity = filtered_df[
    [
        "CREDIT_LAKHS",
        "ANNUITY_LAKHS",
        "TARGET_LABEL"
    ]
].dropna()

credit_annuity = credit_annuity[
    (credit_annuity["CREDIT_LAKHS"] > 0) &
    (credit_annuity["ANNUITY_LAKHS"] > 0)
]


fig_credit_annuity = px.scatter(
    credit_annuity,
    x="CREDIT_LAKHS",
    y="ANNUITY_LAKHS",
    color="TARGET_LABEL",
    opacity=0.55,
    title="Credit Amount vs Loan Annuity",
    labels={
        "CREDIT_LAKHS":
            "Credit Amount (₹ Lakhs)",
        "ANNUITY_LAKHS":
            "Annuity (₹ Lakhs)",
        "TARGET_LABEL":
            "Repayment Status"
    }
)

st.plotly_chart(
    fig_credit_annuity,
    use_container_width=True
)


# ============================================================
# HOUSING ANALYSIS
# ============================================================

st.subheader("🏠 Housing Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Housing Distribution
# ------------------------------------------------------------

with col1:

    housing_data = (
        filtered_df["NAME_HOUSING_TYPE"]
        .value_counts()
        .reset_index()
    )

    housing_data.columns = [
        "Housing Type",
        "Applications"
    ]

    fig_housing = px.bar(
        housing_data,
        x="Applications",
        y="Housing Type",
        orientation="h",
        text="Applications",
        title="Loan Applicants by Housing Type"
    )

    st.plotly_chart(
        fig_housing,
        use_container_width=True
    )


# ------------------------------------------------------------
# Housing vs Target
# ------------------------------------------------------------

with col2:

    housing_target = (
        filtered_df
        .groupby(
            ["NAME_HOUSING_TYPE", "TARGET_LABEL"],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    fig_housing_target = px.bar(
        housing_target,
        x="NAME_HOUSING_TYPE",
        y="Applications",
        color="TARGET_LABEL",
        barmode="stack",
        title="Housing Type by Repayment Status",
        labels={
            "NAME_HOUSING_TYPE": "Housing Type",
            "TARGET_LABEL": "Repayment Status"
        }
    )

    st.plotly_chart(
        fig_housing_target,
        use_container_width=True
    )


# ============================================================
# CREDIT BY HOUSING
# ============================================================

st.subheader("💳 Housing Type vs Loan Amount")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Average Credit by Housing
# ------------------------------------------------------------

with col1:

    housing_credit = (
        filtered_df
        .groupby("NAME_HOUSING_TYPE")[
            "CREDIT_LAKHS"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    housing_credit.columns = [
        "Housing Type",
        "Average Credit"
    ]

    fig_housing_credit = px.bar(
        housing_credit,
        x="Average Credit",
        y="Housing Type",
        orientation="h",
        text="Average Credit",
        title="Average Credit by Housing Type",
        labels={
            "Average Credit":
                "Average Credit (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_housing_credit,
        use_container_width=True
    )


# ------------------------------------------------------------
# Average Income by Housing
# ------------------------------------------------------------

with col2:

    housing_income = (
        filtered_df
        .groupby("NAME_HOUSING_TYPE")[
            "INCOME_LAKHS"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    housing_income.columns = [
        "Housing Type",
        "Average Income"
    ]

    fig_housing_income = px.bar(
        housing_income,
        x="Average Income",
        y="Housing Type",
        orientation="h",
        text="Average Income",
        title="Average Income by Housing Type",
        labels={
            "Average Income":
                "Average Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_housing_income,
        use_container_width=True
    )


# ============================================================
# PROPERTY OWNERSHIP
# ============================================================

st.subheader("🏡 Property Ownership Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Property Distribution
# ------------------------------------------------------------

with col1:

    property_data = (
        filtered_df["REALTY_LABEL"]
        .value_counts()
        .reset_index()
    )

    property_data.columns = [
        "Property Ownership",
        "Applications"
    ]

    fig_property = px.pie(
        property_data,
        names="Property Ownership",
        values="Applications",
        hole=0.45,
        title="Property Ownership"
    )

    st.plotly_chart(
        fig_property,
        use_container_width=True
    )


# ------------------------------------------------------------
# Property vs Target
# ------------------------------------------------------------

with col2:

    property_target = (
        filtered_df
        .groupby(
            ["REALTY_LABEL", "TARGET_LABEL"],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    fig_property_target = px.bar(
        property_target,
        x="REALTY_LABEL",
        y="Applications",
        color="TARGET_LABEL",
        barmode="stack",
        title="Property Ownership by Repayment Status",
        labels={
            "REALTY_LABEL":
                "Property Ownership",
            "TARGET_LABEL":
                "Repayment Status"
        }
    )

    st.plotly_chart(
        fig_property_target,
        use_container_width=True
    )


# ============================================================
# HOUSING VS CONTRACT
# ============================================================

st.subheader("🏠 Housing Type vs Contract Type")

housing_contract = (
    filtered_df
    .groupby(
        [
            "NAME_HOUSING_TYPE",
            "NAME_CONTRACT_TYPE"
        ],
        dropna=False
    )
    .size()
    .reset_index(name="Applications")
)


fig_housing_contract = px.bar(
    housing_contract,
    x="NAME_HOUSING_TYPE",
    y="Applications",
    color="NAME_CONTRACT_TYPE",
    barmode="stack",
    title="Loan Contract Type by Housing Type",
    labels={
        "NAME_HOUSING_TYPE":
            "Housing Type",
        "NAME_CONTRACT_TYPE":
            "Contract Type"
    }
)

st.plotly_chart(
    fig_housing_contract,
    use_container_width=True
)


# ============================================================
# RISK BY CREDIT GROUP
# ============================================================

st.subheader("⚠️ Payment Difficulty by Credit Group")

credit_group_target = (
    filtered_df
    .dropna(subset=["CREDIT_GROUP"])
    .groupby(
        "CREDIT_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .mul(100)
    .reset_index()
)

credit_group_target.columns = [
    "Credit Group",
    "Payment Difficulty Rate"
]


fig_credit_risk = px.bar(
    credit_group_target,
    x="Credit Group",
    y="Payment Difficulty Rate",
    text="Payment Difficulty Rate",
    title="Payment Difficulty Rate by Credit Amount",
    labels={
        "Payment Difficulty Rate":
            "Payment Difficulty Rate (%)"
    }
)

st.plotly_chart(
    fig_credit_risk,
    use_container_width=True
)


# ============================================================
# DATA TABLE
# ============================================================

st.subheader("📋 Loan Application & Housing Data")

display_columns = [
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_REALTY",
    "FLAG_OWN_CAR",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
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
    label="⬇️ Download Loan & Housing Data",
    data=csv_data,
    file_name="loan_application_housing_filtered.csv",
    mime="text/csv"
)