import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Family & Housing Analysis'); df=require_dataset('application_train')
for c in ['NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']:
    if c in df and 'TARGET' in df: st.plotly_chart(px.bar(df.groupby(c).TARGET.mean().mul(100).reset_index(),x=c,y='TARGET'),use_container_width=True)




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Family & Housing Analysis | Home Credit",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Family & Housing Analysis")

st.markdown(
    """
    Analyze customer family composition, marital status, children,
    family size, housing type, property ownership, and their
    relationship with repayment difficulties.
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
# Target Label
# ------------------------------------------------------------

if "TARGET" in df.columns:

    df["TARGET_LABEL"] = df["TARGET"].map(
        {
            0: "No Payment Difficulties",
            1: "Payment Difficulties"
        }
    )


# ------------------------------------------------------------
# Property Ownership Label
# ------------------------------------------------------------

if "FLAG_OWN_REALTY" in df.columns:

    df["REALTY_LABEL"] = df[
        "FLAG_OWN_REALTY"
    ].map(
        {
            "Y": "Own Property",
            "N": "Do Not Own Property"
        }
    )


# ------------------------------------------------------------
# Car Ownership Label
# ------------------------------------------------------------

if "FLAG_OWN_CAR" in df.columns:

    df["CAR_LABEL"] = df[
        "FLAG_OWN_CAR"
    ].map(
        {
            "Y": "Own Car",
            "N": "Do Not Own Car"
        }
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Family & Housing Filters")

filtered_df = df.copy()


# ------------------------------------------------------------
# Family Status
# ------------------------------------------------------------

if "NAME_FAMILY_STATUS" in df.columns:

    family_status_values = sorted(
        df["NAME_FAMILY_STATUS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_family_status = st.sidebar.multiselect(
        "Family Status",
        family_status_values,
        default=family_status_values
    )

    if selected_family_status:

        filtered_df = filtered_df[
            filtered_df["NAME_FAMILY_STATUS"]
            .astype(str)
            .isin(selected_family_status)
        ]


# ------------------------------------------------------------
# Housing Type
# ------------------------------------------------------------

if "NAME_HOUSING_TYPE" in df.columns:

    housing_values = sorted(
        df["NAME_HOUSING_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
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


# ------------------------------------------------------------
# Property Ownership
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Car Ownership
# ------------------------------------------------------------

if "FLAG_OWN_CAR" in df.columns:

    car_options = ["Y", "N"]

    selected_car = st.sidebar.multiselect(
        "Car Ownership",
        car_options,
        default=car_options,
        format_func=lambda x:
            "Own Car" if x == "Y"
            else "Do Not Own Car"
    )

    if selected_car:

        filtered_df = filtered_df[
            filtered_df["FLAG_OWN_CAR"]
            .isin(selected_car)
        ]


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

if "TARGET" in df.columns:

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

average_children = (
    filtered_df["CNT_CHILDREN"].mean()
    if "CNT_CHILDREN" in filtered_df.columns
    else np.nan
)

average_family_members = (
    filtered_df["CNT_FAM_MEMBERS"].mean()
    if "CNT_FAM_MEMBERS" in filtered_df.columns
    else np.nan
)

property_ownership_rate = (
    filtered_df["FLAG_OWN_REALTY"]
    .eq("Y")
    .mean() * 100
    if "FLAG_OWN_REALTY" in filtered_df.columns
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

st.subheader("📊 Family & Housing Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "Avg. Children",
        f"{average_children:.2f}"
        if not pd.isna(average_children)
        else "N/A"
    )

with col3:

    st.metric(
        "Avg. Family Members",
        f"{average_family_members:.2f}"
        if not pd.isna(average_family_members)
        else "N/A"
    )

with col4:

    st.metric(
        "Property Ownership",
        f"{property_ownership_rate:.2f}%"
        if not pd.isna(property_ownership_rate)
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
# FAMILY STATUS
# ============================================================

st.subheader("👨‍👩‍👧 Family Status Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Family Status Distribution
# ------------------------------------------------------------

with col1:

    family_status = (
        filtered_df["NAME_FAMILY_STATUS"]
        .value_counts()
        .reset_index()
    )

    family_status.columns = [
        "Family Status",
        "Customers"
    ]

    fig_family = px.bar(
        family_status,
        x="Customers",
        y="Family Status",
        orientation="h",
        text="Customers",
        title="Customer Distribution by Family Status"
    )

    st.plotly_chart(
        fig_family,
        use_container_width=True
    )


# ------------------------------------------------------------
# Family Status vs Target
# ------------------------------------------------------------

with col2:

    family_target = (
        filtered_df
        .groupby(
            ["NAME_FAMILY_STATUS", "TARGET_LABEL"],
            dropna=False
        )
        .size()
        .reset_index(name="Customers")
    )

    fig_family_target = px.bar(
        family_target,
        x="NAME_FAMILY_STATUS",
        y="Customers",
        color="TARGET_LABEL",
        barmode="stack",
        title="Family Status by Repayment Status",
        labels={
            "NAME_FAMILY_STATUS": "Family Status",
            "TARGET_LABEL": "Repayment Status"
        }
    )

    st.plotly_chart(
        fig_family_target,
        use_container_width=True
    )


# ============================================================
# CHILDREN ANALYSIS
# ============================================================

st.subheader("👶 Children Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Children Distribution
# ------------------------------------------------------------

with col1:

    children_data = (
        filtered_df["CNT_CHILDREN"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    children_data.columns = [
        "Children",
        "Customers"
    ]

    # Keep visualization readable
    children_data = children_data[
        children_data["Children"] <= 10
    ]

    fig_children = px.bar(
        children_data,
        x="Children",
        y="Customers",
        text="Customers",
        title="Customers by Number of Children"
    )

    st.plotly_chart(
        fig_children,
        use_container_width=True
    )


# ------------------------------------------------------------
# Children vs Payment Difficulty
# ------------------------------------------------------------

with col2:

    children_target = (
        filtered_df
        .groupby("CNT_CHILDREN")["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    children_target.columns = [
        "Children",
        "Payment Difficulty Rate"
    ]

    children_target = children_target[
        children_target["Children"] <= 10
    ]

    fig_children_target = px.bar(
        children_target,
        x="Children",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment Difficulty Rate by Number of Children",
        labels={
            "Payment Difficulty Rate":
                "Payment Difficulty Rate (%)"
        }
    )

    st.plotly_chart(
        fig_children_target,
        use_container_width=True
    )


# ============================================================
# FAMILY SIZE
# ============================================================

st.subheader("👪 Family Size Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Family Size Distribution
# ------------------------------------------------------------

with col1:

    family_size_data = filtered_df[
        filtered_df["CNT_FAM_MEMBERS"].between(
            1, 10
        )
    ]

    fig_family_size = px.histogram(
        family_size_data,
        x="CNT_FAM_MEMBERS",
        nbins=10,
        title="Family Members Distribution",
        labels={
            "CNT_FAM_MEMBERS":
                "Number of Family Members"
        }
    )

    st.plotly_chart(
        fig_family_size,
        use_container_width=True
    )


# ------------------------------------------------------------
# Family Size vs Target
# ------------------------------------------------------------

with col2:

    family_size_target = (
        filtered_df[
            filtered_df["CNT_FAM_MEMBERS"].between(
                1, 10
            )
        ]
        .groupby("CNT_FAM_MEMBERS")["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
    )

    family_size_target.columns = [
        "Family Members",
        "Payment Difficulty Rate"
    ]

    fig_family_size_target = px.bar(
        family_size_target,
        x="Family Members",
        y="Payment Difficulty Rate",
        text="Payment Difficulty Rate",
        title="Payment Difficulty Rate by Family Size",
        labels={
            "Payment Difficulty Rate":
                "Payment Difficulty Rate (%)"
        }
    )

    st.plotly_chart(
        fig_family_size_target,
        use_container_width=True
    )


# ============================================================
# HOUSING TYPE
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
        "Customers"
    ]

    fig_housing = px.bar(
        housing_data,
        x="Customers",
        y="Housing Type",
        orientation="h",
        text="Customers",
        title="Customer Distribution by Housing Type"
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
        .reset_index(name="Customers")
    )

    fig_housing_target = px.bar(
        housing_target,
        x="NAME_HOUSING_TYPE",
        y="Customers",
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
# OWNERSHIP ANALYSIS
# ============================================================

st.subheader("🚗🏡 Ownership Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Property Ownership
# ------------------------------------------------------------

with col1:

    realty_data = (
        filtered_df["REALTY_LABEL"]
        .value_counts()
        .reset_index()
    )

    realty_data.columns = [
        "Ownership",
        "Customers"
    ]

    fig_realty = px.pie(
        realty_data,
        names="Ownership",
        values="Customers",
        hole=0.45,
        title="Property Ownership"
    )

    st.plotly_chart(
        fig_realty,
        use_container_width=True
    )


# ------------------------------------------------------------
# Car Ownership
# ------------------------------------------------------------

with col2:

    car_data = (
        filtered_df["CAR_LABEL"]
        .value_counts()
        .reset_index()
    )

    car_data.columns = [
        "Ownership",
        "Customers"
    ]

    fig_car = px.pie(
        car_data,
        names="Ownership",
        values="Customers",
        hole=0.45,
        title="Car Ownership"
    )

    st.plotly_chart(
        fig_car,
        use_container_width=True
    )


# ============================================================
# FAMILY STATUS VS INCOME
# ============================================================

st.subheader("💰 Family & Income Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Family Status vs Income
# ------------------------------------------------------------

with col1:

    family_income = (
        filtered_df
        .groupby("NAME_FAMILY_STATUS")[
            "AMT_INCOME_TOTAL"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    family_income[
        "Average Income Lakhs"
    ] = (
        family_income["AMT_INCOME_TOTAL"] / 100000
    )

    fig_family_income = px.bar(
        family_income,
        x="Average Income Lakhs",
        y="NAME_FAMILY_STATUS",
        orientation="h",
        text="Average Income Lakhs",
        title="Average Income by Family Status",
        labels={
            "NAME_FAMILY_STATUS": "Family Status",
            "Average Income Lakhs":
                "Average Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_family_income,
        use_container_width=True
    )


# ------------------------------------------------------------
# Housing Type vs Income
# ------------------------------------------------------------

with col2:

    housing_income = (
        filtered_df
        .groupby("NAME_HOUSING_TYPE")[
            "AMT_INCOME_TOTAL"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    housing_income[
        "Average Income Lakhs"
    ] = (
        housing_income["AMT_INCOME_TOTAL"] / 100000
    )

    fig_housing_income = px.bar(
        housing_income,
        x="Average Income Lakhs",
        y="NAME_HOUSING_TYPE",
        orientation="h",
        text="Average Income Lakhs",
        title="Average Income by Housing Type",
        labels={
            "NAME_HOUSING_TYPE": "Housing Type",
            "Average Income Lakhs":
                "Average Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_housing_income,
        use_container_width=True
    )


# ============================================================
# CHILDREN VS FAMILY MEMBERS
# ============================================================

st.subheader("📈 Children vs Family Members")

scatter_data = filtered_df[
    [
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS"
    ]
].dropna()

scatter_data = scatter_data[
    scatter_data["CNT_CHILDREN"] <= 10
]

scatter_data = scatter_data[
    scatter_data["CNT_FAM_MEMBERS"] <= 12
]


fig_family_scatter = px.scatter(
    scatter_data,
    x="CNT_CHILDREN",
    y="CNT_FAM_MEMBERS",
    opacity=0.55,
    title="Number of Children vs Family Members",
    labels={
        "CNT_CHILDREN": "Number of Children",
        "CNT_FAM_MEMBERS":
            "Number of Family Members"
    }
)

st.plotly_chart(
    fig_family_scatter,
    use_container_width=True
)


# ============================================================
# FAMILY / HOUSING DATA TABLE
# ============================================================

st.subheader("📋 Family & Housing Data")

display_columns = [
    "SK_ID_CURR",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_REALTY",
    "FLAG_OWN_CAR",
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
# DOWNLOAD
# ============================================================

csv_data = (
    filtered_df[available_columns]
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Family & Housing Data",
    data=csv_data,
    file_name="family_housing_analysis_filtered.csv",
    mime="text/csv"
)