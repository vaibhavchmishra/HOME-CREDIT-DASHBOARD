import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
st.title('Customer Demographics'); df=require_dataset('application_train')
for c in ['CODE_GENDER','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS']:
    if c in df and 'TARGET' in df: st.plotly_chart(px.bar(df.groupby(c).TARGET.mean().mul(100).sort_values(ascending=False).reset_index(),x=c,y='TARGET',title=f'Default Rate by {c}'),use_container_width=True)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Demographics | Home Credit",
    page_icon="👥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👥 Customer Demographics Analysis")
st.markdown(
    """
    Analyze customer demographic characteristics such as **age, gender,
    education, family status, income, children, housing, car ownership,
    and property ownership**.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    file_path = "data/application_train.csv"

    df = pd.read_csv(file_path)

    return df


df = load_data()


# ============================================================
# CREATE DEMOGRAPHIC FEATURES
# ============================================================

if "DAYS_BIRTH" in df.columns:

    # DAYS_BIRTH is stored as negative number of days
    df["AGE"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

else:

    df["AGE"] = np.nan


# Income in Lakhs for easier visualization
if "AMT_INCOME_TOTAL" in df.columns:

    df["INCOME_LAKHS"] = df["AMT_INCOME_TOTAL"] / 100000

else:

    df["INCOME_LAKHS"] = np.nan


# Target labels
if "TARGET" in df.columns:

    df["TARGET_LABEL"] = df["TARGET"].map({
        0: "No Payment Difficulties",
        1: "Payment Difficulties"
    })


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Demographic Filters")


filtered_df = df.copy()


# Gender filter
if "CODE_GENDER" in df.columns:

    gender_values = sorted(
        df["CODE_GENDER"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        gender_values,
        default=gender_values
    )

    if selected_gender:
        filtered_df = filtered_df[
            filtered_df["CODE_GENDER"].astype(str).isin(selected_gender)
        ]


# Education filter
if "NAME_EDUCATION_TYPE" in df.columns:

    education_values = sorted(
        df["NAME_EDUCATION_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
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


# Family status filter
if "NAME_FAMILY_STATUS" in df.columns:

    family_values = sorted(
        df["NAME_FAMILY_STATUS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_family = st.sidebar.multiselect(
        "Family Status",
        family_values,
        default=family_values
    )

    if selected_family:
        filtered_df = filtered_df[
            filtered_df["NAME_FAMILY_STATUS"]
            .astype(str)
            .isin(selected_family)
        ]


# Target filter
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
            filtered_df["TARGET"].isin(selected_target)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

avg_age = (
    filtered_df["AGE"].mean()
    if "AGE" in filtered_df.columns
    else np.nan
)

avg_income = (
    filtered_df["AMT_INCOME_TOTAL"].mean()
    if "AMT_INCOME_TOTAL" in filtered_df.columns
    else np.nan
)

avg_children = (
    filtered_df["CNT_CHILDREN"].mean()
    if "CNT_CHILDREN" in filtered_df.columns
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

st.subheader("📊 Customer Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Average Age",
        f"{avg_age:.1f} years"
        if not pd.isna(avg_age)
        else "N/A"
    )

with col3:
    st.metric(
        "Average Income",
        f"₹{avg_income:,.0f}"
        if not pd.isna(avg_income)
        else "N/A"
    )

with col4:
    st.metric(
        "Avg. Children",
        f"{avg_children:.2f}"
        if not pd.isna(avg_children)
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
# GENDER DISTRIBUTION
# ============================================================

st.subheader("👨‍👩‍👧 Gender Distribution")

col1, col2 = st.columns(2)


with col1:

    if "CODE_GENDER" in filtered_df.columns:

        gender_df = (
            filtered_df["CODE_GENDER"]
            .value_counts()
            .reset_index()
        )

        gender_df.columns = ["Gender", "Customers"]

        fig_gender = px.bar(
            gender_df,
            x="Gender",
            y="Customers",
            text="Customers",
            title="Customer Distribution by Gender"
        )

        fig_gender.update_traces(
            textposition="outside"
        )

        fig_gender.update_layout(
            xaxis_title="Gender",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            fig_gender,
            use_container_width=True
        )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

with col2:

    if "TARGET_LABEL" in filtered_df.columns:

        target_df = (
            filtered_df["TARGET_LABEL"]
            .value_counts()
            .reset_index()
        )

        target_df.columns = ["Status", "Customers"]

        fig_target = px.bar(
            target_df,
            x="Status",
            y="Customers",
            text="Customers",
            title="Repayment Status Distribution"
        )

        fig_target.update_traces(
            textposition="outside"
        )

        fig_target.update_layout(
            xaxis_title="Repayment Status",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            fig_target,
            use_container_width=True
        )


# ============================================================
# AGE ANALYSIS
# ============================================================

st.subheader("🎂 Age Analysis")

col1, col2 = st.columns(2)


with col1:

    age_data = filtered_df[
        filtered_df["AGE"].between(18, 100)
    ]

    fig_age = px.histogram(
        age_data,
        x="AGE",
        nbins=30,
        title="Customer Age Distribution",
        labels={
            "AGE": "Age",
            "count": "Customers"
        }
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    if "TARGET_LABEL" in filtered_df.columns:

        age_target = filtered_df[
            filtered_df["AGE"].between(18, 100)
        ].dropna(
            subset=["TARGET_LABEL"]
        )

        fig_age_target = px.box(
            age_target,
            x="TARGET_LABEL",
            y="AGE",
            points=False,
            title="Age Distribution by Repayment Status",
            labels={
                "TARGET_LABEL": "Repayment Status",
                "AGE": "Age"
            }
        )

        st.plotly_chart(
            fig_age_target,
            use_container_width=True
        )


# ============================================================
# EDUCATION ANALYSIS
# ============================================================

st.subheader("🎓 Education Analysis")

if "NAME_EDUCATION_TYPE" in filtered_df.columns:

    education_df = (
        filtered_df["NAME_EDUCATION_TYPE"]
        .value_counts()
        .reset_index()
    )

    education_df.columns = [
        "Education",
        "Customers"
    ]

    fig_education = px.bar(
        education_df,
        x="Customers",
        y="Education",
        orientation="h",
        text="Customers",
        title="Customers by Education Level"
    )

    fig_education.update_layout(
        yaxis_title="Education",
        xaxis_title="Number of Customers"
    )

    st.plotly_chart(
        fig_education,
        use_container_width=True
    )


# ============================================================
# FAMILY STATUS
# ============================================================

st.subheader("👪 Family Status Analysis")

col1, col2 = st.columns(2)


with col1:

    if "NAME_FAMILY_STATUS" in filtered_df.columns:

        family_df = (
            filtered_df["NAME_FAMILY_STATUS"]
            .value_counts()
            .reset_index()
        )

        family_df.columns = [
            "Family Status",
            "Customers"
        ]

        fig_family = px.bar(
            family_df,
            x="Family Status",
            y="Customers",
            text="Customers",
            title="Customer Distribution by Family Status"
        )

        fig_family.update_layout(
            xaxis_title="Family Status",
            yaxis_title="Customers"
        )

        st.plotly_chart(
            fig_family,
            use_container_width=True
        )


# ============================================================
# INCOME ANALYSIS
# ============================================================

with col2:

    if "INCOME_LAKHS" in filtered_df.columns:

        income_data = filtered_df[
            filtered_df["INCOME_LAKHS"] >= 0
        ]

        # Remove extreme values only for visualization
        income_upper = income_data["INCOME_LAKHS"].quantile(0.99)

        income_data = income_data[
            income_data["INCOME_LAKHS"] <= income_upper
        ]

        fig_income = px.histogram(
            income_data,
            x="INCOME_LAKHS",
            nbins=30,
            title="Annual Income Distribution",
            labels={
                "INCOME_LAKHS": "Income (₹ Lakhs)"
            }
        )

        st.plotly_chart(
            fig_income,
            use_container_width=True
        )


# ============================================================
# INCOME VS TARGET
# ============================================================

st.subheader("💰 Income by Repayment Status")

if "TARGET_LABEL" in filtered_df.columns:

    income_target = filtered_df[
        filtered_df["INCOME_LAKHS"] > 0
    ].copy()

    # Limit extreme values only for chart readability
    upper_limit = income_target["INCOME_LAKHS"].quantile(0.99)

    income_target = income_target[
        income_target["INCOME_LAKHS"] <= upper_limit
    ]

    fig_income_target = px.box(
        income_target,
        x="TARGET_LABEL",
        y="INCOME_LAKHS",
        points=False,
        title="Income Distribution by Repayment Status",
        labels={
            "TARGET_LABEL": "Repayment Status",
            "INCOME_LAKHS": "Income (₹ Lakhs)"
        }
    )

    st.plotly_chart(
        fig_income_target,
        use_container_width=True
    )


# ============================================================
# CHILDREN & FAMILY MEMBERS
# ============================================================

st.subheader("👨‍👩‍👧 Family Composition")

col1, col2 = st.columns(2)


with col1:

    if "CNT_CHILDREN" in filtered_df.columns:

        children_df = (
            filtered_df["CNT_CHILDREN"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        children_df.columns = [
            "Children",
            "Customers"
        ]

        fig_children = px.bar(
            children_df,
            x="Children",
            y="Customers",
            text="Customers",
            title="Customers by Number of Children"
        )

        st.plotly_chart(
            fig_children,
            use_container_width=True
        )


with col2:

    if "CNT_FAM_MEMBERS" in filtered_df.columns:

        family_members = filtered_df[
            filtered_df["CNT_FAM_MEMBERS"] <= 10
        ]

        fig_family_members = px.histogram(
            family_members,
            x="CNT_FAM_MEMBERS",
            title="Family Members Distribution",
            labels={
                "CNT_FAM_MEMBERS": "Number of Family Members"
            }
        )

        st.plotly_chart(
            fig_family_members,
            use_container_width=True
        )


# ============================================================
# HOUSING ANALYSIS
# ============================================================

st.subheader("🏠 Housing & Ownership")

col1, col2 = st.columns(2)


with col1:

    if "NAME_HOUSING_TYPE" in filtered_df.columns:

        housing_df = (
            filtered_df["NAME_HOUSING_TYPE"]
            .value_counts()
            .reset_index()
        )

        housing_df.columns = [
            "Housing Type",
            "Customers"
        ]

        fig_housing = px.bar(
            housing_df,
            x="Customers",
            y="Housing Type",
            orientation="h",
            text="Customers",
            title="Customers by Housing Type"
        )

        st.plotly_chart(
            fig_housing,
            use_container_width=True
        )


# ============================================================
# CAR OWNERSHIP
# ============================================================

with col2:

    if "FLAG_OWN_CAR" in filtered_df.columns:

        car_df = (
            filtered_df["FLAG_OWN_CAR"]
            .value_counts()
            .reset_index()
        )

        car_df.columns = [
            "Car Ownership",
            "Customers"
        ]

        fig_car = px.pie(
            car_df,
            names="Car Ownership",
            values="Customers",
            hole=0.45,
            title="Car Ownership"
        )

        st.plotly_chart(
            fig_car,
            use_container_width=True
        )


# ============================================================
# REALTY OWNERSHIP
# ============================================================

if "FLAG_OWN_REALTY" in filtered_df.columns:

    realty_df = (
        filtered_df["FLAG_OWN_REALTY"]
        .value_counts()
        .reset_index()
    )

    realty_df.columns = [
        "Property Ownership",
        "Customers"
    ]

    fig_realty = px.pie(
        realty_df,
        names="Property Ownership",
        values="Customers",
        hole=0.45,
        title="Property Ownership"
    )

    st.plotly_chart(
        fig_realty,
        use_container_width=True
    )


# ============================================================
# AGE VS INCOME
# ============================================================

st.subheader("📈 Age vs Income")

scatter_data = filtered_df.copy()

scatter_data = scatter_data[
    scatter_data["AGE"].between(18, 100)
]

scatter_data = scatter_data[
    scatter_data["INCOME_LAKHS"] > 0
]

# Remove extreme income values for visualization
if len(scatter_data) > 0:

    income_limit = scatter_data[
        "INCOME_LAKHS"
    ].quantile(0.99)

    scatter_data = scatter_data[
        scatter_data["INCOME_LAKHS"] <= income_limit
    ]


if "TARGET_LABEL" in scatter_data.columns:

    fig_scatter = px.scatter(
        scatter_data,
        x="AGE",
        y="INCOME_LAKHS",
        color="TARGET_LABEL",
        opacity=0.55,
        hover_data=[
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS"
        ],
        title="Customer Age vs Annual Income",
        labels={
            "AGE": "Age",
            "INCOME_LAKHS": "Income (₹ Lakhs)",
            "TARGET_LABEL": "Repayment Status"
        }
    )

else:

    fig_scatter = px.scatter(
        scatter_data,
        x="AGE",
        y="INCOME_LAKHS",
        opacity=0.55,
        title="Customer Age vs Annual Income",
        labels={
            "AGE": "Age",
            "INCOME_LAKHS": "Income (₹ Lakhs)"
        }
    )


st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ============================================================
# DATA TABLE
# ============================================================

st.subheader("📋 Customer Demographic Data")

display_columns = [
    "SK_ID_CURR",
    "CODE_GENDER",
    "AGE",
    "NAME_FAMILY_STATUS",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "AMT_INCOME_TOTAL",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "TARGET"
]

available_columns = [
    col for col in display_columns
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

csv_data = filtered_df[available_columns].to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Demographic Data",
    data=csv_data,
    file_name="customer_demographics_filtered.csv",
    mime="text/csv"
)