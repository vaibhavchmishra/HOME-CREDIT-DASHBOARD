import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
st.title('Risk Factor Analysis'); df=add_application_features(require_dataset('application_train'))
for c in ['AGE_YEARS','AMT_INCOME_TOTAL','CREDIT_INCOME_RATIO']:
    if c in df: st.plotly_chart(px.histogram(df,x=c,color='TARGET',nbins=40,title=f'{c} by Default'),use_container_width=True)




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk Factor Analysis | Home Credit",
    page_icon="⚠️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚠️ Risk-Factor Analysis")

st.markdown(
    """
    Explore customer, demographic, financial, employment, housing,
    loan, asset, regional, and external-score factors associated
    with historical payment difficulties.
    
    **EDA only — no machine-learning prediction is used.**
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
# TARGET LABEL
# ------------------------------------------------------------

df["TARGET_LABEL"] = df["TARGET"].map({
    0: "No Payment Difficulties",
    1: "Payment Difficulties"
})


# ------------------------------------------------------------
# AGE
# ------------------------------------------------------------

df["AGE_YEARS"] = (
    -df["DAYS_BIRTH"] / 365.25
)


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
# EMPLOYMENT
# ------------------------------------------------------------

df["EMPLOYMENT_YEARS"] = np.where(
    df["DAYS_EMPLOYED"] == 365243,
    np.nan,
    -df["DAYS_EMPLOYED"] / 365.25
)


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


# ------------------------------------------------------------
# CREDIT-TO-INCOME
# ------------------------------------------------------------

income_safe = df[
    "AMT_INCOME_TOTAL"
].replace(0, np.nan)

df["CREDIT_TO_INCOME"] = (
    df["AMT_CREDIT"] / income_safe
)


# ------------------------------------------------------------
# ANNUITY-TO-INCOME
# ------------------------------------------------------------

df["ANNUITY_TO_INCOME"] = (
    df["AMT_ANNUITY"] / income_safe
)


# ------------------------------------------------------------
# CREDIT-TO-GOODS
# ------------------------------------------------------------

goods_safe = df[
    "AMT_GOODS_PRICE"
].replace(0, np.nan)

df["CREDIT_TO_GOODS"] = (
    df["AMT_CREDIT"] / goods_safe
)


# ------------------------------------------------------------
# CHILDREN GROUP
# ------------------------------------------------------------

df["CHILDREN_GROUP"] = pd.cut(
    df["CNT_CHILDREN"],
    bins=[
        -1,
        0,
        1,
        2,
        3,
        np.inf
    ],
    labels=[
        "0",
        "1",
        "2",
        "3",
        "4+"
    ]
)


# ------------------------------------------------------------
# FAMILY SIZE GROUP
# ------------------------------------------------------------

df["FAMILY_SIZE_GROUP"] = pd.cut(
    df["CNT_FAM_MEMBERS"],
    bins=[
        0,
        1,
        2,
        3,
        4,
        np.inf
    ],
    labels=[
        "1",
        "2",
        "3",
        "4",
        "5+"
    ]
)


# ============================================================
# EXTERNAL SCORE GROUPS
# ============================================================

score_columns = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]


for col in score_columns:

    if col in df.columns:

        df[f"{col}_GROUP"] = pd.cut(
            df[col],
            bins=[
                -np.inf,
                0.2,
                0.4,
                0.6,
                0.8,
                np.inf
            ],
            labels=[
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Risk-Factor Filters")

filtered_df = df.copy()


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
# CONTRACT TYPE
# ============================================================

contract_values = sorted(
    df["NAME_CONTRACT_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)


selected_contract = st.sidebar.multiselect(
    "Contract Type",
    contract_values,
    default=contract_values
)


if selected_contract:

    filtered_df = filtered_df[
        filtered_df["NAME_CONTRACT_TYPE"]
        .astype(str)
        .isin(selected_contract)
    ]


# ============================================================
# INCOME TYPE
# ============================================================

income_types = sorted(
    df["NAME_INCOME_TYPE"]
    .dropna()
    .astype(str)
    .unique()
)


selected_income = st.sidebar.multiselect(
    "Income Type",
    income_types,
    default=income_types
)


if selected_income:

    filtered_df = filtered_df[
        filtered_df["NAME_INCOME_TYPE"]
        .astype(str)
        .isin(selected_income)
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
# KPI
# ============================================================

total_applications = len(filtered_df)

risk_rate = (
    filtered_df["TARGET"].mean() * 100
)

risk_cases = (
    filtered_df["TARGET"] == 1
).sum()

non_risk_cases = (
    filtered_df["TARGET"] == 0
).sum()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Risk Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Applications",
        f"{total_applications:,}"
    )


with col2:

    st.metric(
        "Payment-Difficulty Cases",
        f"{risk_cases:,}"
    )


with col3:

    st.metric(
        "No-Difficulty Cases",
        f"{non_risk_cases:,}"
    )


with col4:

    st.metric(
        "Payment-Difficulty Rate",
        f"{risk_rate:.2f}%"
    )


st.divider()


# ============================================================
# HELPER FUNCTION
# ============================================================

def risk_rate_chart(
    data,
    category,
    title,
    x_label=None
):

    summary = (
        data
        .groupby(
            category,
            dropna=False,
            observed=True
        )["TARGET"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    summary.columns = [
        category,
        "Payment Difficulty Rate"
    ]

    summary[category] = (
        summary[category]
        .astype(str)
    )

    fig = px.bar(
        summary,
        x="Payment Difficulty Rate",
        y=category,
        orientation="h",
        text="Payment Difficulty Rate",
        title=title,
        labels={
            "Payment Difficulty Rate":
                "Payment Difficulty Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    return fig


# ============================================================
# DEMOGRAPHIC RISK FACTORS
# ============================================================

st.subheader("👥 Demographic Risk Factors")


col1, col2 = st.columns(2)


with col1:

    fig = risk_rate_chart(
        filtered_df,
        "CODE_GENDER",
        "Payment-Difficulty Rate by Gender"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = risk_rate_chart(
        filtered_df,
        "AGE_GROUP",
        "Payment-Difficulty Rate by Age Group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EDUCATION & FAMILY
# ============================================================

col1, col2 = st.columns(2)


with col1:

    fig = risk_rate_chart(
        filtered_df,
        "NAME_EDUCATION_TYPE",
        "Payment-Difficulty Rate by Education"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = risk_rate_chart(
        filtered_df,
        "NAME_FAMILY_STATUS",
        "Payment-Difficulty Rate by Family Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CHILDREN
# ============================================================

st.subheader("👨‍👩‍👧 Family Size Risk")


col1, col2 = st.columns(2)


with col1:

    fig = risk_rate_chart(
        filtered_df,
        "CHILDREN_GROUP",
        "Payment-Difficulty Rate by Number of Children"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = risk_rate_chart(
        filtered_df,
        "FAMILY_SIZE_GROUP",
        "Payment-Difficulty Rate by Family Size"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# INCOME & EMPLOYMENT
# ============================================================

st.subheader("💼 Income & Employment Risk Factors")


col1, col2 = st.columns(2)


with col1:

    fig = risk_rate_chart(
        filtered_df,
        "NAME_INCOME_TYPE",
        "Payment-Difficulty Rate by Income Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = risk_rate_chart(
        filtered_df,
        "EMPLOYMENT_GROUP",
        "Payment-Difficulty Rate by Employment Duration"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# OCCUPATION
# ============================================================

if "OCCUPATION_TYPE" in filtered_df.columns:

    occupation_df = (
        filtered_df
        .groupby("OCCUPATION_TYPE")[
            "TARGET"
        ]
        .agg(
            ["mean", "count"]
        )
        .reset_index()
    )

    occupation_df.columns = [
        "Occupation",
        "Risk Rate",
        "Applications"
    ]

    occupation_df["Risk Rate"] *= 100

    occupation_df = occupation_df[
        occupation_df["Applications"] >= 100
    ]

    occupation_df = occupation_df.sort_values(
        "Risk Rate",
        ascending=False
    )

    fig_occupation = px.bar(
        occupation_df,
        x="Risk Rate",
        y="Occupation",
        orientation="h",
        text="Risk Rate",
        title="Payment-Difficulty Rate by Occupation "
              "(minimum 100 applications)",
        labels={
            "Risk Rate":
                "Payment Difficulty Rate (%)"
        }
    )

    fig_occupation.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_occupation,
        use_container_width=True
    )


# ============================================================
# INCOME DISTRIBUTION
# ============================================================

st.subheader("💰 Income as a Risk Factor")


income_box = filtered_df[
    [
        "TARGET_LABEL",
        "AMT_INCOME_TOTAL"
    ]
].dropna()


income_limit = income_box[
    "AMT_INCOME_TOTAL"
].quantile(0.99)


income_box = income_box[
    income_box["AMT_INCOME_TOTAL"]
    <= income_limit
]


fig_income = px.box(
    income_box,
    x="TARGET_LABEL",
    y="AMT_INCOME_TOTAL",
    points=False,
    title="Income Distribution by Payment Status",
    labels={
        "TARGET_LABEL":
            "Payment Status",
        "AMT_INCOME_TOTAL":
            "Annual Income"
    }
)


st.plotly_chart(
    fig_income,
    use_container_width=True
)


# ============================================================
# LOAN AFFORDABILITY
# ============================================================

st.subheader("💳 Loan Affordability Risk Factors")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Credit-to-Income
# ------------------------------------------------------------

with col1:

    ratio_data = filtered_df[
        [
            "TARGET_LABEL",
            "CREDIT_TO_INCOME"
        ]
    ].dropna()

    ratio_data = ratio_data[
        ratio_data["CREDIT_TO_INCOME"]
        .between(0, 10)
    ]

    fig_ratio = px.box(
        ratio_data,
        x="TARGET_LABEL",
        y="CREDIT_TO_INCOME",
        points=False,
        title="Credit-to-Income by Payment Status",
        labels={
            "TARGET_LABEL":
                "Payment Status",
            "CREDIT_TO_INCOME":
                "Credit / Annual Income"
        }
    )

    st.plotly_chart(
        fig_ratio,
        use_container_width=True
    )


# ------------------------------------------------------------
# Annuity-to-Income
# ------------------------------------------------------------

with col2:

    annuity_ratio = filtered_df[
        [
            "TARGET_LABEL",
            "ANNUITY_TO_INCOME"
        ]
    ].dropna()

    annuity_ratio = annuity_ratio[
        annuity_ratio["ANNUITY_TO_INCOME"]
        .between(0, 1)
    ]

    fig_annuity = px.box(
        annuity_ratio,
        x="TARGET_LABEL",
        y="ANNUITY_TO_INCOME",
        points=False,
        title="Annuity-to-Income by Payment Status",
        labels={
            "TARGET_LABEL":
                "Payment Status",
            "ANNUITY_TO_INCOME":
                "Annuity / Annual Income"
        }
    )

    st.plotly_chart(
        fig_annuity,
        use_container_width=True
    )


# ============================================================
# CREDIT & ANNUITY AMOUNTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    credit_data = filtered_df[
        [
            "TARGET_LABEL",
            "AMT_CREDIT"
        ]
    ].dropna()

    credit_limit = credit_data[
        "AMT_CREDIT"
    ].quantile(0.99)

    credit_data = credit_data[
        credit_data["AMT_CREDIT"]
        <= credit_limit
    ]

    fig_credit = px.box(
        credit_data,
        x="TARGET_LABEL",
        y="AMT_CREDIT",
        points=False,
        title="Credit Amount by Payment Status"
    )

    st.plotly_chart(
        fig_credit,
        use_container_width=True
    )


with col2:

    annuity_data = filtered_df[
        [
            "TARGET_LABEL",
            "AMT_ANNUITY"
        ]
    ].dropna()

    annuity_limit = annuity_data[
        "AMT_ANNUITY"
    ].quantile(0.99)

    annuity_data = annuity_data[
        annuity_data["AMT_ANNUITY"]
        <= annuity_limit
    ]

    fig_annuity_amount = px.box(
        annuity_data,
        x="TARGET_LABEL",
        y="AMT_ANNUITY",
        points=False,
        title="Annuity Amount by Payment Status"
    )

    st.plotly_chart(
        fig_annuity_amount,
        use_container_width=True
    )


# ============================================================
# HOUSING & ASSET RISK
# ============================================================

st.subheader("🏠 Housing & Asset Risk Factors")


col1, col2 = st.columns(2)


with col1:

    fig = risk_rate_chart(
        filtered_df,
        "NAME_HOUSING_TYPE",
        "Payment-Difficulty Rate by Housing Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    if "FLAG_OWN_CAR" in filtered_df.columns:

        fig = risk_rate_chart(
            filtered_df,
            "FLAG_OWN_CAR",
            "Payment-Difficulty Rate by Car Ownership"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# REALTY OWNERSHIP
# ============================================================

if "FLAG_OWN_REALTY" in filtered_df.columns:

    fig = risk_rate_chart(
        filtered_df,
        "FLAG_OWN_REALTY",
        "Payment-Difficulty Rate by Realty Ownership"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EXTERNAL SCORE RISK
# ============================================================

st.subheader("📊 External Credit Score Risk Factors")


available_scores = [
    col
    for col in score_columns
    if col in filtered_df.columns
]


score_tabs = st.tabs(
    [
        col
        for col in available_scores
    ]
)


for tab, score_column in zip(
    score_tabs,
    available_scores
):

    with tab:

        score_group = f"{score_column}_GROUP"

        if score_group in filtered_df.columns:

            fig = risk_rate_chart(
                filtered_df,
                score_group,
                f"Payment-Difficulty Rate by {score_column} Group"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            score_data = filtered_df[
                [
                    "TARGET_LABEL",
                    score_column
                ]
            ].dropna()

            fig = px.box(
                score_data,
                x="TARGET_LABEL",
                y=score_column,
                points=False,
                title=f"{score_column} by Payment Status"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# REGION RISK
# ============================================================

st.subheader("🌍 Regional Risk Factors")


region_columns = [
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY"
]


available_region_columns = [
    col
    for col in region_columns
    if col in filtered_df.columns
]


for region_column in available_region_columns:

    fig = risk_rate_chart(
        filtered_df,
        region_column,
        f"Payment-Difficulty Rate by {region_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RISK FACTOR SUMMARY
# ============================================================

st.subheader("📋 Risk-Factor Summary")


summary_rows = []


factor_columns = [
    ("Gender", "CODE_GENDER"),
    ("Education", "NAME_EDUCATION_TYPE"),
    ("Family Status", "NAME_FAMILY_STATUS"),
    ("Housing", "NAME_HOUSING_TYPE"),
    ("Income Type", "NAME_INCOME_TYPE"),
    ("Age Group", "AGE_GROUP"),
    ("Employment Group", "EMPLOYMENT_GROUP"),
    ("Children Group", "CHILDREN_GROUP"),
    ("Family Size", "FAMILY_SIZE_GROUP"),
]


for factor_name, factor_column in factor_columns:

    if factor_column not in filtered_df.columns:
        continue

    grouped = (
        filtered_df
        .groupby(
            factor_column,
            observed=True
        )["TARGET"]
        .mean()
        .mul(100)
        .dropna()
    )

    if len(grouped) == 0:
        continue

    highest_category = grouped.idxmax()
    highest_rate = grouped.max()

    lowest_category = grouped.idxmin()
    lowest_rate = grouped.min()

    summary_rows.append({
        "Risk Factor": factor_name,
        "Highest-Risk Category":
            str(highest_category),
        "Highest Risk Rate (%)":
            round(highest_rate, 2),
        "Lowest-Risk Category":
            str(lowest_category),
        "Lowest Risk Rate (%)":
            round(lowest_rate, 2)
    })


risk_summary = pd.DataFrame(
    summary_rows
)


st.dataframe(
    risk_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FILTERED DATA
# ============================================================

st.subheader("🔍 Risk-Factor Data")


display_columns = [
    "SK_ID_CURR",
    "TARGET",
    "TARGET_LABEL",
    "CODE_GENDER",
    "AGE_YEARS",
    "AGE_GROUP",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "OCCUPATION_TYPE",
    "EMPLOYMENT_YEARS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY"
]


available_display = [
    col
    for col in display_columns
    if col in filtered_df.columns
]


st.dataframe(
    filtered_df[available_display],
    use_container_width=True,
    height=450
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = (
    filtered_df[available_display]
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Risk-Factor Data",
    data=csv_data,
    file_name="risk_factor_filtered.csv",
    mime="text/csv"
)