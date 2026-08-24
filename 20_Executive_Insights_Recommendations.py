import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
st.title('Executive Insights & Recommendations'); df=add_application_features(require_dataset('application_train'))
if 'TARGET' in df: st.metric('Observed training default rate',f'{df.TARGET.mean()*100:.2f}%')
recommendations=['Investigate high credit-to-income ratios as an affordability signal.','Compare default rates across demographic and employment groups without treating correlations as causal.','Review missingness patterns because missingness itself may carry information.','Validate external-score relationships using robust model validation before operational use.','Use bureau and payment histories to build aggregated customer-level features for future modeling.']
for r in recommendations: st.markdown(f'- {r}')
st.info('These are EDA recommendations, not lending decisions. Validate all relationships statistically and for fairness before operational use.')





# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Executive Insights & Recommendations",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Executive Insights & Recommendations")

st.markdown(
    """
    This page summarizes the major findings from the Home Credit
    exploratory data analysis and converts them into actionable
    business insights and recommendations.

    **No machine-learning prediction is performed on this page.**
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/application_train.csv"
    )

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
# BASIC CLEANING
# ============================================================

df = df.copy()

# Remove impossible / invalid income values
df = df[
    df["AMT_INCOME_TOTAL"].notna()
    & (df["AMT_INCOME_TOTAL"] > 0)
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

if "DAYS_BIRTH" in df.columns:

    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )


if {
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL"
}.issubset(df.columns):

    df["CREDIT_TO_INCOME"] = (
        df["AMT_CREDIT"]
        / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )


if {
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL"
}.issubset(df.columns):

    df["ANNUITY_TO_INCOME"] = (
        df["AMT_ANNUITY"]
        / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )


if {
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS"
}.issubset(df.columns):

    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"]
        / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pct(value):
    return f"{value:.2f}%"


def format_money(value):

    if pd.isna(value):
        return "N/A"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:.0f}"


# ============================================================
# EXECUTIVE KPIs
# ============================================================

total_customers = (
    df["SK_ID_CURR"].nunique()
    if "SK_ID_CURR" in df.columns
    else len(df)
)

total_applications = len(df)

default_customers = (
    int(df["TARGET"].sum())
    if "TARGET" in df.columns
    else 0
)

non_default_customers = (
    total_applications - default_customers
)

default_rate = (
    default_customers / total_applications * 100
    if total_applications > 0
    else 0
)

total_credit = (
    df["AMT_CREDIT"].sum()
    if "AMT_CREDIT" in df.columns
    else 0
)

average_credit = (
    df["AMT_CREDIT"].mean()
    if "AMT_CREDIT" in df.columns
    else 0
)

average_income = (
    df["AMT_INCOME_TOTAL"].mean()
    if "AMT_INCOME_TOTAL" in df.columns
    else 0
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("Executive Summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with c2:
    st.metric(
        "Applications",
        f"{total_applications:,}"
    )

with c3:
    st.metric(
        "Default Customers",
        f"{default_customers:,}"
    )

with c4:
    st.metric(
        "Default Rate",
        pct(default_rate)
    )

with c5:
    st.metric(
        "Average Income",
        format_money(average_income)
    )

with c6:
    st.metric(
        "Average Credit",
        format_money(average_credit)
    )


st.divider()


# ============================================================
# SECTION 1 - DEFAULT DISTRIBUTION
# ============================================================

st.subheader("1. Portfolio Default Overview")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# DONUT CHART
# ------------------------------------------------------------

with col1:

    target_counts = (
        df["TARGET"]
        .value_counts()
        .rename(index={
            0: "Non-Default",
            1: "Default"
        })
        .reset_index()
    )

    target_counts.columns = [
        "Status",
        "Customers"
    ]

    fig = px.pie(
        target_counts,
        names="Status",
        values="Customers",
        hole=0.55,
        title="Default vs Non-Default Customers"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# DEFAULT RATE BAR
# ------------------------------------------------------------

with col2:

    default_summary = pd.DataFrame({
        "Status": [
            "Non-Default",
            "Default"
        ],
        "Customers": [
            non_default_customers,
            default_customers
        ]
    })

    fig = px.bar(
        default_summary,
        x="Status",
        y="Customers",
        text="Customers",
        title="Customer Risk Distribution"
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 2 - RISK FACTOR ANALYSIS
# ============================================================

st.subheader("2. Key Risk Factors")


def risk_by_category(
    data,
    column,
    top_n=10
):

    if column not in data.columns:
        return pd.DataFrame()

    temp = data[
        [column, "TARGET"]
    ].dropna()

    summary = (
        temp
        .groupby(column)
        .agg(
            Applications=("TARGET", "size"),
            Defaults=("TARGET", "sum")
        )
        .reset_index()
    )

    summary["Default Rate"] = (
        summary["Defaults"]
        / summary["Applications"]
        * 100
    )

    summary = summary[
        summary["Applications"] >= 100
    ]

    return (
        summary
        .sort_values(
            "Default Rate",
            ascending=False
        )
        .head(top_n)
    )


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# EDUCATION
# ------------------------------------------------------------

with col1:

    education_risk = risk_by_category(
        df,
        "NAME_EDUCATION_TYPE",
        10
    )

    if not education_risk.empty:

        fig = px.bar(
            education_risk.sort_values(
                "Default Rate"
            ),
            x="Default Rate",
            y="NAME_EDUCATION_TYPE",
            orientation="h",
            text="Default Rate",
            title="Default Rate by Education"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ------------------------------------------------------------
# INCOME TYPE
# ------------------------------------------------------------

with col2:

    income_type_risk = risk_by_category(
        df,
        "NAME_INCOME_TYPE",
        10
    )

    if not income_type_risk.empty:

        fig = px.bar(
            income_type_risk.sort_values(
                "Default Rate"
            ),
            x="Default Rate",
            y="NAME_INCOME_TYPE",
            orientation="h",
            text="Default Rate",
            title="Default Rate by Income Type"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CONTRACT TYPE
# ============================================================

contract_risk = risk_by_category(
    df,
    "NAME_CONTRACT_TYPE",
    10
)

if not contract_risk.empty:

    fig = px.bar(
        contract_risk,
        x="NAME_CONTRACT_TYPE",
        y="Default Rate",
        text="Default Rate",
        title="Default Rate by Contract Type"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 3 - INCOME DISTRIBUTION
# ============================================================

st.subheader("3. Income & Credit Exposure")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# INCOME HISTOGRAM
# ------------------------------------------------------------

with col1:

    income_sample = df[
        "AMT_INCOME_TOTAL"
    ].dropna()

    income_sample = income_sample[
        income_sample <= income_sample.quantile(0.99)
    ]

    fig = px.histogram(
        income_sample,
        x="AMT_INCOME_TOTAL",
        nbins=50,
        title="Income Distribution"
    )

    fig.update_xaxes(
        title="Total Income"
    )

    fig.update_yaxes(
        title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# CREDIT BOX PLOT
# ------------------------------------------------------------

with col2:

    box_df = df[
        ["TARGET", "AMT_CREDIT"]
    ].dropna()

    box_df["Risk"] = box_df["TARGET"].map({
        0: "Non-Default",
        1: "Default"
    })

    # Limit extreme values for visualization
    upper_limit = box_df[
        "AMT_CREDIT"
    ].quantile(0.99)

    box_df = box_df[
        box_df["AMT_CREDIT"] <= upper_limit
    ]

    fig = px.box(
        box_df,
        x="Risk",
        y="AMT_CREDIT",
        color="Risk",
        title="Credit Amount by Risk Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 4 - AFFORDABILITY
# ============================================================

st.subheader("4. Credit Affordability Analysis")


scatter_df = df[
    [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "TARGET"
    ]
].dropna()


# Remove extreme outliers for visualization

income_limit = scatter_df[
    "AMT_INCOME_TOTAL"
].quantile(0.99)

credit_limit = scatter_df[
    "AMT_CREDIT"
].quantile(0.99)

scatter_df = scatter_df[
    (scatter_df["AMT_INCOME_TOTAL"] <= income_limit)
    &
    (scatter_df["AMT_CREDIT"] <= credit_limit)
]


scatter_df["Risk"] = scatter_df["TARGET"].map({
    0: "Non-Default",
    1: "Default"
})


fig = px.scatter(
    scatter_df.sample(
        min(15000, len(scatter_df)),
        random_state=42
    ),
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    color="Risk",
    opacity=0.6,
    title="Income vs Credit Amount",
    labels={
        "AMT_INCOME_TOTAL": "Income",
        "AMT_CREDIT": "Credit Amount"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SECTION 5 - CREDIT TO INCOME
# ============================================================

if "CREDIT_TO_INCOME" in df.columns:

    ratio_df = df[
        [
            "CREDIT_TO_INCOME",
            "TARGET"
        ]
    ].dropna()

    ratio_df = ratio_df[
        ratio_df["CREDIT_TO_INCOME"] <=
        ratio_df["CREDIT_TO_INCOME"].quantile(0.99)
    ]

    ratio_df["Risk"] = ratio_df["TARGET"].map({
        0: "Non-Default",
        1: "Default"
    })

    fig = px.box(
        ratio_df,
        x="Risk",
        y="CREDIT_TO_INCOME",
        color="Risk",
        title="Credit-to-Income Ratio by Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 6 - CORRELATION HEATMAP
# ============================================================

st.subheader("5. Risk Variable Relationships")


risk_columns = [
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AGE_YEARS",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME"
]


available_columns = [
    col for col in risk_columns
    if col in df.columns
]


corr = df[
    available_columns
].corr(numeric_only=True)


fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Heatmap of Key Risk Variables"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SECTION 7 - SUNBURST
# ============================================================

st.subheader("6. Hierarchical Risk Analysis")


sunburst_columns = [
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "TARGET"
]


if all(
    col in df.columns
    for col in sunburst_columns
):

    sun_df = df[
        sunburst_columns
    ].dropna()

    sun_df["TARGET_LABEL"] = sun_df[
        "TARGET"
    ].map({
        0: "Non-Default",
        1: "Default"
    })

    sun_df = sun_df.drop(
        columns=["TARGET"]
    )

    sun_df = (
        sun_df
        .groupby([
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "TARGET_LABEL"
        ])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.sunburst(
        sun_df,
        path=[
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "TARGET_LABEL"
        ],
        values="Customers",
        title="Education → Income Type → Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 8 - RISK FACTOR RANKING
# ============================================================

st.subheader("7. Risk Factor Ranking")


risk_factor_results = []


risk_factor_columns = {
    "Education": "NAME_EDUCATION_TYPE",
    "Income Type": "NAME_INCOME_TYPE",
    "Housing Type": "NAME_HOUSING_TYPE",
    "Family Status": "NAME_FAMILY_STATUS",
    "Contract Type": "NAME_CONTRACT_TYPE",
    "Occupation": "OCCUPATION_TYPE"
}


for factor_name, column in risk_factor_columns.items():

    if column not in df.columns:
        continue

    temp = df[
        [column, "TARGET"]
    ].dropna()

    grouped = (
        temp
        .groupby(column)["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
    )

    grouped = grouped[
        grouped["count"] >= 100
    ]

    if grouped.empty:
        continue

    max_risk = grouped[
        "mean"
    ].max() * 100

    min_risk = grouped[
        "mean"
    ].min() * 100

    risk_factor_results.append({
        "Risk Factor": factor_name,
        "Highest Default Rate": max_risk,
        "Lowest Default Rate": min_risk,
        "Risk Gap": max_risk - min_risk
    })


risk_factor_df = pd.DataFrame(
    risk_factor_results
)


if not risk_factor_df.empty:

    risk_factor_df = risk_factor_df.sort_values(
        "Risk Gap",
        ascending=False
    )

    fig = px.bar(
        risk_factor_df,
        x="Risk Gap",
        y="Risk Factor",
        orientation="h",
        text="Risk Gap",
        title="Risk Factor Impact Ranking"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        risk_factor_df.style.format({
            "Highest Default Rate": "{:.2f}%",
            "Lowest Default Rate": "{:.2f}%",
            "Risk Gap": "{:.2f}%"
        }),
        use_container_width=True
    )


# ============================================================
# SECTION 9 - BUSINESS INSIGHTS
# ============================================================

st.subheader("8. Key Business Insights")


insights = []


# Default rate insight

insights.append(
    f"""
    **Overall portfolio risk:** The observed default rate is
    **{default_rate:.2f}%** across {total_applications:,}
    applications.
    """
)


# Income insight

if not income_type_risk.empty:

    highest_income_type = (
        income_type_risk.iloc[0]
        ["NAME_INCOME_TYPE"]
    )

    highest_income_rate = (
        income_type_risk.iloc[0]
        ["Default Rate"]
    )

    insights.append(
        f"""
        **Income-type risk:** Among income groups with at least
        100 applications, **{highest_income_type}** has the highest
        observed default rate at approximately
        **{highest_income_rate:.2f}%**.
        """
    )


# Education insight

if not education_risk.empty:

    highest_education = (
        education_risk.iloc[0]
        ["NAME_EDUCATION_TYPE"]
    )

    highest_education_rate = (
        education_risk.iloc[0]
        ["Default Rate"]
    )

    insights.append(
        f"""
        **Education risk:** **{highest_education}** shows the
        highest observed default rate among the education groups
        retained for analysis, at approximately
        **{highest_education_rate:.2f}%**.
        """
    )


# Affordability insight

if "CREDIT_TO_INCOME" in df.columns:

    median_ratio_default = df.loc[
        df["TARGET"] == 1,
        "CREDIT_TO_INCOME"
    ].median()

    median_ratio_nondefault = df.loc[
        df["TARGET"] == 0,
        "CREDIT_TO_INCOME"
    ].median()

    if (
        pd.notna(median_ratio_default)
        and pd.notna(median_ratio_nondefault)
    ):

        insights.append(
            f"""
            **Affordability:** The median credit-to-income ratio
            for default customers is **{median_ratio_default:.2f}**
            compared with **{median_ratio_nondefault:.2f}** for
            non-default customers.
            """
        )


for insight in insights:

    st.markdown(
        f"- {insight.strip()}"
    )


# ============================================================
# SECTION 10 - RECOMMENDATIONS
# ============================================================

st.subheader("9. Business Recommendations")


recommendations = [
    {
        "Priority": "High",
        "Recommendation": (
            "Monitor customer segments with consistently "
            "higher observed default rates."
        ),
        "Business Action": (
            "Perform deeper affordability and customer-profile "
            "analysis before approving higher-risk applications."
        )
    },
    {
        "Priority": "High",
        "Recommendation": (
            "Monitor high credit-to-income exposure."
        ),
        "Business Action": (
            "Review applications where requested credit is large "
            "relative to reported income."
        )
    },
    {
        "Priority": "Medium",
        "Recommendation": (
            "Investigate income-type and education segments "
            "with elevated default rates."
        ),
        "Business Action": (
            "Create segment-level lending policies and "
            "financial education initiatives."
        )
    },
    {
        "Priority": "Medium",
        "Recommendation": (
            "Monitor large credit amounts and affordability."
        ),
        "Business Action": (
            "Use credit amount, income, annuity and goods price "
            "together for manual portfolio monitoring."
        )
    },
    {
        "Priority": "Medium",
        "Recommendation": (
            "Use historical bureau and repayment information "
            "for deeper customer analysis."
        ),
        "Business Action": (
            "Combine application-level findings with bureau, "
            "installment and previous-application history."
        )
    },
    {
        "Priority": "Low",
        "Recommendation": (
            "Continuously monitor portfolio-level default trends."
        ),
        "Business Action": (
            "Build periodic management reports comparing "
            "customer segments and loan characteristics."
        )
    }
]


recommendation_df = pd.DataFrame(
    recommendations
)


st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL EXECUTIVE SUMMARY
# ============================================================

st.divider()

st.subheader("10. Executive Conclusion")

st.success(
    f"""
    **Executive Summary**

    The Home Credit portfolio contains **{total_applications:,}**
    applications with an observed default rate of
    **{default_rate:.2f}%**.

    The analysis indicates that risk should be evaluated from
    multiple dimensions, including customer demographics,
    income characteristics, credit exposure, affordability,
    education, income type and contract characteristics.

    The most important business focus should be on identifying
    high-risk customer segments, monitoring credit affordability,
    understanding portfolio concentration and combining these
    findings with historical credit and repayment information.

    This page is intended for **EDA, business insights and
    decision-support reporting only** and does not perform
    machine-learning prediction.
    """
)