import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import require_dataset
from utils.feature_engineering import add_application_features
st.title('Customer Risk Segmentation'); df=add_application_features(require_dataset('application_train'))
if 'TARGET' in df:
    def seg(r):
        if r.TARGET==1:return 'High Risk'
        if 'CREDIT_INCOME_RATIO' in df and r.get('CREDIT_INCOME_RATIO',0)>5:return 'Affordability Watch'
        return 'Lower Observed Risk'
    df['RISK_SEGMENT']=df.apply(seg,axis=1); st.dataframe(df.RISK_SEGMENT.value_counts().rename('customers').to_frame(),use_container_width=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Risk Segmentation",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎯 Customer Risk Segmentation & Recommendations")

st.markdown(
    """
    This page segments Home Credit customers into interpretable
    risk groups using customer affordability, income, credit
    exposure and external credit-score indicators.

    **This is rule-based EDA segmentation and does not train
    a machine-learning prediction model.**
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


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "application_train.csv was not found in the data/ folder."
    )

    st.stop()


df = df.copy()


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY"
]


missing_required = [
    col for col in required_columns
    if col not in df.columns
]


if missing_required:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_required)
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["CREDIT_TO_INCOME"] = (
    df["AMT_CREDIT"]
    / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
)


df["ANNUITY_TO_INCOME"] = (
    df["AMT_ANNUITY"]
    / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
)


if "DAYS_BIRTH" in df.columns:

    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )


# ============================================================
# EXTERNAL SCORE
# ============================================================

external_score_columns = [
    col
    for col in [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]
    if col in df.columns
]


if external_score_columns:

    df["EXTERNAL_SCORE"] = (
        df[external_score_columns]
        .mean(axis=1)
    )

else:

    df["EXTERNAL_SCORE"] = np.nan


# ============================================================
# CLEAN EXTREME VALUES FOR SEGMENTATION
# ============================================================

df["CREDIT_TO_INCOME"] = (
    df["CREDIT_TO_INCOME"]
    .replace([np.inf, -np.inf], np.nan)
)


df["ANNUITY_TO_INCOME"] = (
    df["ANNUITY_TO_INCOME"]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# SIDEBAR THRESHOLDS
# ============================================================

st.sidebar.header("Risk Segmentation Rules")


credit_ratio_threshold = st.sidebar.slider(
    "High Credit-to-Income Threshold",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.25
)


annuity_ratio_threshold = st.sidebar.slider(
    "High Annuity-to-Income Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.40,
    step=0.05
)


external_score_threshold = st.sidebar.slider(
    "Low External Score Threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05
)


# ============================================================
# RISK SCORE
# ============================================================

df["RISK_SCORE"] = 0


# High credit burden

df.loc[
    df["CREDIT_TO_INCOME"]
    >= credit_ratio_threshold,
    "RISK_SCORE"
] += 1


# High annuity burden

df.loc[
    df["ANNUITY_TO_INCOME"]
    >= annuity_ratio_threshold,
    "RISK_SCORE"
] += 1


# Low external score

if "EXTERNAL_SCORE" in df.columns:

    df.loc[
        df["EXTERNAL_SCORE"]
        < external_score_threshold,
        "RISK_SCORE"
    ] += 1


# Existing target is NOT used to create the segment.
# TARGET is used only afterward to compare observed
# default rates across segments.


# ============================================================
# SEGMENT CREATION
# ============================================================

def assign_segment(score):

    if score >= 2:
        return "High Risk"

    elif score == 1:
        return "Medium Risk"

    else:
        return "Low Risk"


df["RISK_SEGMENT"] = (
    df["RISK_SCORE"]
    .apply(assign_segment)
)


# ============================================================
# SEGMENT ORDER
# ============================================================

segment_order = [
    "Low Risk",
    "Medium Risk",
    "High Risk"
]


df["RISK_SEGMENT"] = pd.Categorical(
    df["RISK_SEGMENT"],
    categories=segment_order,
    ordered=True
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = (
    df["SK_ID_CURR"]
    .nunique()
)


high_risk_customers = (
    df["RISK_SEGMENT"]
    .eq("High Risk")
    .sum()
)


medium_risk_customers = (
    df["RISK_SEGMENT"]
    .eq("Medium Risk")
    .sum()
)


low_risk_customers = (
    df["RISK_SEGMENT"]
    .eq("Low Risk")
    .sum()
)


high_risk_percentage = (
    high_risk_customers
    / total_customers
    * 100
)


medium_risk_percentage = (
    medium_risk_customers
    / total_customers
    * 100
)


low_risk_percentage = (
    low_risk_customers
    / total_customers
    * 100
)


# ============================================================
# EXECUTIVE KPI CARDS
# ============================================================

st.subheader("1. Risk Portfolio Overview")


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with k2:

    st.metric(
        "High Risk",
        f"{high_risk_customers:,}",
        f"{high_risk_percentage:.2f}%"
    )


with k3:

    st.metric(
        "Medium Risk",
        f"{medium_risk_customers:,}",
        f"{medium_risk_percentage:.2f}%"
    )


with k4:

    st.metric(
        "Low Risk",
        f"{low_risk_customers:,}",
        f"{low_risk_percentage:.2f}%"
    )


with k5:

    st.metric(
        "Risk Indicators",
        f"{df['RISK_SCORE'].mean():.2f}",
        "Average score"
    )


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

st.subheader("2. Customer Risk Distribution")


segment_counts = (
    df["RISK_SEGMENT"]
    .value_counts()
    .reindex(segment_order)
    .fillna(0)
    .reset_index()
)


segment_counts.columns = [
    "Risk Segment",
    "Customers"
]


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# DONUT CHART
# ------------------------------------------------------------

with col1:

    fig = px.pie(
        segment_counts,
        names="Risk Segment",
        values="Customers",
        hole=0.55,
        title="Customer Risk Segment Composition"
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
# BAR CHART
# ------------------------------------------------------------

with col2:

    fig = px.bar(
        segment_counts,
        x="Risk Segment",
        y="Customers",
        text="Customers",
        title="Customers by Risk Segment"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DEFAULT RATE BY SEGMENT
# ============================================================

st.subheader("3. Observed Default Rate by Risk Segment")


segment_risk = (
    df.groupby(
        "RISK_SEGMENT",
        observed=False
    )
    .agg(
        Customers=("SK_ID_CURR", "nunique"),
        Defaults=("TARGET", "sum")
    )
    .reset_index()
)


segment_risk["Default Rate"] = (
    segment_risk["Defaults"]
    / segment_risk["Customers"]
    * 100
)


fig = px.bar(
    segment_risk,
    x="RISK_SEGMENT",
    y="Default Rate",
    text="Default Rate",
    title="Observed Default Rate by Risk Segment"
)


fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig.update_layout(
    xaxis_title="Risk Segment",
    yaxis_title="Default Rate (%)"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.caption(
    "TARGET is used here only to evaluate the observed historical "
    "default rate within each rule-based segment; it is not used "
    "to create the segment."
)


# ============================================================
# CREDIT-TO-INCOME BY SEGMENT
# ============================================================

st.subheader("4. Affordability by Risk Segment")


ratio_plot = df[
    [
        "RISK_SEGMENT",
        "CREDIT_TO_INCOME"
    ]
].dropna()


upper_ratio = ratio_plot[
    "CREDIT_TO_INCOME"
].quantile(0.99)


ratio_plot = ratio_plot[
    ratio_plot["CREDIT_TO_INCOME"]
    <= upper_ratio
]


fig = px.box(
    ratio_plot,
    x="RISK_SEGMENT",
    y="CREDIT_TO_INCOME",
    color="RISK_SEGMENT",
    category_orders={
        "RISK_SEGMENT": segment_order
    },
    title="Credit-to-Income Ratio by Risk Segment"
)


fig.update_layout(
    xaxis_title="Risk Segment",
    yaxis_title="Credit-to-Income Ratio"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# INCOME BY SEGMENT
# ============================================================

st.subheader("5. Income Distribution by Risk Segment")


income_plot = df[
    [
        "RISK_SEGMENT",
        "AMT_INCOME_TOTAL"
    ]
].dropna()


upper_income = income_plot[
    "AMT_INCOME_TOTAL"
].quantile(0.99)


income_plot = income_plot[
    income_plot["AMT_INCOME_TOTAL"]
    <= upper_income
]


fig = px.box(
    income_plot,
    x="RISK_SEGMENT",
    y="AMT_INCOME_TOTAL",
    color="RISK_SEGMENT",
    category_orders={
        "RISK_SEGMENT": segment_order
    },
    title="Income Distribution by Risk Segment"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# INCOME VS CREDIT
# ============================================================

st.subheader("6. Income vs Credit Exposure")


scatter_columns = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "RISK_SEGMENT"
]


scatter_df = df[
    scatter_columns
].dropna()


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


if len(scatter_df) > 15000:

    scatter_df = scatter_df.sample(
        15000,
        random_state=42
    )


fig = px.scatter(
    scatter_df,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    color="RISK_SEGMENT",
    opacity=0.55,
    category_orders={
        "RISK_SEGMENT": segment_order
    },
    title="Customer Income vs Credit Amount"
)


fig.update_layout(
    xaxis_title="Customer Income",
    yaxis_title="Credit Amount"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FINANCIAL BURDEN SCATTER
# ============================================================

st.subheader("7. Credit Burden vs Annuity Burden")


burden_df = df[
    [
        "CREDIT_TO_INCOME",
        "ANNUITY_TO_INCOME",
        "RISK_SEGMENT"
    ]
].dropna()


burden_df = burden_df[
    burden_df["CREDIT_TO_INCOME"]
    <= burden_df["CREDIT_TO_INCOME"].quantile(0.99)
]


burden_df = burden_df[
    burden_df["ANNUITY_TO_INCOME"]
    <= burden_df["ANNUITY_TO_INCOME"].quantile(0.99)
]


if len(burden_df) > 15000:

    burden_df = burden_df.sample(
        15000,
        random_state=42
    )


fig = px.scatter(
    burden_df,
    x="CREDIT_TO_INCOME",
    y="ANNUITY_TO_INCOME",
    color="RISK_SEGMENT",
    opacity=0.55,
    category_orders={
        "RISK_SEGMENT": segment_order
    },
    title="Credit-to-Income vs Annuity-to-Income"
)


fig.update_layout(
    xaxis_title="Credit-to-Income Ratio",
    yaxis_title="Annuity-to-Income Ratio"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SEGMENT PROFILE
# ============================================================

st.subheader("8. Risk Segment Profile")


profile_columns = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "EXTERNAL_SCORE",
    "AGE_YEARS"
]


available_profile_columns = [
    col
    for col in profile_columns
    if col in df.columns
]


profile_df = (
    df.groupby(
        "RISK_SEGMENT",
        observed=False
    )[available_profile_columns]
    .mean()
    .reindex(segment_order)
    .T
)


st.dataframe(
    profile_df.style.format(
        "{:,.2f}"
    ),
    use_container_width=True
)


# ============================================================
# HEATMAP
# ============================================================

st.subheader("9. Risk Segment Comparison Heatmap")


heatmap_df = (
    df.groupby(
        "RISK_SEGMENT",
        observed=False
    )[available_profile_columns]
    .mean()
    .reindex(segment_order)
)


# Normalize each variable to make different
# units comparable.

heatmap_normalized = (
    heatmap_df
    .copy()
)


for column in heatmap_normalized.columns:

    minimum = heatmap_normalized[column].min()
    maximum = heatmap_normalized[column].max()

    if maximum != minimum:

        heatmap_normalized[column] = (
            heatmap_normalized[column] - minimum
        ) / (
            maximum - minimum
        )


fig = px.imshow(
    heatmap_normalized.T,
    text_auto=".2f",
    aspect="auto",
    title="Normalized Risk Segment Profile"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# SUNBURST
# ============================================================

st.subheader("10. Hierarchical Customer Segmentation")


sunburst_columns = []


for col in [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE"
]:

    if col in df.columns:

        sunburst_columns.append(
            col
        )


sunburst_columns.append(
    "RISK_SEGMENT"
)


sun_df = df[
    sunburst_columns
].dropna()


if len(sun_df) > 100000:

    sun_df = sun_df.sample(
        100000,
        random_state=42
    )


sun_df["Customer Count"] = 1


grouped_sunburst = (
    sun_df
    .groupby(
        sunburst_columns
    )
    .size()
    .reset_index(
        name="Customers"
    )
)


fig = px.sunburst(
    grouped_sunburst,
    path=sunburst_columns,
    values="Customers",
    title="Income Type → Education → Risk Segment"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# AVERAGE CREDIT BY SEGMENT
# ============================================================

st.subheader("11. Credit Exposure by Segment")


credit_profile = (
    df.groupby(
        "RISK_SEGMENT",
        observed=False
    )
    .agg(
        Average_Credit=(
            "AMT_CREDIT",
            "mean"
        ),
        Median_Credit=(
            "AMT_CREDIT",
            "median"
        ),
        Average_Income=(
            "AMT_INCOME_TOTAL",
            "mean"
        ),
        Average_Annuity=(
            "AMT_ANNUITY",
            "mean"
        )
    )
    .reset_index()
)


fig = px.bar(
    credit_profile,
    x="RISK_SEGMENT",
    y="Average_Credit",
    text="Average_Credit",
    title="Average Credit Amount by Risk Segment"
)


fig.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.subheader("12. Customer Risk Recommendations")


recommendations = [
    {
        "Risk Segment": "High Risk",
        "Observed Characteristics": (
            "Higher affordability burden and/or lower "
            "external credit-score profile."
        ),
        "Recommended Action": (
            "Perform enhanced affordability review and "
            "closely monitor credit exposure."
        ),
        "Priority": "High"
    },
    {
        "Risk Segment": "Medium Risk",
        "Observed Characteristics": (
            "Moderate risk indicators across affordability "
            "or external-score measures."
        ),
        "Recommended Action": (
            "Apply standard risk monitoring and review "
            "customer affordability before larger exposures."
        ),
        "Priority": "Medium"
    },
    {
        "Risk Segment": "Low Risk",
        "Observed Characteristics": (
            "Lower observed affordability burden and "
            "relatively stronger risk indicators."
        ),
        "Recommended Action": (
            "Maintain normal monitoring while considering "
            "appropriate customer retention strategies."
        ),
        "Priority": "Low"
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
# AUTOMATIC OBSERVATIONS
# ============================================================

st.subheader("13. Automatic Segment Insights")


for _, row in segment_risk.iterrows():

    segment = row["RISK_SEGMENT"]

    customers = int(
        row["Customers"]
    )

    rate = row["Default Rate"]

    st.markdown(
        f"""
        **{segment}:** {customers:,} customers with an
        observed historical default rate of **{rate:.2f}%**.
        """
    )


# ============================================================
# SEGMENT DOWNLOAD
# ============================================================

st.subheader("14. Download Segmented Customer Data")


download_columns = [
    "SK_ID_CURR",
    "TARGET",
    "RISK_SCORE",
    "RISK_SEGMENT",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "EXTERNAL_SCORE"
]


download_columns = [
    col
    for col in download_columns
    if col in df.columns
]


segmented_data = df[
    download_columns
].copy()


csv_data = segmented_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Customer Risk Segmentation",
    data=csv_data,
    file_name="customer_risk_segmentation.csv",
    mime="text/csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

st.divider()

st.success(
    f"""
    **Customer Risk Segmentation Summary**

    The dashboard classified {total_customers:,} customers into
    Low, Medium and High Risk groups using transparent
    affordability and credit-profile rules.

    High Risk: {high_risk_customers:,} customers
    ({high_risk_percentage:.2f}%)

    Medium Risk: {medium_risk_customers:,} customers
    ({medium_risk_percentage:.2f}%)

    Low Risk: {low_risk_customers:,} customers
    ({low_risk_percentage:.2f}%)

    The resulting segments are intended for EDA, portfolio
    analysis and business decision support rather than
    machine-learning prediction.
    """
)