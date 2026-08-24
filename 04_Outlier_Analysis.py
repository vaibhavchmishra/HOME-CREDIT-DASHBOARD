import streamlit as st, plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils.data_loader import require_dataset
st.title('Outlier Analysis'); df=require_dataset('application_train'); nums=df.select_dtypes('number').columns.tolist(); col=st.selectbox('Numeric variable',nums); s=df[col].dropna(); q1,q3=s.quantile([.25,.75]); iqr=q3-q1; out=((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum(); st.metric('IQR outliers',f'{out:,}'); st.plotly_chart(px.box(df,y=col),use_container_width=True)



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Home Credit - Outlier Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATA PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "application_train.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "application_train.csv was not found. "
        "Please place it inside the data/ folder."
    )

    st.stop()


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# Remove TARGET from general outlier analysis
analysis_columns = [
    col
    for col in numeric_columns
    if col != "TARGET"
]


if not analysis_columns:

    st.warning(
        "No numerical columns are available "
        "for outlier analysis."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Outlier Settings")


method = st.sidebar.selectbox(
    "Outlier Detection Method",
    [
        "IQR",
        "Z-Score"
    ]
)


if method == "IQR":

    multiplier = st.sidebar.slider(
        "IQR Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

else:

    z_threshold = st.sidebar.slider(
        "Z-Score Threshold",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )


selected_column = st.sidebar.selectbox(
    "Select Numerical Column",
    analysis_columns
)


# ============================================================
# IQR OUTLIER FUNCTION
# ============================================================

def calculate_iqr_outliers(series, multiplier=1.5):

    clean_series = series.dropna()

    q1 = clean_series.quantile(0.25)

    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr

    upper_bound = q3 + multiplier * iqr

    mask = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    return (
        mask,
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound
    )


# ============================================================
# Z-SCORE FUNCTION
# ============================================================

def calculate_zscore_outliers(
    series,
    threshold=3.0
):

    mean_value = series.mean()

    std_value = series.std()

    if std_value == 0 or pd.isna(std_value):

        z_scores = pd.Series(
            0,
            index=series.index
        )

    else:

        z_scores = (
            series - mean_value
        ) / std_value

    mask = (
        z_scores.abs()
        > threshold
    )

    return (
        mask,
        z_scores,
        mean_value,
        std_value
    )


# ============================================================
# CALCULATE OUTLIER PROFILE
# ============================================================

outlier_results = []


for column in analysis_columns:

    series = df[column]

    if method == "IQR":

        (
            mask,
            q1,
            q3,
            iqr,
            lower_bound,
            upper_bound
        ) = calculate_iqr_outliers(
            series,
            multiplier
        )

        outlier_count = int(
            mask.sum()
        )

        lower_outliers = int(
            (series < lower_bound).sum()
        )

        upper_outliers = int(
            (series > upper_bound).sum()
        )

    else:

        (
            mask,
            z_scores,
            mean_value,
            std_value
        ) = calculate_zscore_outliers(
            series,
            z_threshold
        )

        outlier_count = int(
            mask.sum()
        )

        lower_outliers = int(
            (z_scores < -z_threshold).sum()
        )

        upper_outliers = int(
            (z_scores > z_threshold).sum()
        )

        lower_bound = (
            mean_value
            - z_threshold * std_value
        )

        upper_bound = (
            mean_value
            + z_threshold * std_value
        )

    valid_count = series.notna().sum()

    outlier_percentage = (
        outlier_count
        / valid_count
        * 100
        if valid_count > 0
        else 0
    )

    outlier_results.append({
        "Column": column,
        "Outlier Count": outlier_count,
        "Outlier Percentage": outlier_percentage,
        "Lower Outliers": lower_outliers,
        "Upper Outliers": upper_outliers,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound
    })


outlier_profile = pd.DataFrame(
    outlier_results
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Outlier Summary")


total_outliers = int(
    outlier_profile["Outlier Count"].sum()
)


columns_with_outliers = int(
    (
        outlier_profile["Outlier Count"] > 0
    ).sum()
)


highest_outlier_column = (
    outlier_profile
    .sort_values(
        "Outlier Count",
        ascending=False
    )
    .iloc[0]["Column"]
)


average_outlier_rate = (
    outlier_profile[
        "Outlier Percentage"
    ].mean()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Outlier Observations",
        f"{total_outliers:,}"
    )


with col2:

    st.metric(
        "Columns with Outliers",
        f"{columns_with_outliers:,}"
    )


with col3:

    st.metric(
        "Highest Outlier Column",
        highest_outlier_column
    )


with col4:

    st.metric(
        "Average Outlier Rate",
        f"{average_outlier_rate:.2f}%"
    )


st.divider()


# ============================================================
# CHART 1
# OUTLIER COUNT BY COLUMN
# ============================================================

st.subheader("1️⃣ Outlier Count by Column")


outlier_count_data = (
    outlier_profile
    .sort_values(
        "Outlier Count",
        ascending=True
    )
)


fig_outlier_count = px.bar(
    outlier_count_data,
    x="Outlier Count",
    y="Column",
    orientation="h",
    text="Outlier Count",
    title="Number of Outliers by Numerical Column"
)


fig_outlier_count.update_traces(
    textposition="outside"
)


fig_outlier_count.update_layout(
    height=max(
        500,
        len(outlier_count_data) * 22
    ),
    xaxis_title="Outlier Count",
    yaxis_title="Column"
)


st.plotly_chart(
    fig_outlier_count,
    use_container_width=True
)


# ============================================================
# CHART 2
# OUTLIER PERCENTAGE
# ============================================================

st.subheader("2️⃣ Outlier Percentage by Column")


outlier_percentage_data = (
    outlier_profile
    .sort_values(
        "Outlier Percentage",
        ascending=True
    )
)


fig_outlier_percentage = px.bar(
    outlier_percentage_data,
    x="Outlier Percentage",
    y="Column",
    orientation="h",
    text="Outlier Percentage",
    title="Outlier Percentage by Numerical Column"
)


fig_outlier_percentage.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig_outlier_percentage.update_layout(
    height=max(
        500,
        len(outlier_percentage_data) * 22
    ),
    xaxis_title="Outlier Percentage (%)",
    yaxis_title="Column"
)


st.plotly_chart(
    fig_outlier_percentage,
    use_container_width=True
)


# ============================================================
# CHART 3
# LOWER VS UPPER OUTLIERS
# ============================================================

st.subheader("3️⃣ Lower vs Upper Outliers")


direction_data = outlier_profile[
    [
        "Column",
        "Lower Outliers",
        "Upper Outliers"
    ]
].copy()


direction_data = direction_data.sort_values(
    "Lower Outliers",
    ascending=True
)


fig_direction = px.bar(
    direction_data,
    x="Column",
    y=[
        "Lower Outliers",
        "Upper Outliers"
    ],
    barmode="group",
    title="Lower and Upper Outlier Counts"
)


fig_direction.update_layout(
    height=550,
    xaxis_title="Column",
    yaxis_title="Outlier Count"
)


st.plotly_chart(
    fig_direction,
    use_container_width=True
)


# ============================================================
# CHART 4 — SELECTED COLUMN BOX PLOT
# ============================================================

st.subheader("4️⃣ Box Plot — Selected Variable")


box_data = df[
    [selected_column]
].dropna()


fig_box = px.box(
    box_data,
    y=selected_column,
    points="outliers",
    title=f"Box Plot of {selected_column}"
)


fig_box.update_layout(
    height=500,
    yaxis_title=selected_column
)


st.plotly_chart(
    fig_box,
    use_container_width=True
)


# ============================================================
# CHART 5 — DISTRIBUTION
# ============================================================

st.subheader("5️⃣ Distribution — Selected Variable")


distribution_data = df[
    selected_column
].dropna()


fig_distribution = px.histogram(
    distribution_data,
    x=selected_column,
    nbins=50,
    title=f"Distribution of {selected_column}"
)


fig_distribution.update_layout(
    height=500,
    xaxis_title=selected_column,
    yaxis_title="Frequency"
)


st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ============================================================
# SELECTED COLUMN STATISTICS
# ============================================================

st.subheader("📋 Selected Variable Statistics")


selected_series = df[
    selected_column
].dropna()


if method == "IQR":

    (
        selected_mask,
        selected_q1,
        selected_q3,
        selected_iqr,
        selected_lower,
        selected_upper
    ) = calculate_iqr_outliers(
        df[selected_column],
        multiplier
    )

    statistics = pd.DataFrame({
        "Metric": [
            "Count",
            "Mean",
            "Median",
            "Minimum",
            "Q1",
            "Q3",
            "Maximum",
            "IQR",
            "Lower Bound",
            "Upper Bound",
            "Outlier Count"
        ],
        "Value": [
            selected_series.count(),
            selected_series.mean(),
            selected_series.median(),
            selected_series.min(),
            selected_q1,
            selected_q3,
            selected_series.max(),
            selected_iqr,
            selected_lower,
            selected_upper,
            selected_mask.sum()
        ]
    })


else:

    (
        selected_mask,
        selected_zscores,
        selected_mean,
        selected_std
    ) = calculate_zscore_outliers(
        df[selected_column],
        z_threshold
    )

    selected_lower = (
        selected_mean
        - z_threshold * selected_std
    )

    selected_upper = (
        selected_mean
        + z_threshold * selected_std
    )

    statistics = pd.DataFrame({
        "Metric": [
            "Count",
            "Mean",
            "Median",
            "Minimum",
            "Maximum",
            "Standard Deviation",
            "Z-Score Threshold",
            "Lower Bound",
            "Upper Bound",
            "Outlier Count"
        ],
        "Value": [
            selected_series.count(),
            selected_series.mean(),
            selected_series.median(),
            selected_series.min(),
            selected_series.max(),
            selected_series.std(),
            z_threshold,
            selected_lower,
            selected_upper,
            selected_mask.sum()
        ]
    })


statistics["Value"] = statistics[
    "Value"
].round(4)


st.dataframe(
    statistics,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHART 6 — INCOME VS CREDIT
# ============================================================

if (
    "AMT_INCOME_TOTAL" in df.columns
    and "AMT_CREDIT" in df.columns
):

    st.subheader("6️⃣ Income vs Credit — Potential Extreme Observations")


    scatter_data = df[
        [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT"
        ]
    ].dropna()


    # Limit extreme values only for visualization
    income_limit = scatter_data[
        "AMT_INCOME_TOTAL"
    ].quantile(0.99)


    credit_limit = scatter_data[
        "AMT_CREDIT"
    ].quantile(0.99)


    scatter_data = scatter_data[
        (
            scatter_data["AMT_INCOME_TOTAL"]
            <= income_limit
        )
        &
        (
            scatter_data["AMT_CREDIT"]
            <= credit_limit
        )
    ]


    fig_scatter = px.scatter(
        scatter_data,
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        opacity=0.45,
        title="Annual Income vs Credit Amount"
    )


    fig_scatter.update_layout(
        height=550,
        xaxis_title="Annual Income",
        yaxis_title="Credit Amount"
    )


    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# ============================================================
# CHART 7 — TARGET VS OUTLIERS
# ============================================================

if "TARGET" in df.columns:

    st.subheader("7️⃣ Outlier Rate by TARGET")


    target_outlier_data = []


    for target_value in sorted(
        df["TARGET"].dropna().unique()
    ):

        target_subset = df[
            df["TARGET"] == target_value
        ]


        total_target_rows = len(
            target_subset
        )


        outlier_row_count = 0


        for column in analysis_columns:

            series = target_subset[
                column
            ]


            if method == "IQR":

                (
                    mask,
                    _,
                    _,
                    _,
                    _,
                    _
                ) = calculate_iqr_outliers(
                    series,
                    multiplier
                )

            else:

                (
                    mask,
                    _,
                    _,
                    _
                ) = calculate_zscore_outliers(
                    series,
                    z_threshold
                )


            outlier_row_count += int(
                mask.sum()
            )


        rate = (
            outlier_row_count
            / (
                total_target_rows
                * len(analysis_columns)
            )
            * 100
            if total_target_rows > 0
            else 0
        )


        label = (
            "No Payment Difficulty"
            if target_value == 0
            else "Payment Difficulty"
        )


        target_outlier_data.append({
            "TARGET": label,
            "Outlier Rate": rate
        })


    target_outlier_df = pd.DataFrame(
        target_outlier_data
    )


    fig_target_outlier = px.bar(
        target_outlier_df,
        x="TARGET",
        y="Outlier Rate",
        text="Outlier Rate",
        title="Average Outlier Rate by TARGET"
    )


    fig_target_outlier.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig_target_outlier.update_layout(
        height=450,
        yaxis_title="Outlier Rate (%)"
    )


    st.plotly_chart(
        fig_target_outlier,
        use_container_width=True
    )


# ============================================================
# OUTLIER DETAIL TABLE
# ============================================================

st.subheader("📋 Complete Outlier Analysis Report")


display_outliers = outlier_profile.copy()


display_outliers[
    "Outlier Percentage"
] = display_outliers[
    "Outlier Percentage"
].round(2)


display_outliers[
    "Lower Bound"
] = display_outliers[
    "Lower Bound"
].round(2)


display_outliers[
    "Upper Bound"
] = display_outliers[
    "Upper Bound"
].round(2)


st.dataframe(
    display_outliers,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.subheader("⬇️ Download Outlier Report")


csv_report = outlier_profile.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Outlier Analysis CSV",
    data=csv_report,
    file_name="home_credit_outlier_report.csv",
    mime="text/csv"
)