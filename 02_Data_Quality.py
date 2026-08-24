import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils.data_loader import require_dataset
from utils.preprocessing import missing_summary
st.title('Data Quality'); df=require_dataset('application_train')
a,b,c=st.columns(3); a.metric('Rows',f'{len(df):,}'); b.metric('Columns',len(df.columns)); c.metric('Duplicate rows',int(df.duplicated().sum()))
st.dataframe(df.dtypes.astype(str).rename('dtype').to_frame(),use_container_width=True); st.dataframe(missing_summary(df),use_container_width=True)






# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Home Credit - Data Quality",
    page_icon="🔎",
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

DATA_PATH = BASE_DIR / "data" / "application_train.csv"


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
        "Please place the file inside the data/ folder."
    )

    st.stop()


# ============================================================
# BASIC INFORMATION
# ============================================================

rows = df.shape[0]

columns = df.shape[1]

total_cells = rows * columns

missing_cells = df.isna().sum().sum()

complete_cells = total_cells - missing_cells

missing_percentage = (
    missing_cells / total_cells * 100
    if total_cells > 0
    else 0
)

complete_percentage = 100 - missing_percentage

duplicate_rows = df.duplicated().sum()

duplicate_percentage = (
    duplicate_rows / rows * 100
    if rows > 0
    else 0
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Data Quality Summary")


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{rows:,}"
    )

with col2:

    st.metric(
        "Columns",
        f"{columns:,}"
    )

with col3:

    st.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )


col5, col6, col7, col8 = st.columns(4)

with col5:

    st.metric(
        "Complete Cells",
        f"{complete_percentage:.2f}%"
    )

with col6:

    st.metric(
        "Missing Cells %",
        f"{missing_percentage:.2f}%"
    )

with col7:

    st.metric(
        "Duplicate %",
        f"{duplicate_percentage:.2f}%"
    )

with col8:

    memory_usage = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    st.metric(
        "Memory Usage",
        f"{memory_usage:.2f} MB"
    )


st.divider()


# ============================================================
# CREATE QUALITY PROFILE
# ============================================================

quality = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Count": df.isna().sum().values,
    "Missing Percentage": (
        df.isna().mean().values * 100
    ),
    "Non-Missing Count": (
        df.notna().sum().values
    ),
    "Unique Values": (
        df.nunique(dropna=True).values
    )
})


quality["Completeness Percentage"] = (
    100 - quality["Missing Percentage"]
)


quality["Duplicate Values"] = (
    rows - quality["Unique Values"]
)


# ============================================================
# CHART 1
# MISSING VALUES BY COLUMN
# ============================================================

st.subheader("1️⃣ Missing Values by Column")


missing_chart = quality[
    quality["Missing Count"] > 0
].sort_values(
    "Missing Count",
    ascending=True
)


if not missing_chart.empty:

    fig_missing = px.bar(
        missing_chart,
        x="Missing Count",
        y="Column",
        orientation="h",
        text="Missing Count",
        title="Number of Missing Values by Column"
    )

    fig_missing.update_traces(
        textposition="outside"
    )

    fig_missing.update_layout(
        height=max(
            450,
            len(missing_chart) * 25
        ),
        xaxis_title="Missing Values",
        yaxis_title="Column"
    )

    st.plotly_chart(
        fig_missing,
        use_container_width=True
    )

else:

    st.success(
        "No missing values were found."
    )


# ============================================================
# CHART 2
# MISSING VALUE PERCENTAGE
# ============================================================

st.subheader("2️⃣ Missing Value Percentage")


missing_percentage_chart = quality[
    quality["Missing Percentage"] > 0
].sort_values(
    "Missing Percentage",
    ascending=True
)


if not missing_percentage_chart.empty:

    fig_missing_pct = px.bar(
        missing_percentage_chart,
        x="Missing Percentage",
        y="Column",
        orientation="h",
        text="Missing Percentage",
        title="Missing Value Percentage by Column"
    )

    fig_missing_pct.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_missing_pct.update_layout(
        height=max(
            450,
            len(missing_percentage_chart) * 25
        ),
        xaxis_title="Missing Values (%)",
        yaxis_title="Column"
    )

    st.plotly_chart(
        fig_missing_pct,
        use_container_width=True
    )


# ============================================================
# CHART 3
# COMPLETE VS MISSING
# ============================================================

st.subheader("3️⃣ Overall Data Completeness")


completeness_data = pd.DataFrame({
    "Status": [
        "Complete",
        "Missing"
    ],
    "Cells": [
        complete_cells,
        missing_cells
    ]
})


fig_completeness = px.pie(
    completeness_data,
    names="Status",
    values="Cells",
    hole=0.55,
    title="Complete vs Missing Cells"
)


fig_completeness.update_traces(
    textinfo="percent+label"
)


fig_completeness.update_layout(
    height=450
)


st.plotly_chart(
    fig_completeness,
    use_container_width=True
)


# ============================================================
# CHART 4
# DATA TYPE DISTRIBUTION
# ============================================================

st.subheader("4️⃣ Data Type Distribution")


dtype_data = (
    df.dtypes
    .astype(str)
    .value_counts()
    .reset_index()
)


dtype_data.columns = [
    "Data Type",
    "Column Count"
]


fig_dtype = px.bar(
    dtype_data,
    x="Data Type",
    y="Column Count",
    text="Column Count",
    title="Columns by Data Type"
)


fig_dtype.update_traces(
    textposition="outside"
)


fig_dtype.update_layout(
    height=450
)


st.plotly_chart(
    fig_dtype,
    use_container_width=True
)


# ============================================================
# CHART 5
# UNIQUE VALUES BY COLUMN
# ============================================================

st.subheader("5️⃣ Unique Values by Column")


unique_data = quality.sort_values(
    "Unique Values",
    ascending=True
)


fig_unique = px.bar(
    unique_data,
    x="Unique Values",
    y="Column",
    orientation="h",
    title="Number of Unique Values per Column"
)


fig_unique.update_layout(
    height=max(
        500,
        len(unique_data) * 22
    ),
    xaxis_title="Unique Values",
    yaxis_title="Column"
)


st.plotly_chart(
    fig_unique,
    use_container_width=True
)


# ============================================================
# CHART 6
# MISSING VALUES BY DATA TYPE
# ============================================================

st.subheader("6️⃣ Missing Values by Data Type")


missing_by_dtype = (
    quality
    .groupby("Data Type")["Missing Count"]
    .sum()
    .reset_index()
)


fig_missing_dtype = px.bar(
    missing_by_dtype,
    x="Data Type",
    y="Missing Count",
    text="Missing Count",
    title="Missing Cells by Data Type"
)


fig_missing_dtype.update_traces(
    textposition="outside"
)


fig_missing_dtype.update_layout(
    height=450
)


st.plotly_chart(
    fig_missing_dtype,
    use_container_width=True
)


# ============================================================
# CHART 7
# TARGET DISTRIBUTION
# ============================================================

if "TARGET" in df.columns:

    st.subheader("7️⃣ TARGET Distribution")


    target_data = (
        df["TARGET"]
        .value_counts(dropna=False)
        .reset_index()
    )


    target_data.columns = [
        "TARGET",
        "Count"
    ]


    target_data["Status"] = target_data[
        "TARGET"
    ].map({
        0: "No Payment Difficulty",
        1: "Payment Difficulty"
    })


    target_data["Status"] = target_data[
        "Status"
    ].fillna("Missing")


    fig_target = px.pie(
        target_data,
        names="Status",
        values="Count",
        hole=0.55,
        title="TARGET Distribution"
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
# CHART 8
# COLUMN COMPLETENESS
# ============================================================

st.subheader("8️⃣ Column Completeness")


completeness_chart = quality.sort_values(
    "Completeness Percentage",
    ascending=True
)


fig_completeness_column = px.bar(
    completeness_chart,
    x="Completeness Percentage",
    y="Column",
    orientation="h",
    text="Completeness Percentage",
    title="Completeness Percentage by Column"
)


fig_completeness_column.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig_completeness_column.update_layout(
    height=max(
        500,
        len(completeness_chart) * 22
    ),
    xaxis_title="Completeness (%)",
    yaxis_title="Column",
    xaxis_range=[0, 100]
)


st.plotly_chart(
    fig_completeness_column,
    use_container_width=True
)


# ============================================================
# QUALITY TABLE
# ============================================================

st.subheader("📋 Detailed Data Quality Report")


display_quality = quality.copy()


display_quality[
    "Missing Percentage"
] = display_quality[
    "Missing Percentage"
].round(2)


display_quality[
    "Completeness Percentage"
] = display_quality[
    "Completeness Percentage"
].round(2)


st.dataframe(
    display_quality,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MISSING VALUE THRESHOLD ANALYSIS
# ============================================================

st.subheader("⚠️ Missing Value Severity")


threshold = st.slider(
    "Select missing-value threshold (%)",
    min_value=0,
    max_value=100,
    value=40,
    step=5
)


high_missing = quality[
    quality["Missing Percentage"] >= threshold
].sort_values(
    "Missing Percentage",
    ascending=False
)


if high_missing.empty:

    st.success(
        f"No columns have missing values >= {threshold}%."
    )

else:

    st.warning(
        f"{len(high_missing)} column(s) have "
        f"missing values >= {threshold}%."
    )

    st.dataframe(
        high_missing[
            [
                "Column",
                "Missing Count",
                "Missing Percentage",
                "Completeness Percentage"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NUMERIC SUMMARY
# ============================================================

st.subheader("📈 Numeric Data Summary")


numeric_columns = df.select_dtypes(
    include=np.number
).columns


if len(numeric_columns) > 0:

    numeric_summary = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
    )


    numeric_summary = numeric_summary.rename(
        columns={
            "index": "Column"
        }
    )


    st.dataframe(
        numeric_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

st.subheader("🔤 Categorical Data Summary")


categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns


if len(categorical_columns) > 0:

    categorical_summary = pd.DataFrame({
        "Column": categorical_columns,
        "Unique Values": [
            df[column].nunique(dropna=True)
            for column in categorical_columns
        ],
        "Missing Values": [
            df[column].isna().sum()
            for column in categorical_columns
        ],
        "Missing %": [
            df[column].isna().mean() * 100
            for column in categorical_columns
        ]
    })


    categorical_summary[
        "Missing %"
    ] = categorical_summary[
        "Missing %"
    ].round(2)


    st.dataframe(
        categorical_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD QUALITY REPORT
# ============================================================

st.subheader("⬇️ Download Data Quality Report")


csv_report = quality.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Quality Report CSV",
    data=csv_report,
    file_name="home_credit_data_quality_report.csv",
    mime="text/csv"
)