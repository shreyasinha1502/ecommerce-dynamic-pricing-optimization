from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dynamic_pricing_project import FIGURE_DIR, run_analysis


st.set_page_config(
    page_title="Dynamic Pricing and Demand Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(uploaded_file.read()))
    return pd.read_csv(BytesIO(uploaded_file.read()), encoding="latin1")


st.title("Dynamic Pricing and Demand Forecasting for E-commerce")
st.write(
    "Upload the Online Retail dataset to run the full research pipeline: cleaning, forecasting, "
    "price elasticity modeling, revenue optimization, and visual analysis."
)

with st.sidebar:
    st.header("Run Analysis")
    uploaded_file = st.file_uploader(
        "Upload dataset (.xlsx or .csv)",
        type=["xlsx", "csv"],
    )
    run_button = st.button("Run Project", type="primary", use_container_width=True)

st.markdown(
    """
### What this app does
- Forecasts product demand using machine learning
- Estimates price elasticity for a focal product
- Simulates price points to identify the revenue-maximizing price
- Exports figures and result tables inside the project `outputs/` folder
"""
)

if run_button:
    if uploaded_file is None:
        st.error("Please upload the dataset before running the analysis.")
    else:
        with st.spinner("Running the full pricing and forecasting workflow..."):
            raw_df = load_uploaded_dataset(uploaded_file)
            results = run_analysis(raw_df)

        summary = results["eda_summary"]
        best_model = results["model_results"].iloc[0]
        optimal_row = results["optimal_row"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions", f"{summary['transactions']:,}")
        col2.metric("Unique Products", f"{summary['unique_products']:,}")
        col3.metric("Best RMSE", f"{best_model['rmse']:.2f}")
        col4.metric("Elasticity", f"{results['elasticity']:.3f}")

        st.subheader("Forecasting Results")
        st.dataframe(results["model_results"], use_container_width=True)

        st.subheader("Revenue Optimization")
        left, right = st.columns(2)
        left.metric("Optimal Price", f"{optimal_row['UnitPrice']:.2f}")
        right.metric("Maximum Expected Revenue", f"{optimal_row['Revenue']:.2f}")

        st.subheader("Charts")
        chart_cols = st.columns(2)
        chart_cols[0].image(str(FIGURE_DIR / "price_vs_demand.png"), caption="Price vs Demand")
        chart_cols[1].image(str(FIGURE_DIR / "price_vs_revenue.png"), caption="Price vs Revenue")
        st.image(
            str(FIGURE_DIR / "actual_vs_predicted_demand.png"),
            caption="Actual vs Predicted Demand",
        )

        st.subheader("Key Tables")
        st.write("Revenue simulation")
        st.dataframe(results["simulation_df"].head(15), use_container_width=True)
        st.write("Forecast sample")
        st.dataframe(results["predictions_df"].head(15), use_container_width=True)

        st.subheader("Insights and Recommendations")
        st.text(results["insights_text"])

        st.success(
            "Analysis complete. Figures and tables were also saved to the local outputs folder."
        )
else:
    st.info("Upload the dataset and click 'Run Project' to generate the results.")
