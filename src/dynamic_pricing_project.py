from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset_file() -> Path:
    candidates = [
        DATA_DIR / "Online Retail.xlsx",
        DATA_DIR / "online_retail.xlsx",
        DATA_DIR / "Online Retail.csv",
        DATA_DIR / "online_retail.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset not found. Place Online Retail.xlsx or Online Retail.csv inside the data/ folder."
    )


def load_dataset() -> pd.DataFrame:
    dataset_path = find_dataset_file()

    if dataset_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path, encoding="latin1")

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    required_cols = ["StockCode", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Normalize categorical columns to string so downstream encoders do not see mixed types.
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("Unknown").astype(str).str.strip()
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna(-1)
    if "Country" in df.columns:
        df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    if "InvoiceNo" in df.columns:
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]

    df = df.drop_duplicates()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.floor("D")

    print(f"Cleaned shape: {df.shape}")
    return df


def basic_eda(df: pd.DataFrame) -> Dict[str, float]:
    summary = {
        "transactions": int(len(df)),
        "unique_products": int(df["StockCode"].nunique()),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "total_units": float(df["Quantity"].sum()),
        "total_revenue": float(df["Revenue"].sum()),
        "average_price": float(df["UnitPrice"].mean()),
    }

    pd.DataFrame([summary]).to_csv(TABLE_DIR / "eda_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    daily_sales = df.groupby("Date")["Quantity"].sum().reset_index()
    sns.lineplot(data=daily_sales, x="Date", y="Quantity")
    plt.title("Daily Demand Over Time")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "daily_demand_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["UnitPrice"], bins=50, kde=True)
    plt.title("Distribution of Unit Prices")
    plt.xlabel("Unit Price")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "unit_price_distribution.png", dpi=300)
    plt.close()

    return summary


def build_daily_product_dataset(df: pd.DataFrame) -> pd.DataFrame:
    product_daily = (
        df.groupby(["Date", "StockCode"], as_index=False)
        .agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            Revenue=("Revenue", "sum"),
            Description=("Description", "first") if "Description" in df.columns else ("StockCode", "first"),
            Country=("Country", "first") if "Country" in df.columns else ("StockCode", "first"),
        )
        .sort_values(["StockCode", "Date"])
    )

    product_daily["day_of_week"] = product_daily["Date"].dt.dayofweek
    product_daily["month"] = product_daily["Date"].dt.month
    product_daily["week_of_year"] = product_daily["Date"].dt.isocalendar().week.astype(int)
    product_daily["is_weekend"] = (product_daily["day_of_week"] >= 5).astype(int)

    product_daily["lag_1"] = product_daily.groupby("StockCode")["Quantity"].shift(1)
    product_daily["lag_7"] = product_daily.groupby("StockCode")["Quantity"].shift(7)
    product_daily["rolling_mean_7"] = (
        product_daily.groupby("StockCode")["Quantity"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    )
    product_daily["rolling_std_7"] = (
        product_daily.groupby("StockCode")["Quantity"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    )

    product_daily = product_daily.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"])
    product_daily.to_csv(TABLE_DIR / "daily_product_dataset.csv", index=False)
    return product_daily


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date").reset_index(drop=True)
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def build_forecasting_pipeline(model_name: str) -> Pipeline:
    numeric_features = [
        "UnitPrice",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
    ]
    categorical_features = ["StockCode"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError("model_name must be 'linear_regression' or 'random_forest'")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_forecasting_models(product_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = time_split(product_daily)

    feature_cols = [
        "StockCode",
        "UnitPrice",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
    ]
    target_col = "Quantity"

    results = []
    predictions_df = test_df[["Date", "StockCode", "Quantity", "UnitPrice"]].copy()

    for model_name in ["linear_regression", "random_forest"]:
        pipeline = build_forecasting_pipeline(model_name)
        pipeline.fit(train_df[feature_cols], train_df[target_col])
        preds = pipeline.predict(test_df[feature_cols])
        preds = np.clip(preds, a_min=0, a_max=None)

        rmse = np.sqrt(mean_squared_error(test_df[target_col], preds))
        r2 = r2_score(test_df[target_col], preds)

        results.append(
            {
                "model": model_name,
                "rmse": rmse,
                "r2": r2,
            }
        )
        predictions_df[f"pred_{model_name}"] = preds

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(TABLE_DIR / "forecast_model_results.csv", index=False)
    predictions_df.to_csv(TABLE_DIR / "forecast_predictions.csv", index=False)
    return results_df, predictions_df


def select_focal_product(product_daily: pd.DataFrame) -> str:
    product_stats = (
        product_daily.groupby("StockCode")
        .agg(
            observations=("Quantity", "size"),
            price_std=("UnitPrice", "std"),
            quantity_mean=("Quantity", "mean"),
        )
        .fillna(0)
        .reset_index()
    )

    eligible = product_stats[
        (product_stats["observations"] >= 20)
        & (product_stats["price_std"] > 0.05)
        & (product_stats["quantity_mean"] > 0)
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No product has enough daily observations and price variation for elasticity modeling."
        )

    eligible["score"] = eligible["observations"] * eligible["price_std"]
    return eligible.sort_values("score", ascending=False).iloc[0]["StockCode"]


def build_elasticity_dataset(product_daily: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    product_df = product_daily[product_daily["StockCode"] == stock_code].copy()
    product_df = product_df[product_df["UnitPrice"] > 0]
    product_df = product_df[product_df["Quantity"] > 0]

    product_df["log_quantity"] = np.log(product_df["Quantity"])
    product_df["log_price"] = np.log(product_df["UnitPrice"])
    return product_df


def fit_price_elasticity_model(product_df: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, float]:
    X = product_df[["log_price", "day_of_week", "month", "is_weekend"]]
    y = product_df["log_quantity"]

    model = LinearRegression()
    model.fit(X, y)
    product_df["predicted_log_quantity"] = model.predict(X)
    product_df["predicted_quantity"] = np.exp(product_df["predicted_log_quantity"])

    elasticity = float(model.coef_[0])
    product_df.to_csv(TABLE_DIR / "elasticity_product_data.csv", index=False)
    return model, product_df, elasticity


def simulate_revenue_curve(
    elasticity_model: LinearRegression,
    product_df: pd.DataFrame,
    stock_code: str,
) -> pd.DataFrame:
    min_price = product_df["UnitPrice"].quantile(0.05)
    max_price = product_df["UnitPrice"].quantile(0.95)
    price_grid = np.linspace(min_price, max_price, 60)

    scenario_base = {
        "day_of_week": int(product_df["day_of_week"].mode().iloc[0]),
        "month": int(product_df["month"].mode().iloc[0]),
        "is_weekend": int(product_df["is_weekend"].mode().iloc[0]),
    }

    simulation = pd.DataFrame({"UnitPrice": price_grid})
    simulation["log_price"] = np.log(simulation["UnitPrice"])
    simulation["day_of_week"] = scenario_base["day_of_week"]
    simulation["month"] = scenario_base["month"]
    simulation["is_weekend"] = scenario_base["is_weekend"]

    predicted_log_quantity = elasticity_model.predict(
        simulation[["log_price", "day_of_week", "month", "is_weekend"]]
    )
    simulation["PredictedDemand"] = np.exp(predicted_log_quantity)
    simulation["Revenue"] = simulation["UnitPrice"] * simulation["PredictedDemand"]
    simulation["StockCode"] = stock_code

    simulation.to_csv(TABLE_DIR / "revenue_optimization_table.csv", index=False)
    return simulation


def create_visualizations(
    product_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=product_df, x="UnitPrice", y="Quantity", alpha=0.7)
    sns.lineplot(data=simulation_df, x="UnitPrice", y="PredictedDemand", color="crimson")
    plt.title("Price vs Demand")
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_vs_demand.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=simulation_df, x="UnitPrice", y="Revenue", color="darkgreen")
    plt.title("Price vs Revenue")
    plt.xlabel("Price")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_vs_revenue.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=predictions_df,
        x="Quantity",
        y=f"pred_{best_model_name}",
        alpha=0.6,
    )
    max_value = max(
        predictions_df["Quantity"].max(),
        predictions_df[f"pred_{best_model_name}"].max(),
    )
    plt.plot([0, max_value], [0, max_value], linestyle="--", color="black")
    plt.title("Actual vs Predicted Demand")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted_demand.png", dpi=300)
    plt.close()


def build_insights_text(
    eda_summary: Dict[str, float],
    model_results: pd.DataFrame,
    elasticity: float,
    optimal_row: pd.Series,
    stock_code: str,
) -> str:
    best_model = model_results.iloc[0]

    insights = [
        "1. The retail dataset contains real customer purchase behavior over time, making it suitable for both forecasting and pricing analysis.",
        f"2. The best forecasting model was {best_model['model']} with RMSE={best_model['rmse']:.2f} and R^2={best_model['r2']:.3f}, showing that demand is predictable from price and time-based features.",
        f"3. Estimated price elasticity for product {stock_code} is {elasticity:.3f}. A negative value means demand falls when price rises, which is the expected market behavior.",
        f"4. Revenue simulation suggests an approximate optimal price of {optimal_row['UnitPrice']:.2f}, where predicted demand is {optimal_row['PredictedDemand']:.2f} units and expected revenue is {optimal_row['Revenue']:.2f}.",
        f"5. Across the full dataset, the business processed {eda_summary['transactions']} cleaned transactions, sold about {eda_summary['total_units']:.0f} units, and generated total observed revenue of {eda_summary['total_revenue']:.2f}.",
    ]

    recommendations = [
        "Use the forecasting model for weekly inventory and replenishment planning.",
        "Update prices selectively for products with measurable elasticity rather than using blanket price changes.",
        "Test simulated price points through controlled experiments before full deployment.",
        "Monitor competitor activity, promotions, and stock-outs to improve future pricing models.",
        "Integrate the model into a dashboard so category managers can compare predicted demand and expected revenue in near real time.",
    ]

    text = ["Project Insights", "================", ""]
    text.extend(insights)
    text.extend(["", "Business Recommendations", "========================", ""])
    text.extend([f"- {item}" for item in recommendations])
    return "\n".join(text)


def save_insights_report(insights_text: str) -> None:
    (TABLE_DIR / "project_insights.txt").write_text(insights_text, encoding="utf-8")


def save_markdown_report(
    eda_summary: Dict[str, float],
    model_results: pd.DataFrame,
    elasticity: float,
    optimal_row: pd.Series,
    stock_code: str,
) -> None:
    best_model = model_results.iloc[0]
    report = f"""# Project Report

## Abstract

This study applies machine learning to connect demand forecasting and dynamic pricing in e-commerce. Using a real retail transaction dataset, the project estimates future demand, measures price sensitivity, and simulates revenue under alternative price points.

## Dataset and Cleaning

- Cleaned transactions: {eda_summary['transactions']}
- Unique products: {eda_summary['unique_products']}
- Date range: {eda_summary['date_start']} to {eda_summary['date_end']}
- Total observed units sold: {eda_summary['total_units']:.0f}
- Total observed revenue: {eda_summary['total_revenue']:.2f}

## Forecasting Results

- Best model: {best_model['model']}
- RMSE: {best_model['rmse']:.2f}
- R^2: {best_model['r2']:.3f}

## Elasticity Results

- Focal product: {stock_code}
- Estimated price elasticity: {elasticity:.3f}

## Revenue Optimization

- Optimal simulated price: {optimal_row['UnitPrice']:.2f}
- Predicted demand at optimal price: {optimal_row['PredictedDemand']:.2f}
- Maximum expected revenue: {optimal_row['Revenue']:.2f}

## Conclusion

The analysis shows that pricing can be improved by combining demand prediction with elasticity-based simulation. This gives firms a practical way to move from descriptive analytics to pricing decision support.
"""
    (TABLE_DIR / "project_report.md").write_text(report, encoding="utf-8")


def run_analysis(raw_df: pd.DataFrame) -> Dict[str, object]:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    clean_df = clean_data(raw_df)
    eda_summary = basic_eda(clean_df)

    product_daily = build_daily_product_dataset(clean_df)
    model_results, predictions_df = evaluate_forecasting_models(product_daily)
    best_model_name = model_results.iloc[0]["model"]

    stock_code = select_focal_product(product_daily)
    elasticity_df = build_elasticity_dataset(product_daily, stock_code)
    elasticity_model, elasticity_product_df, elasticity = fit_price_elasticity_model(elasticity_df)

    simulation_df = simulate_revenue_curve(elasticity_model, elasticity_product_df, stock_code)
    optimal_row = simulation_df.loc[simulation_df["Revenue"].idxmax()]

    create_visualizations(elasticity_product_df, simulation_df, predictions_df, best_model_name)

    insights_text = build_insights_text(
        eda_summary=eda_summary,
        model_results=model_results,
        elasticity=elasticity,
        optimal_row=optimal_row,
        stock_code=stock_code,
    )
    save_insights_report(insights_text)
    save_markdown_report(
        eda_summary=eda_summary,
        model_results=model_results,
        elasticity=elasticity,
        optimal_row=optimal_row,
        stock_code=stock_code,
    )

    return {
        "clean_df": clean_df,
        "eda_summary": eda_summary,
        "product_daily": product_daily,
        "model_results": model_results,
        "predictions_df": predictions_df,
        "best_model_name": best_model_name,
        "stock_code": stock_code,
        "elasticity_df": elasticity_product_df,
        "elasticity": elasticity,
        "simulation_df": simulation_df,
        "optimal_row": optimal_row,
        "insights_text": insights_text,
    }


def main() -> None:
    raw_df = load_dataset()
    results = run_analysis(raw_df)

    print("\nAnalysis complete.")
    print(results["model_results"])
    print(f"\nSelected focal product for elasticity: {results['stock_code']}")
    print(f"Estimated elasticity: {results['elasticity']:.3f}")
    print(
        "Optimal simulated price: "
        f"{results['optimal_row']['UnitPrice']:.2f} | "
        f"Predicted demand: {results['optimal_row']['PredictedDemand']:.2f} | "
        f"Revenue: {results['optimal_row']['Revenue']:.2f}"
    )


if __name__ == "__main__":
    main()
