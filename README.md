# Dynamic Pricing & Demand Forecasting for E-commerce using Machine Learning

This project is a research-oriented machine learning study on how e-commerce firms can use demand forecasting and dynamic pricing together to improve pricing decisions and maximize revenue.

The project is designed to be strong enough for:

- GitHub portfolio presentation
- professor review
- research-oriented coursework
- interviews in analytics, data science, and business strategy

## 1. Problem Definition

### Business Problem

E-commerce companies often use fixed or intuition-based prices even though demand changes over time. When the price is too high, customers may buy less. When the price is too low, the company may increase volume but lose potential revenue.

The central business problem is:

**How can a company forecast demand and then determine the price that maximizes expected revenue?**

### Why Dynamic Pricing and Demand Forecasting Matter

- Demand forecasting helps inventory planning, replenishment, and operational efficiency.
- Dynamic pricing helps firms adjust prices based on market behavior instead of using static rules.
- Combining both creates a stronger decision system than doing either one in isolation.

## 2. Dataset

This project is designed for the real-world **Online Retail** dataset.

Place one of the following files in [data/README.md](C:/Users/hii/Documents/New%20project/data/README.md):

- `data/Online Retail.xlsx`
- `data/online_retail.xlsx`
- `data/Online Retail.csv`
- `data/online_retail.csv`

Expected columns include:

- `StockCode`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `Description`
- `Country`

## 3. Project Structure

```text
New project/
|-- data/
|   |-- README.md
|-- notebooks/
|   |-- dynamic_pricing_walkthrough.ipynb
|-- outputs/
|   |-- figures/
|   |-- tables/
|-- src/
|   |-- __init__.py
|   |-- dynamic_pricing_project.py
|-- app.py
|-- DEPLOYMENT.md
|-- PROJECT_GUIDE.md
|-- README.md
|-- requirements.txt
|-- research_summary.md
|-- run_project.bat
```

## 4. Workflow

### STEP 1: Problem Definition

We answer two connected questions:

1. Can we predict product demand from historical behavior?
2. Given the demand response to price, what price maximizes revenue?

### STEP 2: Data Understanding and Cleaning

The project:

- loads the transaction dataset
- handles missing values
- converts date columns to datetime
- removes cancellations, returns, and invalid prices
- aggregates data at the daily product level

### STEP 3: Demand Forecasting Model

Two regression models are used:

- `LinearRegression`
- `RandomForestRegressor`

Evaluation metrics:

- `RMSE`
- `R^2`

### STEP 4: Price Elasticity Modeling

The project estimates a simple model of:

`Demand = f(Price, Seasonality)`

In simple terms, price elasticity tells us how much customer demand changes when price changes.

### STEP 5: Revenue Optimization

Revenue is defined as:

`Revenue = Price x Demand`

The script simulates multiple price points, predicts demand at each point, and identifies the price with the highest expected revenue.

### STEP 6: Visualization

The project generates:

- `Price vs Demand`
- `Price vs Revenue`
- `Actual vs Predicted Demand`
- `Daily Demand Over Time`

### STEP 7: Insights and Business Recommendations

The script exports insights to a text file and a markdown report.

### STEP 8: Research Summary

The research-oriented summary is available in [research_summary.md](C:/Users/hii/Documents/New%20project/research_summary.md).

## 5. How to Run

### Option A: Manual run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\dynamic_pricing_project.py
```

### Option B: One-click run

Double-click [run_project.bat](C:/Users/hii/Documents/New%20project/run_project.bat).

### Option C: Run as a web app

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Deployment notes are in [DEPLOYMENT.md](C:/Users/hii/Documents/New%20project/DEPLOYMENT.md).

## 6. Output Files

After running, check:

- `outputs/figures/`
- `outputs/tables/`

Important outputs:

- `eda_summary.csv`
- `daily_product_dataset.csv`
- `forecast_model_results.csv`
- `forecast_predictions.csv`
- `elasticity_product_data.csv`
- `revenue_optimization_table.csv`
- `project_insights.txt`
- `project_report.md`

## 7. Research Value

This project is more than a basic machine learning exercise. It demonstrates:

- business problem framing
- data cleaning and feature engineering
- predictive modeling
- economic interpretation of pricing behavior
- simulation-based decision support

## 8. Suggested Extensions

- include promotions and holiday effects
- add competitor pricing data
- estimate elasticity for more products or product categories
- optimize profit instead of only revenue
- deploy a dashboard for pricing managers
# ecommerce-dynamic-pricing-optimization
