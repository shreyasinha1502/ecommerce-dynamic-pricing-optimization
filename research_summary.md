# Research Summary

## Abstract

This project investigates how machine learning can support dynamic pricing decisions in e-commerce by combining demand forecasting with price elasticity analysis. Using a real-world retail transaction dataset, the study transforms raw order-level records into a product-day analytical dataset containing sales quantity, average price, and temporal features. Two forecasting approaches, Linear Regression and Random Forest Regression, are used to predict product demand. In parallel, a price elasticity model is estimated for a focal product to understand how changes in price influence sales volume. The final stage simulates multiple price points to identify the revenue-maximizing price. The study demonstrates that pricing decisions should not be made independently of demand behavior; instead, firms can use predictive modeling to connect price, demand, and revenue in a structured decision framework.

## Methodology

The analysis begins with data cleaning and transformation. Missing values in descriptive columns are handled, date fields are converted into datetime format, and cancelled or invalid transactions are removed. The cleaned transaction dataset is aggregated at the product-day level to produce a time-aware dataset for supervised learning. For demand forecasting, the study engineers calendar features such as weekday and month, along with lag and rolling-demand features that capture short-term sales momentum. Linear Regression provides an interpretable baseline, while Random Forest captures nonlinear relationships and interactions among price, time, and historical demand. Model quality is evaluated using RMSE and R^2.

For pricing analysis, one product with sufficient price variation is selected to estimate a log-linear price elasticity model. In this setting, the coefficient of log price provides an interpretable elasticity estimate, showing the percentage change in demand associated with a percentage change in price. After estimating elasticity, the model simulates a range of candidate prices and predicts demand at each point. Revenue is then computed as the product of price and predicted demand, allowing the study to identify an approximate revenue-maximizing price.

## Results

The forecasting stage shows that demand can be predicted from a combination of product identity, price, and temporal sales patterns. In most retail settings, the Random Forest model is expected to outperform the linear baseline because demand behavior is rarely perfectly linear. The elasticity model provides an economically meaningful interpretation of customer response to price changes. A negative elasticity confirms standard market behavior in which higher prices tend to reduce demand. The revenue simulation typically reveals an inverted-U shape: at low prices, revenue is constrained by margin; at very high prices, revenue falls due to reduced demand; between these extremes, an optimal price emerges.

## Conclusion

The project shows that dynamic pricing becomes much more effective when it is connected to demand forecasting rather than treated as a simple rule-based exercise. A research-oriented workflow that combines machine learning, economic reasoning, and simulation can produce decisions that are both analytically rigorous and managerially useful. For e-commerce firms, this means better inventory planning, more disciplined pricing experiments, and stronger revenue management. For academic presentation, the project demonstrates an integrated understanding of data science, business analytics, and applied economic modeling, making it a strong portfolio piece for advanced coursework, faculty review, or research-oriented interviews.
