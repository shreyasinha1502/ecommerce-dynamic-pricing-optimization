# Project Guide

This file helps you explain the project clearly in a viva, classroom presentation, or interview.

## Short Project Pitch

This project combines demand forecasting and dynamic pricing for e-commerce. It uses a real-world retail dataset to estimate future demand, measure how sensitive demand is to price, and simulate alternative prices to identify the revenue-maximizing price point.

## What Makes It Strong

- It starts with a business problem.
- It uses real transactional data.
- It combines forecasting with pricing analytics.
- It converts model outputs into business decisions.

## Best Presentation Flow

1. Start with [README.md](C:/Users/hii/Documents/New%20project/README.md) and explain the business problem.
2. Use [research_summary.md](C:/Users/hii/Documents/New%20project/research_summary.md) for the academic framing.
3. Show [dynamic_pricing_project.py](C:/Users/hii/Documents/New%20project/src/dynamic_pricing_project.py) and explain the workflow stages.
4. Show generated charts from `outputs/figures/`.
5. Show `forecast_model_results.csv` and `revenue_optimization_table.csv`.
6. End with managerial implications and future extensions.

## Good Answers for Common Questions

### Why did you use regression?

Because demand is a continuous target variable, so regression is a natural modeling choice.

### Why use both Linear Regression and Random Forest?

Linear Regression gives an interpretable baseline, while Random Forest can capture nonlinear demand behavior.

### What is price elasticity?

It measures how sensitive demand is to price changes.

### Why optimize revenue and not only demand?

Because higher demand does not always mean higher revenue. A price decrease can increase units sold but still reduce total revenue.

### What are the main limitations?

- competitor prices are not included
- promotions are not explicitly modeled
- elasticity is estimated for one focal product
- the project optimizes revenue, not profit

## Strong Closing Statement

This project shows how machine learning can be used not only to predict demand, but also to support real pricing decisions through simulation and economic interpretation.
