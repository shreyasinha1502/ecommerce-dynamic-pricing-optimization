# Deployment Guide

This project can now be deployed as a Streamlit web app.

## Run locally as a web app

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How the app works

- the user uploads the retail dataset
- the app runs the full analysis pipeline
- the app displays forecasting, elasticity, revenue optimization, and charts

## Deployment files

- [app.py](C:/Users/hii/Documents/New%20project/app.py)
- [requirements.txt](C:/Users/hii/Documents/New%20project/requirements.txt)
- [dynamic_pricing_project.py](C:/Users/hii/Documents/New%20project/src/dynamic_pricing_project.py)

## Generic hosting setup

Use a Python hosting platform that supports Streamlit and set the start command to:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Why this deployment path is useful

- no hardcoded local dataset dependency
- easy to share with professors and recruiters
- interactive results instead of only script output
