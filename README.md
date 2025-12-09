# Stock Prediction Pipeline

This repository contains a full workflow for predicting stock price movements using a mix of machine-learning models and a basic LSTM network. The project handles data loading, feature engineering, model training, and evaluation, then compares all models in a single pipeline. It was created as part of seminar work under MIT Professor Mark Vogelsberger.

---

## Overview

The goal of this project is to explore how different algorithms perform on financial time-series data. 
1. The pipeline downloads OHLCV stock data with 'yfinance'
2. It builds a wide set of technical indicators
3. Then it creates lag features for sequence building
4. It trains regression models, classification models, and LSTM
5. Finally, itsaves evaluation results for comparison

All steps are executed through `main_enhanced.py`.

---

## Features

- **Automated preprocessing**
  - Cleans raw OHLCV data
  - Generates technical indicators 
  - Scales features using chronological MinMax scaling

- **Multiple model types**
  - *Regression*: Linear Regression, LSTM
  - *Classification*: Logistic Regression

- **Evaluation metrics**
  - Regression: RMSE, MAE, Direction Accuracy
  - Classification: Accuracy, F1 Score, ROC AUC

- **Automatic result saving**
  - All outputs are stored in the `results/` directory

---

## Output Files

The pipeline automatically generates:
- baseline_regression.csv
- classification_results.csv
- model_comparison.csv
- summary_report.csv

These files summarize each model’s performance and accuracy.

---
## Final Notes
This project was created for research and learning purposes and is absolutely not intended for financial advice!!

