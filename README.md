# Stock Prediction Pipeline

This repository contains a full workflow for predicting stock price movements using a mix of machine-learning models and a basic LSTM network. The project handles data loading, feature engineering, model training, and evaluation, then compares all models in a single pipeline.

---

## Overview

The goal of this project is to explore how different algorithms perform on financial time-series data.  
The pipeline:

- downloads OHLCV stock data with `yfinance`
- builds a wide set of technical indicators  
- creates lag features for sequence modeling  
- trains regression models, classification models, and an LSTM  
- saves evaluation results for comparison

All steps are executed through `main_enhanced.py`.

---

## 🧰 Features

- **Automated preprocessing**
  - Cleans raw OHLCV data
  - Generates technical indicators (SMA, EMA, RSI, MACD, Stochastic, ADX, Aroon, Bollinger Bands, OBV, etc.)
  - Scales features using chronological MinMax scaling

- **Multiple model types**
  - *Regression*: Linear Regression, LSTM
  - *Classification*: Logistic Regression, SVM, Random Forest, Gradient Boosting, AdaBoost, KNN

- **Evaluation metrics**
  - Regression: RMSE, MAE, Direction Accuracy
  - Classification: Accuracy, F1 Score, ROC AUC

- **Automatic result saving**
  - All outputs are stored in the `results/` directory

---

## 📁 Project Structure

