# main_enhanced.py
"""
Enhanced Stock Price Prediction Pipeline (concise)
- Loads MSFT data
- Engineers features
- Compares Linear Regression, LSTM, and several classifiers
- Prints / saves results
"""

from __future__ import annotations
import os
from time import time
from datetime import datetime
import pandas as pd

from data_preprocessing import load_data, add_features, split_and_scale, get_feature_list
from linear_regression_model import run_linear_regression
from lstm_model import run_lstm
from classification_models import run_all_classifiers

def print_section(title: str):
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)

def save_df(df: pd.DataFrame, name: str):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", name)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")

def main():
    print("Stock Prediciton Pipeline")

    t0 = time()

    # STEP 1: DATA
    print_section("STEP 1: DATA LOADING & PREPROCESSING")
    print("Loading MSFT stock data…")
    df = load_data("MSFT", start_date="2011-01-01")
    print(f"✓ Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}\n")

    print("Engineering features…")
    df = add_features(df)

    print("Splitting & scaling…")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

 
    # STEP 2: REGRESSION

    print_section("STEP 2: Baseline Regression Models")
    print("Linear Regression…")
    lr_model, lr_mse, lr_mae, lr_dir = run_linear_regression(X_train, X_test, y_train, y_test)

    print("\nLSTM (Basic)…")
    lstm, lstm_mse, lstm_mae, lstm_dir = run_lstm(X_train, X_test, y_train, y_test, window=10)

    baseline_df = pd.DataFrame([
        {"Model": "Linear Regression", "Type": "Regression", "RMSE": lr_mse**0.5, "MAE": lr_mae, "Direction_Acc": lr_dir},
        {"Model": "LSTM (Basic)",     "Type": "Regression", "RMSE": lstm_mse**0.5, "MAE": lstm_mae, "Direction_Acc": lstm_dir},
    ])
    save_df(baseline_df, "baseline_regression.csv")

    # STEP 3: CLASSIFICATION

    print_section("STEP 3: CLASSIFICATION MODELS")
    clf_df = run_all_classifiers(X_train, X_test, y_train, y_test)
    save_df(clf_df, "classification_results.csv")


    # STEP 4: COMPARISON
    print_section("STEP 4: COMPARISON & SUMMARY")
    rows = []
    for _, r in baseline_df.iterrows():
        rows.append({"Model": r["Model"], "Type": r["Type"], "Primary_Metric": r["Direction_Acc"], "Metric_Name": "Direction_Acc",
                     "RMSE": r["RMSE"], "MAE": r["MAE"]})
    for _, r in clf_df.iterrows():
        rows.append({"Model": r["Model"], "Type": "Classification", "Primary_Metric": r["Accuracy"], "Metric_Name": "Accuracy",
                     "RMSE": None, "MAE": None})

    comp = pd.DataFrame(rows).sort_values("Primary_Metric", ascending=False).reset_index(drop=True)
    print("🏆 FINAL RANKINGS (sorted by Accuracy / Direction_Acc)")
    print(comp.round(4).to_string(index=False))
    save_df(comp, "model_comparison.csv")

    best = comp.iloc[0]
    print("\n🎯 Best Overall:", best["Model"], f"({best['Metric_Name']}={best['Primary_Metric']:.4f})")


    # FINAL
    total = time() - t0
    summary = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_rows": len(df),
        "n_features": len(get_feature_list(df)),
        "best_model": best["Model"],
        "best_metric": best["Metric_Name"],
        "best_score": best["Primary_Metric"],
        "lr_diracc": baseline_df.iloc[0]["Direction_Acc"],
        "lstm_diracc": baseline_df.iloc[1]["Direction_Acc"],
        "runtime_sec": total
    }])
    save_df(summary, "summary_report.csv")

    print(f"Pipeline complete {total:.2f}s\n")

if __name__ == "__main__":
    main()
