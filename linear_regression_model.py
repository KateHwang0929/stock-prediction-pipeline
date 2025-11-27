# linear_regression_model.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (np.sign(y_true) == np.sign(y_pred)).mean()

def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train.values if hasattr(y_train, "values") else y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    dir_acc = direction_accuracy(y_test, y_pred)
    print(f"[LinearRegression] RMSE^0.5={rmse**0.5:.6f}  MAE={mae:.6f}  DirAcc={dir_acc:.4f}")
    return model, rmse, mae, dir_acc
