# lstm_model.py
from __future__ import annotations
import numpy as np
from typing import Tuple

# Keras/TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def _make_sequences(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.asarray(Xs), np.asarray(ys)

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (np.sign(y_true) == np.sign(y_pred)).mean()

def run_lstm(X_train, X_test, y_train, y_test, window: int = 10):
    # Build sequences on train, then transform test with same window
    Xtr_seq, ytr_seq = _make_sequences(X_train, y_train.values if hasattr(y_train, "values") else y_train, window)
    Xte_seq, yte_seq = _make_sequences(X_test,  y_test.values  if hasattr(y_test, "values") else y_test,  window)

    model = Sequential([
        LSTM(32, input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    model.fit(Xtr_seq, ytr_seq, validation_split=0.2, epochs=50, batch_size=64, callbacks=[es], verbose=0)

    y_pred = model.predict(Xte_seq, verbose=0).ravel()
    mse = np.mean((yte_seq - y_pred) ** 2)
    mae = np.mean(np.abs(yte_seq - y_pred))
    dir_acc = direction_accuracy(yte_seq, y_pred)
    print(f"[LSTM] RMSE^0.5={mse**0.5:.6f}  MAE={mae:.6f}  DirAcc={dir_acc:.4f}")

    return model, mse, mae, dir_acc
