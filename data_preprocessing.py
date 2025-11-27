# data_preprocessing.py
"""
Enhanced Data Preprocessing Module for Stock Price Prediction

- Loads OHLCV from yfinance
- Engineers a broad set of technical features
- Provides train/test chronological split with proper scaling
- Avoids premature NumPy conversion to prevent broadcasting bugs
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


# Data Loading

def load_data(ticker: str, start_date: str = "2011-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} from {start_date}.")
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Adj Close, Volume
    df = df.dropna().copy()
    return df

# Helpers (keep everything as Pandas Series to preserve index alignment)

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = _safe_ratio(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    k = 100 * _safe_ratio((close - ll), (hh - ll))
    d = k.rolling(3).mean()
    return k, d

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _atr(high, low, close, n=1)  # 1-day TR for smoothing below
    atr = tr.rolling(n).mean()

    plus_di = 100 * _safe_ratio(plus_dm.rolling(n).mean(), atr)
    minus_di = 100 * _safe_ratio(minus_dm.rolling(n).mean(), atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    return adx

def _aroon(high: pd.Series, low: pd.Series, n: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
    rolling_high_idx = high.rolling(n).apply(lambda x: float(np.argmax(x)), raw=False)
    rolling_low_idx  = low.rolling(n).apply(lambda x: float(np.argmin(x)), raw=False)
    aroon_up = 100 * (n - 1 - rolling_high_idx) / (n - 1)
    aroon_dn = 100 * (n - 1 - rolling_low_idx) / (n - 1)
    aroon_osc = aroon_up - aroon_dn
    return aroon_up, aroon_dn, aroon_osc

# Feature Engineering

def add_features(df: pd.DataFrame, include_lags: bool = True, lag_periods: int = 10) -> pd.DataFrame:
    out = df.copy()
    print("[Feature Engineering] Starting comprehensive feature generation...")

    # Price-based indicators
    print("  → Adding price-based indicators (SMA, EMA, MACD)...")
    out["SMA_10"] = _sma(out["Close"], 10)
    out["SMA_20"] = _sma(out["Close"], 20)
    out["SMA_50"] = _sma(out["Close"], 50)
    out["EMA_10"] = _ema(out["Close"], 10)
    out["EMA_20"] = _ema(out["Close"], 20)
    out["EMA_50"] = _ema(out["Close"], 50)
    out["Returns"] = out["Close"].pct_change()
    out["LogReturns"] = np.log1p(out["Returns"])

    ema12 = _ema(out["Close"], 12)
    ema26 = _ema(out["Close"], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["MACD"] = macd
    out["MACD_Signal"] = signal
    out["MACD_Hist"] = macd - signal

    # Volume-based indicators
    print("  → Adding volume-based indicators (OBV, VROC, VWAP)...")
    out["OBV"] = _obv(out["Close"], out["Volume"])
    out["VROC_10"] = out["Volume"].pct_change(10)

    # VWAP (intraday proxy using daily OHLCV)
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    cum_pv = (typical_price * out["Volume"]).cumsum()
    cum_v = out["Volume"].cumsum()
    out["VWAP"] = _safe_ratio(cum_pv, cum_v)

    # Volatility measures
    print("  → Adding volatility measures (Bollinger Bands, ATR)...")
    bb_mid = _sma(out["Close"], 20)
    bb_std = out["Close"].rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    out["BB_Middle"] = bb_mid
    out["BB_Upper"] = bb_up
    out["BB_Lower"] = bb_dn
    out["BB_Width"] = _safe_ratio((bb_up - bb_dn), bb_mid)
    out["BB_PercentB"] = _safe_ratio((out["Close"] - bb_dn), (bb_up - bb_dn))

    out["ATR"] = _atr(out["High"], out["Low"], out["Close"], 14)
    out["ATR_Percent"] = _safe_ratio(out["ATR"], out["Close"])
    out["HL_Spread"] = _safe_ratio((out["High"] - out["Low"]), out["Close"])

    # Momentum
    print("  → Adding momentum indicators (RSI, Stochastic, ROC)...")
    out["RSI_14"] = _rsi(out["Close"], 14)
    k, d = _stochastic(out["High"], out["Low"], out["Close"], 14)
    out["Stoch_K"] = k
    out["Stoch_D"] = d
    out["ROC_10"] = out["Close"].pct_change(10)

    # Trend
    print("  → Adding trend indicators (ADX, Aroon)...")
    out["ADX_14"] = _adx(out["High"], out["Low"], out["Close"], 14)
    a_up, a_dn, a_osc = _aroon(out["High"], out["Low"], 25)
    out["Aroon_Up"] = a_up
    out["Aroon_Down"] = a_dn
    out["Aroon_Osc"] = a_osc

    # Price pattern features
    print("  → Adding price pattern features...")
    out["Close_to_SMA20"] = _safe_ratio((out["Close"] - out["SMA_20"]), out["SMA_20"])
    out["Close_to_SMA50"] = _safe_ratio((out["Close"] - out["SMA_50"]), out["SMA_50"])
    out["SMA_Cross"] = (out["SMA_10"] > out["SMA_20"]).astype(float)

    # Lagged features
    if include_lags:
        print(f"  → Adding {lag_periods} lagged closing prices...")
        for i in range(1, lag_periods + 1):
            out[f"Close_Lag_{i}"] = out["Close"].shift(i)

    # Target (next-day return)
    print("  → Creating target variable (next-day return)...")
    out["Return_Next"] = out["Returns"].shift(-1)

    # Clean
    print("  → Removing rows with missing values…")
    before = len(out)
    out = out.dropna().copy()
    after = len(out)
    print(f"  → Dropped {before - after} rows. Final: {after} rows × {out.shape[1]} cols\n")

    return out

# Canonical feature list for modeling
def get_feature_list(df: pd.DataFrame | None = None) -> List[str]:
    # If df is provided, infer features by excluding target columns.
    if df is not None:
        return [c for c in df.columns if c not in ("Return_Next",)]
    # Fallback—typical columns we created above
    base = [
        "SMA_10","SMA_20","SMA_50","EMA_10","EMA_20","EMA_50","Returns","LogReturns",
        "MACD","MACD_Signal","MACD_Hist",
        "OBV","VROC_10","VWAP",
        "BB_Middle","BB_Upper","BB_Lower","BB_Width","BB_PercentB",
        "ATR","ATR_Percent","HL_Spread",
        "RSI_14","Stoch_K","Stoch_D","ROC_10",
        "ADX_14","Aroon_Up","Aroon_Down","Aroon_Osc",
        "Close_to_SMA20","Close_to_SMA50","SMA_Cross"
    ] + [f"Close_Lag_{i}" for i in range(1, 11)]
    return base


# Split & Scale

def split_and_scale(df: pd.DataFrame, test_size: float = 0.2
                    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, MinMaxScaler, List[str]]:
    FEATURES = get_feature_list(df)
    TARGET = "Return_Next"

    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    print("\n[Split Summary]")
    print(f"  Full:  {df.index.min().date()} → {df.index.max().date()} ({len(df)})")
    print(f"  Train: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df)})")
    print(f"  Test:  {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df)})")
    print(f"  Features: {len(FEATURES)}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Empty train or test after split—check date ranges.")

    scaler = MinMaxScaler()
    train_df.loc[:, FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df.loc[:, FEATURES] = scaler.transform(test_df[FEATURES])

    X_train = train_df[FEATURES].to_numpy()
    X_test  = test_df[FEATURES].to_numpy()
    y_train = train_df[TARGET]        # keep as Series (aligned with dates)
    y_test  = test_df[TARGET]

    print("[Feature Scaling] done.")
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, scaler, FEATURES
