"""
NSE Confidence Engine — ML Trainer
Trains a LightGBM model on historical NSE data to predict 5-day forward returns.
Runs entirely free, no GPU needed. ~5 min on Render/Railway free tier.

Usage:
  python train.py           # train fresh
  python train.py --eval    # evaluate saved model
"""

import argparse, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from ta import momentum, trend, volatility

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found — falling back to RandomForest")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler


# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOLS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "WIPRO.NS","HINDUNILVR.NS","LT.NS","SBIN.NS","BAJFINANCE.NS",
    "ADANIPORTS.NS","TATAMOTORS.NS","MARUTI.NS","AXISBANK.NS",
    "SUNPHARMA.NS","TITAN.NS","TECHM.NS","HCLTECH.NS",
    "ULTRACEMCO.NS","ONGC.NS","NTPC.NS","POWERGRID.NS",
    "COALINDIA.NS","NESTLEIND.NS","GRASIM.NS",
]
FORWARD_DAYS = 5       # predict 5-day return
BUY_THRESHOLD  =  2.5  # % return → label 1 (BUY signal)
SELL_THRESHOLD = -2.0  # % return → label -1 (SELL signal)
MODEL_PATH = Path("models/lgb_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
META_PATH = Path("models/meta.json")


# ─── Feature builder ──────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    out = pd.DataFrame(index=df.index)

    # Returns
    for d in [1, 3, 5, 10, 20]:
        out[f"ret_{d}d"] = close.pct_change(d) * 100

    # Trend
    for p in [10, 20, 50, 100]:
        out[f"ema_{p}"] = trend.EMAIndicator(close, p).ema_indicator()
    out["ema_cross_20_50"] = out["ema_20"] - out["ema_50"]
    out["price_vs_ema50"]  = (close - out["ema_50"]) / out["ema_50"] * 100

    macd = trend.MACD(close)
    out["macd_hist"] = macd.macd_diff()
    out["macd_sig"]  = macd.macd_signal()

    adx = trend.ADXIndicator(high, low, close, 14)
    out["adx"]      = adx.adx()
    out["adx_pos"]  = adx.adx_pos()
    out["adx_neg"]  = adx.adx_neg()

    # Momentum
    out["rsi"]       = momentum.RSIIndicator(close, 14).rsi()
    out["rsi_9"]     = momentum.RSIIndicator(close, 9).rsi()
    stoch = momentum.StochasticOscillator(high, low, close, 14)
    out["stoch_k"]   = stoch.stoch()
    out["stoch_d"]   = stoch.stoch_signal()

    # Volatility
    bb = volatility.BollingerBands(close, 20, 2)
    out["bb_pct"]    = bb.bollinger_pband()
    out["bb_width"]  = bb.bollinger_wband()
    atr = volatility.AverageTrueRange(high, low, close, 14)
    out["atr_pct"]   = atr.average_true_range() / close * 100

    # Volume
    out["vol_ratio"] = vol / vol.rolling(20).mean()
    out["obv_slope"] = (np.sign(close.diff()) * vol).cumsum().diff(5)

    # Drop EMA cols used only internally
    drop = [f"ema_{p}" for p in [10, 20, 50, 100]]
    out = out.drop(columns=drop, errors="ignore")

    return out


def label_rows(close: pd.Series) -> pd.Series:
    fwd = close.shift(-FORWARD_DAYS)
    ret = (fwd - close) / close * 100
    labels = pd.Series(0, index=close.index, name="label")
    labels[ret >= BUY_THRESHOLD]   = 1
    labels[ret <= SELL_THRESHOLD]  = -1
    return labels


# ─── Data pipeline ────────────────────────────────────────────────────────────

def fetch_all(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        print(f"  Fetching {sym}…", end=" ")
        try:
            df = yf.Ticker(sym).history(period="5y", interval="1d", auto_adjust=True)
            if len(df) < 200:
                print("skip (too short)")
                continue
            feats = build_features(df)
            feats["label"] = label_rows(df["Close"].astype(float))
            feats["symbol"] = sym
            frames.append(feats)
            print(f"✓ ({len(feats)} rows)")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.4)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna()
    # Remove last FORWARD_DAYS rows per symbol (labels would use future data)
    combined = combined.groupby("symbol").apply(lambda g: g.iloc[:-FORWARD_DAYS]).reset_index(drop=True)
    return combined


# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*55}")
    print("  NSE Confidence Engine — Model Trainer")
    print(f"{'='*55}\n")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading 5yr historical data…")
    df = fetch_all(SYMBOLS)
    print(f"\nTotal training rows: {len(df):,}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    feature_cols = [c for c in df.columns if c not in ["label", "symbol"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if HAS_LGB:
        print("Training LightGBM…")
        ds_train = lgb.Dataset(X_train, label=y_train + 1)  # 0/1/2 for lgb
        ds_val   = lgb.Dataset(X_val,   label=y_val + 1, reference=ds_train)
        params = {
            "objective":     "multiclass",
            "num_class":     3,
            "metric":        "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves":    63,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":  5,
            "verbose":       -1,
        }
        model = lgb.train(
            params, ds_train,
            num_boost_round=500,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        joblib.dump(model, MODEL_PATH)
        scaler = None
    else:
        print("Training RandomForest (fallback)…")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                       n_jobs=-1, random_state=42)
        model.fit(X_train, y_train + 1)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    # Evaluate
    if HAS_LGB:
        preds = np.argmax(model.predict(X_val), axis=1)
    else:
        preds = model.predict(scaler.transform(X_val))
    
    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(y_val + 1, preds)
    print(f"\nValidation Accuracy: {acc:.3f}")
    print(classification_report(y_val + 1, preds, target_names=["SELL", "HOLD", "BUY"]))

    # Save metadata
    meta = {
        "trained_at": datetime.now().isoformat(),
        "symbols":    SYMBOLS,
        "features":   feature_cols,
        "accuracy":   round(acc, 4),
        "rows_used":  len(df),
        "framework":  "lightgbm" if HAS_LGB else "randomforest",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Meta  saved → {META_PATH}\n")


# ─── Inference helper (used by engine.py when ML model is present) ────────────

def predict_proba(features_dict: dict) -> dict | None:
    """
    Given a feature dict from engine.compute_features(), return ML probabilities.
    Returns None if no model saved yet.
    """
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None
    try:
        meta = json.loads(META_PATH.read_text())
        feature_cols = meta["features"]
        model = joblib.load(MODEL_PATH)
        x = np.array([[features_dict.get(c, 0) for c in feature_cols]], dtype=np.float32)

        if meta["framework"] == "lightgbm":
            probs = model.predict(x)[0]   # [sell, hold, buy]
        else:
            scaler = joblib.load(SCALER_PATH)
            x = scaler.transform(x)
            probs = model.predict_proba(x)[0]

        return {
            "sell_prob": round(float(probs[0]) * 100, 1),
            "hold_prob": round(float(probs[1]) * 100, 1),
            "buy_prob":  round(float(probs[2]) * 100, 1),
        }
    except Exception as e:
        print(f"ML predict error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    train()
