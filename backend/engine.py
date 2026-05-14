"""
NSE Confidence Engine — Core Signal Processor
Fetches Indian market data from free sources, runs ML signals,
outputs confidence scores for BUY / HOLD / SELL decisions.
"""

import os, json, time, asyncio, logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta import momentum, trend, volatility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

# ─── NSE Watchlist (add/remove tickers freely) ────────────────────────────────
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "BAJFINANCE.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "TATAMOTORS.NS", "MARUTI.NS",
    "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS", "NESTLEIND.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "HCLTECH.NS",
    "ULTRACEMCO.NS", "GRASIM.NS",
]

# ─── Feature Engineering ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    if len(close) < 60:
        return {}

    # Trend
    ema_20  = trend.EMAIndicator(close, 20).ema_indicator()
    ema_50  = trend.EMAIndicator(close, 50).ema_indicator()
    macd_i  = trend.MACD(close)
    adx_i   = trend.ADXIndicator(high, low, close, 14)

    # Momentum
    rsi     = momentum.RSIIndicator(close, 14).rsi()
    stoch   = momentum.StochasticOscillator(high, low, close, 14)

    # Volatility
    bb      = volatility.BollingerBands(close, 20, 2)
    atr_i   = volatility.AverageTrueRange(high, low, close, 14)

    # Volume signals
    obv     = (np.sign(close.diff()) * vol).cumsum()

    latest = -1
    c  = float(close.iloc[latest])
    features = {
        # Price context
        "price":        c,
        "ret_1d":       float((c - close.iloc[-2]) / close.iloc[-2] * 100),
        "ret_5d":       float((c - close.iloc[-5]) / close.iloc[-5] * 100),
        "ret_20d":      float((c - close.iloc[-20]) / close.iloc[-20] * 100),

        # Trend
        "ema_gap":      float((c - ema_50.iloc[latest]) / ema_50.iloc[latest] * 100),
        "ema_cross":    float(ema_20.iloc[latest] - ema_50.iloc[latest]),
        "macd_hist":    float(macd_i.macd_diff().iloc[latest]),
        "adx":          float(adx_i.adx().iloc[latest]),
        "adx_pos":      float(adx_i.adx_pos().iloc[latest]),
        "adx_neg":      float(adx_i.adx_neg().iloc[latest]),

        # Momentum
        "rsi":          float(rsi.iloc[latest]),
        "stoch_k":      float(stoch.stoch().iloc[latest]),
        "stoch_d":      float(stoch.stoch_signal().iloc[latest]),

        # Volatility / bands
        "bb_pct":       float(bb.bollinger_pband().iloc[latest]),
        "bb_width":     float(bb.bollinger_wband().iloc[latest]),
        "atr_pct":      float(atr_i.average_true_range().iloc[latest] / c * 100),

        # Volume
        "vol_20avg":    float(vol.rolling(20).mean().iloc[latest]),
        "vol_ratio":    float(vol.iloc[latest] / vol.rolling(20).mean().iloc[latest]),
        "obv_slope":    float(np.polyfit(range(10), obv.iloc[-10:].values, 1)[0]),
    }
    return features


def rule_based_score(f: dict) -> dict:
    """Fast deterministic signal scoring (0-100) on each axis."""
    scores = {}

    # ── TREND (0–100) ────────────────────────────────────────────────────────
    t = 50.0
    if f["ema_cross"] > 0:     t += 10
    else:                       t -= 10
    t += np.clip(f["ema_gap"] * 4, -20, 20)
    if f["macd_hist"] > 0:     t += 10
    else:                       t -= 10
    if f["adx"] > 25:          t += 5 * (1 if f["adx_pos"] > f["adx_neg"] else -1)
    scores["trend"] = np.clip(t, 0, 100)

    # ── MOMENTUM (0–100) ─────────────────────────────────────────────────────
    m = 50.0
    rsi = f["rsi"]
    if rsi < 30:   m += 25   # Oversold bounce
    elif rsi < 45: m += 10
    elif rsi > 70: m -= 20   # Overbought
    elif rsi > 60: m -= 5
    sk, sd = f["stoch_k"], f["stoch_d"]
    if sk < 20 and sk > sd:  m += 10
    if sk > 80:               m -= 10
    scores["momentum"] = np.clip(m, 0, 100)

    # ── VOLATILITY (0–100, higher = calmer / safer entry) ────────────────────
    v = 50.0
    v -= np.clip(f["atr_pct"] * 5, 0, 30)          # High ATR → riskier
    v += np.clip((1 - f["bb_width"] / 10) * 20, -20, 20)
    if 0.2 < f["bb_pct"] < 0.8: v += 10            # Mid-band = clean
    scores["volatility"] = np.clip(v, 0, 100)

    # ── VOLUME (0–100) ───────────────────────────────────────────────────────
    vl = 50.0
    vl += np.clip((f["vol_ratio"] - 1) * 20, -20, 20)
    if f["obv_slope"] > 0:  vl += 10
    else:                    vl -= 10
    scores["volume"] = np.clip(vl, 0, 100)

    # ── COMPOSITE (weighted average) ─────────────────────────────────────────
    weights = {"trend": 0.35, "momentum": 0.30, "volatility": 0.20, "volume": 0.15}
    composite = sum(scores[k] * weights[k] for k in weights)
    scores["composite"] = np.clip(composite, 0, 100)

    return scores


def derive_signal(scores: dict, features: dict) -> dict:
    """BUY / HOLD / SELL decision + entry/exit guidance."""
    c = scores["composite"]
    rsi = features.get("rsi", 50)
    atr_pct = features.get("atr_pct", 1)
    price = features.get("price", 0)

    # Signal
    if c >= 65:   action = "BUY"
    elif c <= 38: action = "SELL"
    else:         action = "HOLD"

    # Overbought override
    if rsi > 75 and action == "BUY":
        action = "HOLD"

    # ATR-based targets
    sl_pct  = max(1.5, atr_pct * 1.5)
    tp1_pct = max(2.5, atr_pct * 2.5)
    tp2_pct = max(5.0, atr_pct * 4.5)

    entry = price
    sl    = round(entry * (1 - sl_pct / 100), 2)
    tp1   = round(entry * (1 + tp1_pct / 100), 2)
    tp2   = round(entry * (1 + tp2_pct / 100), 2)
    rr    = round(tp1_pct / sl_pct, 2)

    confidence_label = (
        "Very High" if c >= 75 else
        "High"      if c >= 60 else
        "Moderate"  if c >= 45 else
        "Low"       if c >= 30 else
        "Very Low"
    )

    return {
        "action":     action,
        "confidence": round(float(c), 1),
        "label":      confidence_label,
        "entry":      round(entry, 2),
        "stop_loss":  sl,
        "target_1":   tp1,
        "target_2":   tp2,
        "risk_reward": rr,
        "atr_pct":    round(atr_pct, 2),
    }


# ─── Main Scan ────────────────────────────────────────────────────────────────

def scan_ticker(symbol: str) -> dict | None:
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="6mo", interval="1d", auto_adjust=True)
        if df is None or len(df) < 65:
            return None

        info = tk.fast_info
        features = compute_features(df)
        if not features:
            return None

        scores = rule_based_score(features)
        signal = derive_signal(scores, features)

        name = symbol.replace(".NS", "")
        try:
            name = tk.info.get("shortName", name)
        except Exception:
            pass

        return {
            "symbol":    symbol,
            "name":      name,
            "timestamp": datetime.now().isoformat(),
            "price":     features["price"],
            "ret_1d":    round(features["ret_1d"], 2),
            "ret_5d":    round(features["ret_5d"], 2),
            "scores":    {k: round(v, 1) for k, v in scores.items()},
            "signal":    signal,
        }

    except Exception as e:
        log.warning(f"Error scanning {symbol}: {e}")
        return None


def run_scan(symbols: list[str] | None = None) -> list[dict]:
    symbols = symbols or WATCHLIST
    log.info(f"Scanning {len(symbols)} symbols…")
    results = []
    for sym in symbols:
        r = scan_ticker(sym)
        if r:
            results.append(r)
        time.sleep(0.3)   # polite rate limit

    results.sort(key=lambda x: x["scores"]["composite"], reverse=True)
    log.info(f"Scan complete — {len(results)} signals")
    return results


def save_results(results: list[dict], path="data/latest.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"updated": datetime.now().isoformat(), "signals": results}, f, indent=2)
    log.info(f"Saved → {path}")


if __name__ == "__main__":
    import os, json
    from pathlib import Path

    # Download ML model from Hugging Face if available
    hf_token    = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")
    if hf_token and hf_username:
        try:
            from huggingface_hub import hf_hub_download
            Path("models").mkdir(exist_ok=True)
            for fname in ["lgb_model.pkl", "meta.json"]:
                try:
                    hf_hub_download(
                        repo_id=f"{hf_username}/nse-model",
                        filename=fname,
                        repo_type="model",
                        token=hf_token,
                        local_dir="models",
                    )
                    log.info(f"Downloaded {fname} from HuggingFace")
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"HF download skipped: {e}")

    results = run_scan()

    # Save to data/latest.json as before
    save_results(results)

    # Also save to docs/signals.json for GitHub Pages
    Path("docs").mkdir(exist_ok=True)
    with open("docs/signals.json", "w") as f:
        json.dump({
            "updated": datetime.now().isoformat(),
            "signals": results
        }, f, indent=2)
    log.info("Saved → docs/signals.json")
