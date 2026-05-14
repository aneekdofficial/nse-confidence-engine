# NSE Confidence Engine — Complete Launch Guide

A free, self-hosted real-time Indian stock market signal system.
No paid APIs. No GPU. No card required for hosting.

---

## What this does

- Fetches live + historical NSE data from Yahoo Finance (free, no key)
- Runs 20+ technical indicators: RSI, MACD, Bollinger Bands, ADX, ATR, OBV, Stochastic
- Scores each stock on Trend / Momentum / Volatility / Volume axes
- Combines into a 0–100 Confidence Score
- Issues BUY / HOLD / SELL with Entry, Stop Loss, Target 1, Target 2, R:R ratio
- Optional: train a LightGBM ML model on 5 years of NSE history to enhance signals
- Live dashboard with SSE real-time updates
- Rescans automatically every 15 minutes during market hours

---

## Stack (all free)

| Component       | Tool                          | Cost  |
|-----------------|-------------------------------|-------|
| Data            | yfinance (Yahoo Finance)      | Free  |
| ML              | LightGBM + scikit-learn       | Free  |
| Backend         | FastAPI + uvicorn             | Free  |
| Frontend        | Vanilla HTML/JS               | Free  |
| Hosting         | Render.com or Railway.app     | Free  |
| CI/CD           | GitHub Actions                | Free  |
| Domain          | render.onrender.com (subdomain)| Free |

---

## Quick Start (Local)

### 1. Clone & setup

```bash
git clone https://github.com/YOUR_USERNAME/nse-confidence-engine
cd nse-confidence-engine
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run a quick scan

```bash
python backend/engine.py
```

Output:
```
────────────────────────────────────────────────────
  TOP BUY SIGNALS  (8 found)
────────────────────────────────────────────────────
  TCS.NS               ₹3,842.50  conf= 72.4%  SL=₹3,765.00  TP1=₹3,960.00  RR=1.78
  INFY.NS              ₹1,612.00  conf= 68.1%  SL=₹1,580.00  TP1=₹1,680.00  RR=2.10
  ...
```

### 3. Start the API server

```bash
uvicorn backend.server:app --reload --port 8000
```

Open: http://localhost:8000

### 4. (Optional) Train the ML model

```bash
python backend/train.py
```

This downloads 5 years of NSE data (~25 symbols), engineers features,
trains LightGBM, and saves the model to `models/lgb_model.pkl`.
Takes ~5–8 minutes. No GPU needed.

---

## Deploy FREE on Render.com (Recommended)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "initial"
gh repo create nse-confidence-engine --public
git push origin main
```

### Step 2 — Create a Render account

Go to https://render.com → Sign up with GitHub (no card required)

### Step 3 — New Web Service

1. Click "New" → "Web Service"
2. Connect your GitHub repo
3. Render auto-detects `render.yaml`
4. Click "Create Web Service"

**Free tier settings:**
- Plan: Free
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

### Step 4 — Add persistent disk (for data/ folder)

In Render dashboard → your service → "Disks" tab → Add disk:
- Name: `data`
- Mount path: `/opt/render/project/src/data`
- Size: 1 GB (free)

### Step 5 — Deploy

Click "Manual Deploy" → "Deploy latest commit"

Your live URL: `https://nse-confidence-engine.onrender.com`

> **Note:** Render free tier spins down after 15 min of inactivity.
> Use https://cron-job.org (free) to ping `/health` every 10 min to keep it alive.

---

## Deploy FREE on Railway.app (Alternative)

### Step 1 — Create account

Go to https://railway.app → Login with GitHub (no card required)
Railway gives you $5 free credit/month — more than enough.

### Step 2 — New Project

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

Or via dashboard: "New Project" → "Deploy from GitHub repo"

Railway reads `railway.toml` automatically.

### Step 3 — Open

```bash
railway open
```

---

## Deploy FREE on Koyeb (No sleep, 2 free instances)

1. Go to https://www.koyeb.com → Sign up (no card)
2. New App → GitHub → Select repo
3. Builder: Buildpack
4. Run command: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
5. Port: 8000
6. Deploy

Koyeb's free tier does NOT sleep — best for 24/7 live signals.

---

## Keep the server alive (important for Render free tier)

Sign up at https://cron-job.org (free):
- URL: `https://YOUR-APP.onrender.com/health`
- Schedule: every 10 minutes
- This prevents the server from sleeping

---

## Auto-train on deploy (GitHub Actions)

The `.github/workflows/deploy.yml` runs tests on every push.
To auto-train the ML model on first deploy, add this step to the workflow:

```yaml
- name: Train model
  run: python backend/train.py
  if: github.event_name == 'push'
```

Or add a Render build command: `pip install -r requirements.txt && python backend/train.py`

---

## API Endpoints

| Endpoint         | Method | Description                            |
|------------------|--------|----------------------------------------|
| `/`              | GET    | Dashboard (HTML)                       |
| `/signals`       | GET    | All signals as JSON                    |
| `/signals?action=BUY` | GET | Filter by BUY/HOLD/SELL             |
| `/signals?min_conf=65` | GET | Filter by minimum confidence        |
| `/scan`          | POST   | Trigger a fresh scan                   |
| `/stream`        | GET    | SSE stream (live updates)              |
| `/health`        | GET    | Health check                           |

---

## Customize the watchlist

Edit `WATCHLIST` in `backend/engine.py`:

```python
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    # Add any NSE symbol with .NS suffix
    "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS",
    # Nifty Midcap
    "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
]
```

---

## Signal interpretation

| Score | Action | What to do                          |
|-------|--------|-------------------------------------|
| ≥ 75  | BUY    | Strong entry. Enter in 2–3 tranches |
| 65–74 | BUY    | Good setup. Enter with tight SL     |
| 50–64 | HOLD   | Wait for breakout confirmation      |
| 38–49 | HOLD   | Neutral. No action                  |
| 25–37 | SELL   | Avoid / reduce exposure             |
| < 25  | SELL   | Strong exit signal                  |

**Always use the Stop Loss shown. These are not guaranteed returns.**

---

## Improving accuracy

1. **Train the ML model** — `python backend/train.py` (LightGBM on 5yr history)
2. **Add more symbols** — larger watchlist = more signals
3. **Add fundamentals** — P/E, revenue growth via `yf.Ticker(sym).info`
4. **Add news sentiment** — free via NewsAPI.org (free tier: 100 req/day)
5. **Add FII/DII data** — NSEIndia.com publishes CSV daily (scrape-able)

---

## Cost breakdown

Everything listed here is genuinely free:

- **yfinance** — Yahoo Finance wrapper, no API key, no rate limits for reasonable use
- **LightGBM training** — CPU-only, runs in ~5 min on any free cloud instance
- **Render.com free** — 750 hours/month (more than enough for 1 service)
- **Railway.app free** — $5 credit/month ≈ ~500 hours
- **Koyeb free** — 2 nano instances, no sleep
- **GitHub Actions** — 2000 min/month free
- **cron-job.org** — unlimited free cron jobs

**Total cost: ₹0**
