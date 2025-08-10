#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin forecast (continuous): lightweight, Actions-friendly.
Saves: outputs/predictions.csv, outputs/forecast_chart.png, outputs/report.md
"""

import os, math, io, sys, time, datetime as dt
import numpy as np
import pandas as pd

# --- headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- deps that are OK on Actions
import yfinance as yf
import feedparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- VADER (fast, no GPU)
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

UTC_NOW = pd.Timestamp.utcnow()
TODAY = UTC_NOW.floor("D")

# --------------- 1) Pull prices ---------------
def load_btc_prices(days=180):
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days+5)
    tkr = yf.Ticker("BTC-USD")
    px = tkr.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=False)
    if px is None or px.empty:
        # fallback
        px = tkr.history(period=f"{days}d", interval="1d", auto_adjust=False)
    px = px.reset_index().rename(columns={"Date":"date"})
    # normalize cols
    for want in ["Open","High","Low","Close","Adj Close","Volume"]:
        if want not in px.columns and want == "Adj Close":
            px["Adj Close"] = px["Close"]
    px["date"] = pd.to_datetime(px["date"], utc=True).dt.floor("D")
    px["ret"] = px["Adj Close"].pct_change()
    return px.dropna(subset=["Adj Close"])

px = load_btc_prices(240)
spot = float(px["Adj Close"].iloc[-1])

# --------------- 2) Headlines -> sentiment ---------------
def get_google_news(topic="Bitcoin", max_items=200):
    # A public RSS endpoint via Google News
    # (simple, no keys; may occasionally rate limit)
    url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        published = getattr(e, "published", getattr(e, "updated", None))
        try:
            dtp = pd.to_datetime(published, utc=True)
        except Exception:
            dtp = UTC_NOW
        txt = f"{title}. {summary}".strip()
        if len(txt) > 6:
            rows.append({"date": dtp, "text": txt, "source": "google_news"})
    return pd.DataFrame(rows)

def get_reddit_rss(subs=("Bitcoin","CryptoCurrency","BitcoinMarkets"), max_per=150, days=14):
    cutoff = UTC_NOW - pd.Timedelta(days=days)
    rows = []
    for sub in subs:
        url = f"https://www.reddit.com/r/{sub}/.rss"
        feed = feedparser.parse(url)
        cnt = 0
        for e in feed.entries:
            if cnt >= max_per: break
            title = getattr(e, "title", "")
            summary = getattr(e, "summary", "")
            published = getattr(e, "published", getattr(e, "updated", None))
            try:
                dtp = pd.to_datetime(published, utc=True)
            except Exception:
                dtp = UTC_NOW
            if dtp < cutoff: 
                continue
            txt = f"{title}. {summary}".strip()
            if len(txt) > 6:
                rows.append({"date": dtp, "text": txt, "source": f"reddit/{sub}"})
                cnt += 1
    return pd.DataFrame(rows)

news = get_google_news("Bitcoin", 300)
reddit = get_reddit_rss()

posts = pd.concat([news, reddit], ignore_index=True) if not news.empty or not reddit.empty else pd.DataFrame(columns=["date","text","source"])
if posts.empty:
    # keep pipeline running with a placeholder row
    posts = pd.DataFrame([{"date": UTC_NOW, "text": "Bitcoin price update.", "source": "placeholder"}])
posts["date"] = pd.to_datetime(posts["date"], utc=True)
posts["date_day"] = posts["date"].dt.floor("D")

# VADER scores
sia = SentimentIntensityAnalyzer()
scored = []
for t in posts["text"].astype(str):
    s = sia.polarity_scores(t)
    if s["compound"] > 0.05: lab = "positive"
    elif s["compound"] < -0.05: lab = "negative"
    else: lab = "neutral"
    scored.append({"neg":s["neg"], "neu":s["neu"], "pos":s["pos"], "sentiment":lab})
scored = pd.DataFrame(scored)
posts = pd.concat([posts.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

# aggregate per day
g = (posts.groupby("date_day")
           .agg(n=("text","count"),
                pos_share=("sentiment", lambda s: np.mean(s=="positive")),
                neg_share=("sentiment", lambda s: np.mean(s=="negative")),
                pos_mean=("pos","mean"),
                neg_mean=("neg","mean"))
           .reset_index())
g["sent_balance"] = g["pos_share"] - g["neg_share"]

# --------------- 3) Join with price & build features ---------------
df = px.merge(g, left_on="date", right_on="date_day", how="left")
df = df.sort_values("date").reset_index(drop=True)

# forward-fill sentiment a bit to avoid NaNs
for c in ["n","pos_share","neg_share","pos_mean","neg_mean","sent_balance"]:
    if c in df.columns:
        df[c] = df[c].fillna(method="ffill")

df["up_next"] = (df["ret"].shift(-1) > 0).astype(int)

# Direction features (strictly lagged)
df["sent_lag1"] = df["sent_balance"].shift(1)
df["sent_r3"]   = df["sent_balance"].shift(1).rolling(3, min_periods=2).mean()
df["ret_lag1"]  = df["ret"].shift(1)
df["vol_5"]     = df["ret"].shift(1).rolling(5, min_periods=3).std()

feat_cols = [c for c in ["sent_lag1","sent_r3","ret_lag1","vol_5"] if c in df.columns]
model_df = df.dropna(subset=feat_cols + ["up_next"]).copy()

if len(model_df) >= 30 and model_df["up_next"].nunique() >= 2:
    X = model_df[feat_cols].to_numpy()
    y = model_df["up_next"].astype(int).to_numpy()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X, y)
    x_next = df.iloc[[-1]][feat_cols].to_numpy()
    p_up = float(pipe.predict_proba(x_next)[:,1])
    dir_model = "Logit(balanced)"
else:
    # tiny-mode fallback
    s1 = float(df["sent_balance"].shift(1).iloc[-1] if "sent_balance" in df.columns else 0.0)
    sstd = float(df["sent_balance"].shift(1).std()) if "sent_balance" in df.columns else 1.0
    r1 = float(df["ret"].shift(1).iloc[-1] if "ret" in df.columns else 0.0)
    rstd = float(df["ret"].shift(1).std()) if "ret" in df.columns else 1.0
    z_s = 0.0 if sstd == 0 or np.isnan(sstd) else np.tanh(s1 / (sstd + 1e-9))
    z_r = 0.0 if rstd == 0 or np.isnan(rstd) else np.tanh(r1 / (rstd + 1e-9))
    p_up = float(np.clip(0.5 + 0.20*z_s + 0.10*z_r, 0.05, 0.95))
    dir_model = "heuristic"

# --------------- 4) Volatility (daily RV proxy) ---------------
# Simple daily realized variance proxy = ret^2 with EMAs (HAR-like)
df["rv"] = (np.log(df["Adj Close"]).diff().fillna(0.0))**2
df["rv_ema5"]  = df["rv"].ewm(span=5,  min_periods=5).mean()
df["rv_ema22"] = df["rv"].ewm(span=22, min_periods=10).mean()
# next-day rv linear blend (super simple)
rv_next = float(df["rv_ema5"].iloc[-1] * 0.6 + df["rv_ema22"].iloc[-1] * 0.4)
sigma_1d = math.sqrt(max(rv_next, 1e-12))

# drift: recent mean + sentiment tilt
mu_recent = float(df["ret"].dropna().tail(30).mean())
mu_1d = mu_recent + 0.5 * (p_up - 0.5) * sigma_1d

def horizon_params(days:int):
    return mu_1d * days, sigma_1d * math.sqrt(days)

rng = np.random.default_rng(123)

def simulate_prices(S0, days, sims=20000):
    mu, sig = horizon_params(days)
    r = rng.normal(mu, sig, size=sims)
    return S0 * np.exp(r)

def summarize(ST, S0):
    pct = np.percentile
    return {
        "spot": S0,
        "median": float(pct(ST, 50)),
        "p10": float(pct(ST, 10)),
        "p90": float(pct(ST, 90)),
        "p05": float(pct(ST, 5)),
        "p95": float(pct(ST, 95)),
        "prob_up": float(np.mean(ST > S0))
    }

res_1 = summarize(simulate_prices(spot, 1), spot)
res_7 = summarize(simulate_prices(spot, 7), spot)
res_30= summarize(simulate_prices(spot, 30), spot)

# --------------- 5) Save CSV ---------------
rows = []
for name, res in [("1d", res_1), ("7d", res_7), ("30d", res_30)]:
    rows.append({
        "horizon": name,
        "as_of": TODAY.isoformat(),
        "spot": res["spot"],
        "p_up_tomorrow": p_up if name=="1d" else np.nan,
        "median": res["median"],
        "p10": res["p10"], "p90": res["p90"],
        "p05": res["p05"], "p95": res["p95"]
    })
pred_csv = os.path.join(OUT_DIR, "predictions.csv")
pd.DataFrame(rows).to_csv(pred_csv, index=False)

# --------------- 6) Plot & save chart ---------------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["date"], df["Adj Close"], label="BTC Adj Close", lw=2)
ax.set_title(f"BTC-USD — Price & Forecast bands (as of {TODAY.date()})")
ax.set_xlabel("Date"); ax.set_ylabel("USD")

# draw forecast ranges as horizontal lines from 'today' (visual cue)
for (label, res, color) in [("1d",res_1,"tab:green"), ("7d",res_7,"tab:orange"), ("30d",res_30,"tab:red")]:
    x = [df["date"].iloc[-1], df["date"].iloc[-1] + pd.Timedelta(days=int(label.replace("d","")))]
    # plot as band end markers
    ax.hlines([res["p10"], res["median"], res["p90"]], xmin=x[0], xmax=x[1], colors=color, linestyles=["--","-","--"], label=f"{label} 10/50/90%")

ax.legend(loc="upper left")
ax.grid(True)
chart_path = os.path.join(OUT_DIR, "forecast_chart.png")
plt.tight_layout()
plt.savefig(chart_path, dpi=160)
plt.close(fig)

# --------------- 7) Markdown report ---------------
report = f"""# Bitcoin Forecast (Auto)

**As of:** {TODAY.date()}  
**Spot:** {spot:,.2f} USD  
**Direction model:** {dir_model}  
**P(up) tomorrow:** {100*p_up:.1f}%  
**Daily vol (sigma):** {sigma_1d:.2%}

## Forecasts (median, with 10–90% and 5–95% bands)

| Horizon | Median | 10–90% low | 10–90% high | 5–95% low | 5–95% high | P(Up vs spot) |
|---|---:|---:|---:|---:|---:|---:|
| 1 day | {res_1['median']:,.2f} | {res_1['p10']:,.2f} | {res_1['p90']:,.2f} | {res_1['p05']:,.2f} | {res_1['p95']:,.2f} | {100*res_1['prob_up']:.1f}% |
| 7 days | {res_7['median']:,.2f} | {res_7['p10']:,.2f} | {res_7['p90']:,.2f} | {res_7['p05']:,.2f} | {res_7['p95']:,.2f} | {100*res_7['prob_up']:.1f}% |
| 30 days | {res_30['median']:,.2f} | {res_30['p10']:,.2f} | {res_30['p90']:,.2f} | {res_30['p05']:,.2f} | {res_30['p95']:,.2f} | {100*res_30['prob_up']:.1f}% |

![chart](forecast_chart.png)

*Notes:*  
- Sentiment from Google News & Reddit RSS via VADER (lightweight, no keys).  
- Volatility from daily log-return RV with short/long EMAs (HAR-like proxy).  
- This is an automated, purely informational forecast.
"""
with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
    f.write(report)

print(f"Saved:\n- {pred_csv}\n- {chart_path}\n- {os.path.join(OUT_DIR, 'report.md')}")
