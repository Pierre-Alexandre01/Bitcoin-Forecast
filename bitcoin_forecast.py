# bitcoin_forecast.py 
# Action-safe Bitcoin forecast pipeline (daily, headless)
# - Collects Bitcoin headlines (Google News + Reddit RSS)
# - Scores sentiment (VADER)
# - Joins with BTC-USD prices (yfinance)
# - Direction model (logistic)
# - HAR-lite daily volatility model from squared daily returns
# - Monte Carlo price distributions (1d / 7d / 30d)
# - Saves figures + CSVs + LaTeX inputs for a Beamer report

import os, re, math, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# -------------------- FOLDERS --------------------
REPORT_DIR = "report"
REPORT_OUT = os.path.join(REPORT_DIR, "outputs")
os.makedirs(REPORT_OUT, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------- HEADLESS PLOTTING --------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)


# -------------------- UTILS --------------------
def ensure_utc(series):
    s = pd.to_datetime(series, errors="coerce")
    try:
        tz = s.dt.tz
    except Exception:
        tz = None
    if tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"@[A-Za-z0-9_]+", "@user", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def usd(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "--"


# -------------------- 1) HEADLINES --------------------
import requests, certifi, feedparser
from gnews import GNews

os.environ["SSL_CERT_FILE"] = certifi.where()

SCRAPE_DAYS = 7
QUERY_TOPIC = "Bitcoin"


def _to_utc(ts):
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return pd.Timestamp.utcnow()


def get_google_news(topic=QUERY_TOPIC, max_results=400):
    try:
        g = GNews(language="en", max_results=max_results)
        arts = g.get_news(topic)
        rows = []
        for a in arts:
            dt = a.get("published date") or a.get("publishedDate") or datetime.utcnow()
            title = a.get("title") or ""
            desc = a.get("description") or a.get("summary") or ""
            rows.append({"date": _to_utc(dt), "text": f"{title}. {desc}".strip(), "source": "google_news"})
        print(f"[google_news] {len(rows)} items")
        return pd.DataFrame(rows)
    except Exception as e:
        print("[google_news] failed ->", e)
        return pd.DataFrame(columns=["date", "text", "source"])


def get_reddit_rss(subreddits=("Bitcoin", "CryptoCurrency", "BitcoinMarkets", "CryptoMarkets"),
                   days=SCRAPE_DAYS, max_per_sub=300):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/.rss"
        try:
            feed = feedparser.parse(url)
            cnt = 0
            for e in feed.entries:
                if cnt >= max_per_sub:
                    break
                dt = _to_utc(getattr(e, "published", getattr(e, "updated", datetime.utcnow())))
                if dt < cutoff:
                    continue
                title = getattr(e, "title", "")
                summary = getattr(e, "summary", "")
                txt = f"{title}. {summary}".strip()
                if len(txt) < 6:
                    continue
                rows.append({"date": dt, "text": txt, "source": f"reddit/r/{sub}"})
                cnt += 1
            print(f"[reddit:r/{sub}] {cnt} items")
        except Exception as ex:
            print(f"[reddit:r/{sub}] failed -> {ex}")
    return pd.DataFrame(rows)


frames = []
gn = get_google_news()
rd = get_reddit_rss()
for df_ in (gn, rd):
    if not df_.empty:
        frames.append(df_)

if not frames:
    raise RuntimeError("No headlines collected (Google News + Reddit both empty).")

df_posts = (
    pd.concat(frames, ignore_index=True)
      .drop_duplicates(subset=["text"])
      .sort_values("date")
      .reset_index(drop=True)
)

# -------------------- 2) SENTIMENT (VADER) --------------------
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()  # the workflow downloads the lexicon

df_posts["text_clean"] = df_posts["text"].astype(str).apply(clean_text)
rows = []
for t in df_posts["text_clean"]:
    sc = sia.polarity_scores(t)
    if sc["compound"] > 0.05:
        lab = "positive"
    elif sc["compound"] < -0.05:
        lab = "negative"
    else:
        lab = "neutral"
    rows.append({"neg": sc["neg"], "neu": sc["neu"], "pos": sc["pos"], "sentiment": lab})

df = pd.concat([df_posts.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
df["date_utc"] = ensure_utc(df["date"])

# -------------------- 3) DAILY FEATURES --------------------
d = df.copy()
d["date_day"] = d["date_utc"].dt.floor("D")
d["is_pos"] = (d["sentiment"] == "positive").astype(int)
d["is_neg"] = (d["sentiment"] == "negative").astype(int)

agg = (
    d.groupby("date_day")
    .agg(
        n=("text", "count"),
        pos_share=("is_pos", "mean"),
        neg_share=("is_neg", "mean"),
        pos_mean=("pos", "mean"),
        neg_mean=("neg", "mean"),
    )
    .reset_index()
)
agg["sent_balance"] = agg["pos_share"] - agg["neg_share"]
df_daily = agg.sort_values("date_day")

# strictly lag (yesterday only)
for c in ["pos_share", "neg_share", "pos_mean", "neg_mean", "sent_balance"]:
    df_daily[c] = df_daily[c].shift(1)

# -------------------- 4) BTC DAILY PRICES --------------------
import yfinance as yf

end = datetime.utcnow()
start = end - timedelta(days=120)
px = yf.Ticker("BTC-USD").history(start=start.date(), end=end.date(), interval="1d")

# widen if empty
if px is None or px.empty:
    px = yf.Ticker("BTC-USD").history(period="120d", interval="1d")
if px is None or px.empty:
    raise RuntimeError("yfinance returned no data for BTC-USD")

px = px.reset_index().rename(columns={"Date": "date_day"})
px["date_day"] = ensure_utc(px["date_day"]).dt.floor("D")
px["Adj Close"] = px.get("Adj Close", px["Close"])
px["ret"] = px["Adj Close"].pct_change()
px["ret_next"] = px["ret"].shift(-1)
px["up_next"] = (px["ret_next"] > 0).astype(int)

# join
data = (
    px.merge(df_daily, on="date_day", how="left")
      .sort_values("date_day")
      .reset_index(drop=True)
)

# forward-fill sentiment a little to handle sparse days
for col in ["n", "pos_share", "neg_share", "pos_mean", "neg_mean", "sent_balance"]:
    if col in data.columns:
        data[col] = data[col].fillna(method="ffill", limit=3)

# -------------------- 5) DIRECTION MODEL (LOGIT) --------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

fe = data.copy()
fe["sent_lag1"] = fe["sent_balance"].shift(1)
fe["ret_lag1"] = fe["ret"].shift(1)
fe = fe.dropna(subset=["up_next"]).copy()

feat_cols = [c for c in ["sent_lag1", "ret_lag1", "pos_share", "neg_share"] if c in fe.columns]
feat_cols = [c for c in feat_cols if fe[c].notna().any()]

if len(fe) < 20 or len(feat_cols) == 0 or fe["up_next"].nunique() < 2:
    p_up_1d = 0.5
    dir_model_name = "heuristic"
else:
    X = fe[feat_cols].to_numpy()
    y = fe["up_next"].astype(int).to_numpy()
    pipe_dir = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe_dir.fit(X, y)
    x_next = fe.iloc[[-1]][feat_cols].to_numpy()
    p_up_1d = float(pipe_dir.predict_proba(x_next)[:, 1])
    p_up_1d = float(np.clip(p_up_1d, 0.05, 0.95))
    dir_model_name = "Logit(balanced)"

print(f"\nDirection P(up) via {dir_model_name}: {p_up_1d:.3f}")

# -------------------- 6) VOL MODEL: HAR-lite (from daily squared returns) --------------------
rv = fe[["date_day", "ret"]].copy()
rv["rv"] = rv["ret"].pow(2)
rv = rv.dropna()

har = rv.set_index("date_day").copy()
har["rv_ema5"] = har["rv"].ewm(span=5, min_periods=5).mean()
har["rv_ema22"] = har["rv"].ewm(span=22, min_periods=22).mean()
har["rv_d_lag1"] = har["rv"].shift(1)
har["rv_w_lag1"] = har["rv_ema5"].shift(1)
har["rv_m_lag1"] = har["rv_ema22"].shift(1)
har["y"] = np.log(har["rv"].shift(-1).clip(lower=1e-12))
har = har.dropna()

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

X_cols = ["rv_d_lag1", "rv_w_lag1", "rv_m_lag1"]
Xv = har[X_cols].to_numpy()
yv = har["y"].to_numpy()

scaler_vol = RobustScaler()
Xv_s = scaler_vol.fit_transform(Xv)

gb = GradientBoostingRegressor(
    loss="huber", n_estimators=500, learning_rate=0.03,
    subsample=0.7, max_depth=3, random_state=42
)
gb.fit(Xv_s, yv)

# next-day RV prediction
x_last = har.iloc[[-1]][X_cols].to_numpy()
rv_pred_next = float(np.exp(gb.predict(scaler_vol.transform(x_last))[0]))
sigma_1d = math.sqrt(rv_pred_next)
print(f"Daily vol (HAR-lite): sigma_1d={sigma_1d:.4%}  RV={rv_pred_next:.6f}")

# in-sample RV predictions (for the figure)
yhat_all = np.exp(gb.predict(Xv_s))
rv_plot = pd.DataFrame({
    "rv_true": np.exp(yv),
    "rv_pred": yhat_all
}, index=har.index)

# -------------------- 7) PRICE DISTRIBUTIONS --------------------
spot = float(data["Adj Close"].dropna().iloc[-1])
last_day = pd.to_datetime(data["date_day"].dropna().iloc[-1])

mu_recent = float(data["ret"].dropna().tail(30).mean())
sent_tilt = 0.5 * (p_up_1d - 0.5)  # small tilt by sentiment
mu_1d = mu_recent + sent_tilt * sigma_1d


def horizon_params(days):
    return mu_1d * days, sigma_1d * math.sqrt(days)


rng = np.random.default_rng(42)


def simulate_prices(S0, days, sims=20000):
    mu, sig = horizon_params(days)
    r = rng.normal(mu, sig, size=sims)
    return S0 * np.exp(r)


def summarize(ST):
    pct = np.percentile
    return {
        "median": float(pct(ST, 50)),
        "p10": float(pct(ST, 10)),
        "p90": float(pct(ST, 90)),
        "p05": float(pct(ST, 5)),
        "p95": float(pct(ST, 95)),
        "mean": float(np.mean(ST)),
        "prob_up": float(np.mean(ST > spot)),
    }


res_1d = summarize(simulate_prices(spot, 1))
res_7d = summarize(simulate_prices(spot, 7))
res_30d = summarize(simulate_prices(spot, 30))

print(f"\n# ===== Price Forecasts (spot = {spot:.2f}, as of {last_day.date()}) =====")
print(f"Direction (P[Up] tomorrow): {100 * p_up_1d:.1f}%")
print(f"Daily vol (HAR-lite): {sigma_1d:.2%} (sigma_1d), RV={rv_pred_next:.6f}")

for name, res in [("1-Day", res_1d), ("7-Day", res_7d), ("30-Day", res_30d)]:
    print(
        f"\n{name} forecast:\n"
        "  Median: {:,.2f} | 10–90%: [{:,.2f}, {:,.2f}] | 5–95%: [{:,.2f}, {:,.2f}] | P(Up): {:.1f}%"
            .format(res["median"], res["p10"], res["p90"], res["p05"], res["p95"], 100 * res["prob_up"])
    )

# -------------------- 8) SAVE ARTIFACTS --------------------
stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")

# CSV / JSON
df.to_csv(os.path.join(REPORT_OUT, f"posts_scored_{stamp}.csv"), index=False)
df_daily.to_csv(os.path.join(REPORT_OUT, f"sentiment_daily_{stamp}.csv"), index=False)
data.to_csv(os.path.join(REPORT_OUT, f"joined_price_sentiment_{stamp}.csv"), index=False)

summary = {
    "as_of_utc": pd.Timestamp.utcnow().isoformat(),
    "spotUSD": spot,
    "probUpOneDay": f"{100 * p_up_1d:.1f}",
    "sigmaOneDay": f"{100 * sigma_1d:.2f}",
    "p1": {"med": res_1d["median"], "p10": res_1d["p10"], "p90": res_1d["p90"], "p05": res_1d["p05"], "p95": res_1d["p95"]},
    "p7": {"med": res_7d["median"], "p10": res_7d["p10"], "p90": res_7d["p90"], "p05": res_7d["p05"], "p95": res_7d["p95"]},
    "p30": {"med": res_30d["median"], "p10": res_30d["p10"], "p90": res_30d["p90"], "p05": res_30d["p05"], "p95": res_30d["p95"]},
    "reportDate": last_day.date().isoformat()
}
with open(os.path.join(REPORT_OUT, f"forecast_summary_{stamp}.json"), "w") as f:
    json.dump(summary, f, indent=2)

# -------------------- 9) FIGURES --------------------
# (A) BTC price vs sentiment balance
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["date_day"], data["Adj Close"], label="BTC Adj Close")
    ax2 = ax.twinx()
    ax2.plot(data["date_day"], data["sent_balance"], label="Sentiment balance (lag1)", linestyle="--")
    ax.set_title("BTC price vs sentiment balance")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)"); ax2.set_ylabel("Sentiment")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_OUT, "fig_price_sent.png"), dpi=160)
    plt.close(fig)
except Exception as e:
    print("[warn] could not render price/sent figure:", e)

# (B) RV true vs predicted (in-sample for illustration)
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rv_plot.index, rv_plot["rv_true"], label="RV next (true)")
    ax.plot(rv_plot.index, rv_plot["rv_pred"], label="RV next (pred)", alpha=0.9)
    ax.legend(); ax.set_title("Next-day realized variance — HAR-lite (in-sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_OUT, "fig_rv.png"), dpi=160)
    plt.close(fig)
except Exception as e:
    print("[warn] could not render RV figure:", e)

# (C) Forecast intervals (10–90%) across horizons
try:
    horizons = ["1d", "7d", "30d"]
    meds = [res_1d["median"], res_7d["median"], res_30d["median"]]
    p10 = [res_1d["p10"], res_7d["p10"], res_30d["p10"]]
    p90 = [res_1d["p90"], res_7d["p90"], res_30d["p90"]]
    x = np.arange(len(horizons))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, meds, label="Median")
    for i in range(len(x)):
        ax.vlines(x[i], p10[i], p90[i], lw=6, alpha=0.6)
    ax.set_xticks(x, horizons)
    ax.set_title("Price forecast intervals (10–90%)")
    ax.set_ylabel("USD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_OUT, "fig_intervals.png"), dpi=160)
    plt.close(fig)
except Exception as e:
    print("[warn] could not render intervals figure:", e)

# -------------------- 10) WRITE LaTeX INPUTS --------------------
inputs_tex = r"""
%% Auto-generated by pipeline. DO NOT EDIT.

\expandafter\def\csname reportDate\endcsname{%s}
\expandafter\def\csname spotUSD\endcsname{%s}
\expandafter\def\csname probUpOneDay\endcsname{%s}
\expandafter\def\csname sigmaOneDay\endcsname{%s}

%% One-day
\expandafter\def\csname pOneMed\endcsname{%s}
\expandafter\def\csname pOneP10\endcsname{%s}
\expandafter\def\csname pOneP90\endcsname{%s}
\expandafter\def\csname pOneP05\endcsname{%s}
\expandafter\def\csname pOneP95\endcsname{%s}

%% Seven-day
\expandafter\def\csname pSevenMed\endcsname{%s}
\expandafter\def\csname pSevenP10\endcsname{%s}
\expandafter\def\csname pSevenP90\endcsname{%s}
\expandafter\def\csname pSevenP05\endcsname{%s}
\expandafter\def\csname pSevenP95\endcsname{%s}

%% Thirty-day
\expandafter\def\csname pThirtyMed\endcsname{%s}
\expandafter\def\csname pThirtyP10\endcsname{%s}
\expandafter\def\csname pThirtyP90\endcsname{%s}
\expandafter\def\csname pThirtyP05\endcsname{%s}
\expandafter\def\csname pThirtyP95\endcsname{%s}

%% Numeric (no commas) for tables
\expandafter\def\csname pOneMedNum\endcsname{%.2f}
\expandafter\def\csname pOneP10Num\endcsname{%.2f}
\expandafter\def\csname pOneP90Num\endcsname{%.2f}
\expandafter\def\csname pOneP05Num\endcsname{%.2f}
\expandafter\def\csname pOneP95Num\endcsname{%.2f}
\expandafter\def\csname pSevenMedNum\endcsname{%.2f}
\expandafter\def\csname pSevenP10Num\endcsname{%.2f}
\expandafter\def\csname pSevenP90Num\endcsname{%.2f}
\expandafter\def\csname pSevenP05Num\endcsname{%.2f}
\expandafter\def\csname pSevenP95Num\endcsname{%.2f}
\expandafter\def\csname pThirtyMedNum\endcsname{%.2f}
\expandafter\def\csname pThirtyP10Num\endcsname{%.2f}
\expandafter\def\csname pThirtyP90Num\endcsname{%.2f}
\expandafter\def\csname pThirtyP05Num\endcsname{%.2f}
\expandafter\def\csname pThirtyP95Num\endcsname{%.2f}
""" % (
    last_day.date().isoformat(),
    usd(spot),
    f"{100 * p_up_1d:.1f}",
    f"{100 * sigma_1d:.2f}",

    usd(res_1d["median"]), usd(res_1d["p10"]), usd(res_1d["p90"]), usd(res_1d["p05"]), usd(res_1d["p95"]),
    usd(res_7d["median"]), usd(res_7d["p10"]), usd(res_7d["p90"]), usd(res_7d["p05"]), usd(res_7d["p95"]),
    usd(res_30d["median"]), usd(res_30d["p10"]), usd(res_30d["p90"]), usd(res_30d["p05"]), usd(res_30d["p95"]),

    res_1d["median"], res_1d["p10"], res_1d["p90"], res_1d["p05"], res_1d["p95"],
    res_7d["median"], res_7d["p10"], res_7d["p90"], res_7d["p05"], res_7d["p95"],
    res_30d["median"], res_30d["p10"], res_30d["p90"], res_30d["p05"], res_30d["p95"],
)

with open(os.path.join(REPORT_DIR, "inputs.tex"), "w") as f:
    f.write(inputs_tex)

print(f"[report] wrote {os.path.join(REPORT_DIR, 'inputs.tex')}")
print(f"[report] assets saved to {REPORT_OUT}")
