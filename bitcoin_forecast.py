# bitcoin_forecast.py
# Action-safe Bitcoin forecast pipeline (daily, headless)
# - Collects Bitcoin headlines (Google News + Reddit RSS + curated RSS + GDELT)
# - Scores sentiment (FinBERT optional via USE_FINBERT, VADER fallback)
# - Joins with BTC-USD prices (yfinance)
# - Direction model (logistic, unchanged)
# - HAR-lite daily volatility model from squared daily returns (unchanged)
# - Monte Carlo price distributions (1d / 7d / 30d) (unchanged)
# - Saves figures + CSVs + LaTeX inputs for a Beamer report (unchanged)

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
    s = re.sub(r"http\S+|www\.\S+", "", s)         # strip urls
    s = re.sub(r"@[A-Za-z0-9_]+", "@user", s)      # anonymize handles
    s = re.sub(r"#", "", s)                        # drop hash but keep word
    s = re.sub(r"\s+", " ", s).strip()
    return s

def usd(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "--"

# =============================================================================
# 1) HEADLINES (upgraded collectors, same interfaces/columns)
# =============================================================================
import requests, certifi, feedparser
from urllib.parse import urlparse
from gnews import GNews

os.environ["SSL_CERT_FILE"] = certifi.where()

SCRAPE_DAYS   = int(os.getenv("SCRAPE_DAYS", "60"))
QUERY_TOPIC   = os.getenv("QUERY_TOPIC", "Bitcoin")
USE_GNEWS     = os.getenv("USE_GNEWS", "true").lower() == "true"
USE_REDDIT    = os.getenv("USE_REDDIT", "true").lower() == "true"
USE_RSS       = os.getenv("USE_RSS", "true").lower() == "true"
USE_GDELT     = os.getenv("USE_GDELT", "true").lower() == "true"
MAX_PER_FEED  = int(os.getenv("MAX_PER_FEED", "400"))

def _to_utc(ts):
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return pd.Timestamp.utcnow()

def _host(u):
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

# ---- Google News (no key) ----
def get_google_news(topic=QUERY_TOPIC, max_results=400):
    if not USE_GNEWS:
        return pd.DataFrame(columns=["date","text","source","url"])
    try:
        g = GNews(language="en", max_results=max_results)
        arts = g.get_news(topic)
        rows = []
        for a in arts:
            dt    = a.get("published date") or a.get("publishedDate") or datetime.utcnow()
            title = a.get("title") or ""
            desc  = a.get("description") or a.get("summary") or ""
            link  = a.get("url") or ""
            txt   = f"{title}. {desc}".strip()
            if txt:
                rows.append({"date": _to_utc(dt), "text": txt, "source": "google_news", "url": link})
        print(f"[google_news] {len(rows)} items")
        return pd.DataFrame(rows)
    except Exception as e:
        print("[google_news] failed ->", e)
        return pd.DataFrame(columns=["date","text","source","url"])

# ---- Reddit RSS (no key) ----
def get_reddit_rss(subreddits=("Bitcoin","CryptoCurrency","BitcoinMarkets","CryptoMarkets"),
                   days=SCRAPE_DAYS, max_per_sub=300):
    if not USE_REDDIT:
        return pd.DataFrame(columns=["date","text","source","url"])
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
                dt  = _to_utc(getattr(e, "published", getattr(e, "updated", datetime.utcnow())))
                if dt < cutoff:
                    continue
                title   = getattr(e, "title", "")
                summary = getattr(e, "summary", "")
                link    = getattr(e, "link", "")
                txt = f"{title}. {summary}".strip()
                if len(txt) < 6:
                    continue
                rows.append({"date": dt, "text": txt, "source": f"reddit/r/{sub}", "url": link})
                cnt += 1
            print(f"[reddit:r/{sub}] {cnt} items")
        except Exception as ex:
            print(f"[reddit:r/{sub}] failed -> {ex}")
    return pd.DataFrame(rows)

# ---- Curated RSS pack (headlines/snippets only) ----
NEWS_RSS = {
    # Big papers (headline feeds; no paywalled full text)
    "nyt_business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "nyt_markets":  "https://rss.nytimes.com/services/xml/rss/nyt/YourMoney.xml",
    "ft_markets":   "https://www.ft.com/markets?format=rss",
    "ft_companies": "https://www.ft.com/companies?format=rss",
    # Crypto/finance outlets
    "reuters_crypto":  "https://www.reuters.com/markets/cryptocurrency/rss",
    "coindesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph":   "https://cointelegraph.com/rss",
    "cnbc_crypto":     "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "yahoo_crypto":    "https://finance.yahoo.com/topic/crypto/rss",
}

def get_rss_many(feeds=NEWS_RSS, days=SCRAPE_DAYS, max_per_feed=MAX_PER_FEED):
    if not USE_RSS:
        return pd.DataFrame(columns=["date","text","source","url"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for name, url in feeds.items():
        try:
            f = feedparser.parse(url)
            cnt = 0
            for e in f.entries:
                if cnt >= max_per_feed:
                    break
                dt  = _to_utc(getattr(e, "published", getattr(e, "updated", datetime.utcnow())))
                if dt < cutoff:
                    continue
                title   = getattr(e, "title", "")
                # Prefer summary/detail, fallback to title only
                summary = getattr(e, "summary", "") or getattr(e, "description", "")
                link    = getattr(e, "link", "")
                txt = f"{title}. {summary}".strip() if summary else (title or "")
                if len(txt) < 6:
                    continue
                rows.append({"date": dt, "text": txt, "source": f"rss/{name}", "url": link})
                cnt += 1
            print(f"[rss:{name}] {cnt} items")
        except Exception as ex:
            print(f"[rss:{name}] failed -> {ex}")
    return pd.DataFrame(rows)

# ---- GDELT (broad, free, no key) ----
def get_gdelt(q="bitcoin OR btc", days=SCRAPE_DAYS):
    if not USE_GDELT:
        return pd.DataFrame(columns=["date","text","source","url"])
    try:
        base = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": q, "mode":"ArtList", "format":"JSON",
                  "maxrecords": 250, "timespan": f"{int(days)}d"}
        r = requests.get(base, params=params, timeout=30)
        arts = (r.json().get("articles") or []) if r.status_code == 200 else []
        rows=[]
        for a in arts:
            dt  = _to_utc(a.get("seendate"))
            ttl = a.get("title","")
            src = a.get("sourceCommonName","")
            url = a.get("url","")
            txt = f"{ttl}. {src}".strip() if src else ttl
            if txt:
                rows.append({"date": dt, "text": txt, "source": "gdelt", "url": url})
        print(f"[gdelt] {len(rows)} items")
        return pd.DataFrame(rows)
    except Exception as e:
        print("[gdelt] failed ->", e)
        return pd.DataFrame(columns=["date","text","source","url"])

# ---- Pull all collectors (toggle by flags) ----
frames = []
for fn in (get_google_news, get_reddit_rss, get_rss_many, get_gdelt):
    try:
        df_ = fn()
        if df_ is not None and not df_.empty:
            frames.append(df_)
    except Exception as e:
        print(f"[collector:{fn.__name__}] failed -> {e}")

if not frames:
    raise RuntimeError("No headlines collected (try enabling RSS/GDELT or widening SCRAPE_DAYS).")

# Stronger de-dup: prefer (host + text) when URL exists
raw = pd.concat(frames, ignore_index=True)
raw["url"] = raw.get("url", "")
raw["url_host"] = raw["url"].fillna("").map(_host)

# Simple dedup by (url_host, text) when url exists; otherwise by text alone
df_posts = (
    raw.drop_duplicates(subset=["url_host", "text"])
       .drop(columns=["url_host"])
       .sort_values("date")
       .reset_index(drop=True)
)

# Tag a high-level source category (news vs reddit vs other) for later analysis (doesn't change model)
def categorize(src: str) -> str:
    s = (src or "").lower()
    if s.startswith("reddit/"):
        return "reddit"
    if s.startswith("rss/") or s in ("google_news", "gdelt"):
        return "news"
    return "other"

df_posts["source_cat"] = df_posts["source"].astype(str).map(categorize)

print(f"[collect] total={len(raw)} unique={len(df_posts)}")

# =============================================================================
# 2) SENTIMENT (FinBERT optional, VADER fallback) — UNCHANGED
# =============================================================================
USE_FINBERT = os.getenv("USE_FINBERT", "false").lower() == "true"

def score_vader(texts):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        s = sia.polarity_scores(t)
        lab = "positive" if s["compound"] > 0.05 else "negative" if s["compound"] < -0.05 else "neutral"
        rows.append({"neg": s["neg"], "neu": s["neu"], "pos": s["pos"], "sentiment": lab})
    return pd.DataFrame(rows)

def score_finbert(texts):
    import torch, numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    out, labels = [], ["negative","neutral","positive"]
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = mdl(**enc).logits.cpu().numpy()
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            for p in probs:
                out.append({
                    "neg": float(p[0]), "neu": float(p[1]), "pos": float(p[2]),
                    "sentiment": labels[int(p.argmax())]
                })
    return pd.DataFrame(out)

df_posts["text_clean"] = df_posts["text"].astype(str).apply(clean_text)
texts = df_posts["text_clean"].tolist()

try:
    df_scores = score_finbert(texts) if USE_FINBERT else score_vader(texts)
    method = "FinBERT" if USE_FINBERT else "VADER"
except Exception as e:
    print("[sentiment] FinBERT failed, falling back to VADER:", e)
    df_scores = score_vader(texts)
    method = "VADER"

print(f"[sentiment] method = {method}, rows = {len(df_scores)}")

df = pd.concat([df_posts.reset_index(drop=True), df_scores.reset_index(drop=True)], axis=1)
df["date_utc"] = ensure_utc(df["date"])

# =============================================================================
# 3) DAILY FEATURES — UNCHANGED (plus harmless extras you can ignore)
# =============================================================================
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

# strictly lag (yesterday only) — model uses these as before
for c in ["pos_share", "neg_share", "pos_mean", "neg_mean", "sent_balance"]:
    df_daily[c] = df_daily[c].shift(1)

# (Optional extras, not used by your model; safe to ignore)
# Per-category counts (news vs reddit)
try:
    cat_counts = (
        d.groupby(["date_day","source_cat"])
         .size()
         .unstack(fill_value=0)
         .add_prefix("n_")
         .reset_index()
    )
    df_daily = df_daily.merge(cat_counts, on="date_day", how="left")
except Exception:
    pass

# =============================================================================
# 4) BTC DAILY PRICES — UNCHANGED
# =============================================================================
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

# forward-fill sentiment a little to handle sparse days (unchanged)
for col in ["n", "pos_share", "neg_share", "pos_mean", "neg_mean", "sent_balance"]:
    if col in data.columns:
        data[col] = data[col].fillna(method="ffill", limit=3)

# =============================================================================
# 5) DIRECTION MODEL (LOGIT) — UNCHANGED
# =============================================================================
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

# =============================================================================
# 6) VOL MODEL: HAR-lite — UNCHANGED
# =============================================================================
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

# =============================================================================
# 7) PRICE DISTRIBUTIONS — UNCHANGED
# =============================================================================
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

# =============================================================================
# 8) SAVE ARTIFACTS — UNCHANGED
# =============================================================================
stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")

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

# =============================================================================
# 9) FIGURES — improved price/sent plot; others unchanged
# =============================================================================
# (A) BTC price vs sentiment balance (smoothed)
try:
    plot_df = data.copy()
    # Smooth sentiment for readability (does not alter modeling)
    if "sent_balance" in plot_df.columns:
        plot_df["sent_balance_smooth"] = (
            plot_df["sent_balance"]
            .rolling(7, min_periods=3)
            .mean()
        )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["date_day"], plot_df["Adj Close"], label="BTC Adj Close", linewidth=1.8)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    yseries = plot_df.get("sent_balance_smooth", plot_df.get("sent_balance"))
    ax2.plot(plot_df["date_day"], yseries, label="Sentiment balance (7D avg)", linestyle="--", linewidth=1.8)
    ax2.set_ylabel("Sentiment balance")

    # unified legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # nicer date ticks
    try:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_OUT, "fig_price_sent.png"), dpi=160)
    plt.close(fig)
except Exception as e:
    print("[warn] could not render price/sent figure:", e)

# (B) RV true vs predicted (in-sample)
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rv_plot.index, rv_plot["rv_true"], label="RV next (true)")
    ax.plot(rv_plot.index, rv_plot["rv_pred"], label="RV next (pred)", alpha=0.9)
    ax.legend(); ax.set_title("Next-day realized variance — HAR-lite (in-sample)")
    ax.grid(True, alpha=0.25)
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
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_OUT, "fig_intervals.png"), dpi=160)
    plt.close(fig)
except Exception as e:
    print("[warn] could not render intervals figure:", e)

# =============================================================================
# 10) WRITE LaTeX INPUTS — UNCHANGED (keeps \csname…\endcsname style)
# =============================================================================
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
