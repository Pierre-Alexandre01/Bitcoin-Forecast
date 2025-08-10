#!/usr/bin/env python3
import os, math, io, datetime as dt
import numpy as np, pandas as pd, matplotlib.pyplot as plt, feedparser, requests
from dateutil import tz
from jinja2 import Environment, FileSystemLoader
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --- config ---
TICKER = os.getenv("TICKER", "BTC-USD")
SCRAPE_DAYS = int(os.getenv("SCRAPE_DAYS", "14"))
MAX_NEWS = int(os.getenv("MAX_NEWS", "400"))
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- util --------
def _utcnow():
    return pd.Timestamp.utcnow().tz_localize("UTC")

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    import re, emoji
    URL_RE = re.compile(r"http\S+|www\.\S+")
    HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
    RT_RE = re.compile(r"\bRT\b:?")
    MULTI = re.compile(r"\s+")
    s = RT_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = HANDLE_RE.sub("@user", s)
    try:
        s = emoji.replace_emoji(s, replace="")
    except Exception:
        pass
    return MULTI.sub(" ", s).strip()

# -------- data: news via Google RSS + Reddit RSS (no keys) --------
def fetch_google_news(query="Bitcoin", max_items=300):
    # Google News RSS (web-friendly, no auth)
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        dtp = getattr(e, "published", getattr(e, "updated", ""))
        ts = pd.to_datetime(dtp, errors="coerce", utc=True)
        if pd.isna(ts):
            ts = _utcnow()
        rows.append({"date": ts, "text": f"{title}. {summary}".strip(), "source": "google_news"})
    return pd.DataFrame(rows)

def fetch_reddit_rss(subreddits=("Bitcoin","CryptoCurrency"), max_items=300):
    rows = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/.rss"
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "")
            summary = getattr(e, "summary", "")
            dtp = getattr(e, "published", getattr(e, "updated", ""))
            ts = pd.to_datetime(dtp, errors="coerce", utc=True)
            if pd.isna(ts):
                ts = _utcnow()
            rows.append({"date": ts, "text": f"{title}. {summary}".strip(), "source": f"reddit/{sub}"})
    return pd.DataFrame(rows)

def collect_posts(days=SCRAPE_DAYS, max_total=MAX_NEWS):
    cutoff = _utcnow() - pd.Timedelta(days=days)
    g = fetch_google_news("Bitcoin", max_items=max_total//2)
    r = fetch_reddit_rss(max_items=max_total//2)
    df = pd.concat([g, r], ignore_index=True)
    df = df[df["date"] >= cutoff]
    df["text_clean"] = df["text"].astype(str).map(clean_text)
    df = df[df["text_clean"].str.len() >= 8].drop_duplicates(subset=["text_clean"])
    return df.sort_values("date")

# -------- sentiment: VADER (fast, no GPU) --------
def score_sentiment(df_posts: pd.DataFrame) -> pd.DataFrame:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    s = df_posts["text_clean"].map(sia.polarity_scores).apply(pd.Series)
    out = df_posts.copy()
    out["neg"], out["neu"], out["pos"], out["compound"] = s["neg"], s["neu"], s["pos"], s["compound"]
    out["sentiment"] = np.where(out["compound"] > 0.05, "positive",
                         np.where(out["compound"] < -0.05, "negative", "neutral"))
    out["date_utc"] = pd.to_datetime(out["date"], utc=True)
    out["date_day"] = out["date_utc"].dt.floor("D")
    return out

def daily_features(df_scored: pd.DataFrame) -> pd.DataFrame:
    d = df_scored.copy()
    d["is_pos"] = (d["sentiment"] == "positive").astype(int)
    d["is_neg"] = (d["sentiment"] == "negative").astype(int)
    agg = (d.groupby("date_day")
             .agg(n=("text","count"),
                  pos_share=("is_pos","mean"),
                  neg_share=("is_neg","mean"),
                  pos_mean=("pos","mean"),
                  neg_mean=("neg","mean"))
             .reset_index())
    agg["sent_balance"] = agg["pos_share"] - agg["neg_share"]
    # one-day lag to avoid look-ahead
    for c in ["pos_share","neg_share","pos_mean","neg_mean","sent_balance"]:
        agg[c] = agg[c].shift(1)
    return agg

# -------- prices --------
def fetch_prices(ticker=TICKER, lookback_days=120):
    end = dt.datetime.utcnow().date()
    start = end - dt.timedelta(days=lookback_days+5)
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    hist = hist.reset_index().rename(columns={"Date":"date_day"})
    hist["date_day"] = pd.to_datetime(hist["date_day"], utc=True).dt.floor("D")
    if "Adj Close" not in hist.columns:
        hist["Adj Close"] = hist["Close"]
    hist["ret"] = np.log(hist["Adj Close"]).diff()
    hist["up_next"] = (hist["ret"].shift(-1) > 0).astype(int)
    return hist

# -------- simple HAR on daily RV (no intraday to keep Actions reliable) --------
def build_har_rv(px: pd.DataFrame) -> pd.DataFrame:
    rv = px.copy()
    rv["rv"] = rv["ret"].pow(2)  # daily realized variance proxy
    rv["rv_next"] = rv["rv"].shift(-1)
    rv["rv_d_lag1"] = rv["rv"].shift(1)
    rv["rv_w_lag1"] = rv["rv"].rolling(5, min_periods=3).mean().shift(1)
    rv["rv_m_lag1"] = rv["rv"].rolling(22, min_periods=10).mean().shift(1)
    rv = rv.dropna(subset=["rv_d_lag1","rv_w_lag1","rv_m_lag1","rv_next"]).copy()
    rv["y"] = np.log(rv["rv_next"].clip(lower=1e-12))
    return rv

# -------- tiny direction model --------
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def build_direction_model(data_join: pd.DataFrame):
    fe = data_join.copy()
    fe["sent_lag1"] = fe["sent_balance"].shift(1)
    fe["sent_r3"]   = fe["sent_balance"].shift(1).rolling(3, min_periods=2).mean()
    fe["ret_lag1"]  = fe["ret"].shift(1)
    fe = fe.dropna(subset=["up_next"]).copy()
    feat = ["sent_lag1","sent_r3","ret_lag1"]
    fe = fe.dropna(subset=feat).copy()
    if len(fe) < 20 or fe["up_next"].nunique() < 2:
        return None, feat, fe
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(fe[feat])
    y = fe["up_next"].astype(int).to_numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    return (clf, imp), feat, fe

# -------- render charts --------
def plot_fan(spot, mu, sigma, horizon_days, path_png):
    rng = np.random.default_rng(42)
    r = rng.normal(loc=mu, scale=sigma, size=50_000)
    ST = spot * np.exp(r)
    q = np.percentile(ST, [5,10,50,90,95])
    plt.figure(figsize=(8,4))
    plt.title(f"{horizon_days}-day Price Distribution")
    plt.axvline(q[2], linestyle="--")
    plt.hist(ST, bins=80, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()
    return {"p5":q[0], "p10":q[1], "median":q[2], "p90":q[3], "p95":q[4]}

def plot_sentiment_price(data_join, path_png):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(data_join["date_day"], data_join["Adj Close"], label="BTC Adj Close")
    ax2 = ax.twinx()
    ax2.plot(data_join["date_day"], data_join["sent_balance"], color="orange", alpha=0.7, label="sent_balance")
    ax.set_title("BTC vs Sentiment Balance")
    fig.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)

# -------- main pipeline --------
def main():
    posts = collect_posts()
    if posts.empty:
        raise RuntimeError("No posts collected")
    scored = score_sentiment(posts)
    daily = daily_features(scored)

    px = fetch_prices()
    data_join = px.merge(daily, on="date_day", how="left").sort_values("date_day")
    # ffill sentiment a bit
    for c in ["sent_balance","pos_share","neg_share"]:
        if c in data_join.columns:
            data_join[c] = data_join[c].fillna(method="ffill")

    # direction model prob for tomorrow
    model, feat_cols, fe = build_direction_model(data_join)
    if model:
        clf, imp = model
        xnext = fe.iloc[[-1]][feat_cols]
        p_up = float(clf.predict_proba(imp.transform(xnext))[:,1])
        dir_used = "Logit(balanced)"
    else:
        # heuristic
        s1 = data_join["sent_balance"].shift(1).iloc[-1]
        sstd = data_join["sent_balance"].shift(1).std()
        z = 0.0 if (sstd==0 or pd.isna(sstd)) else np.tanh(s1/(sstd+1e-9))
        p_up = float(np.clip(0.5 + 0.2*z, 0.05, 0.95))
        dir_used = "heuristic"

    # HAR on daily RV
    har = build_har_rv(px)
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import GradientBoostingRegressor
    scaler = RobustScaler()
    gb = GradientBoostingRegressor(loss="huber", n_estimators=500, max_depth=3, learning_rate=0.04, subsample=0.8, random_state=42)
    Xv = har[["rv_d_lag1","rv_w_lag1","rv_m_lag1"]].to_numpy()
    yv = har["y"].to_numpy()
    gb.fit(scaler.fit_transform(Xv), yv)
    x_last = har.iloc[[-1]][["rv_d_lag1","rv_w_lag1","rv_m_lag1"]].to_numpy()
    rv_next = float(np.exp(gb.predict(scaler.transform(x_last))[0]))
    sigma_1d = math.sqrt(rv_next)

    # horizon params (drift tilt by sentiment)
    spot = float(px["Adj Close"].dropna().iloc[-1])
    mu_recent = float(px["ret"].dropna().tail(30).mean())
    mu_1d = mu_recent + 0.5*(p_up-0.5)*sigma_1d
    def params(days): return mu_1d*days, sigma_1d*math.sqrt(days)

    # charts & numbers
    fan1 = plot_fan(spot, *params(1), 1, os.path.join(OUT_DIR, "fan_1d.png"))
    fan7 = plot_fan(spot, *params(7), 7, os.path.join(OUT_DIR, "fan_7d.png"))
    fan30= plot_fan(spot, *params(30), 30, os.path.join(OUT_DIR, "fan_30d.png"))
    plot_sentiment_price(data_join, os.path.join(OUT_DIR, "sent_vs_px.png"))

    # render LaTeX via Jinja2
    env = Environment(loader=FileSystemLoader("report"))
    tmpl = env.get_template("template.tex")
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    tex = tmpl.render(
        date=today,
        spot=f"{spot:,.2f}",
        p_up=f"{100*p_up:.1f}%",
        sigma=f"{sigma_1d*100:.2f}%",
        dir_used=dir_used,
        fan1=fan1, fan7=fan7, fan30=fan30
    )
    tex_path = os.path.join(OUT_DIR, "main.tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    # compile with tectonic if available, else just leave .tex and PNGs
    try:
        import subprocess
        subprocess.check_call(["bash","-lc", f"cd {OUT_DIR} && tectonic main.tex"])
    except Exception as e:
        print("Tectonic not available, left .tex uncompiled:", e)

    # also export a compact JSON for API use
    summary = {
        "as_of": today,
        "spot": spot,
        "p_up_1d": p_up,
        "sigma_1d": sigma_1d,
        "fan_1d": fan1, "fan_7d": fan7, "fan_30d": fan30
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        import json; json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
