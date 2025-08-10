# Auto BTC Forecast (Continuous)

This repo runs a **scheduled pipeline** (GitHub Actions) that:
1. Pulls BTC-USD daily prices (Yahoo Finance)
2. Scrapes **Google News RSS** and **Reddit RSS** for Bitcoin headlines
3. Scores sentiment with **VADER** (no GPU / fast)
4. Builds a **HAR** model on daily realized variance (RV)
5. Produces 1D/7D/30D price distributions (Monte Carlo) and charts
6. Renders a **LaTeX PDF report** (using Tectonic), plus a JSON summary

Outputs land in the `out/` folder and are committed back to the repo on every run.

## Quick start
- Create a new GitHub repo and upload these files.
- Enable Actions (no secrets needed).
- The workflow runs every **6 hours** (see `.github/workflows/forecast.yml`).

## Overleaf (optional)
- If you have Overleaf Git integration, add your Overleaf remote and push `out/main.tex` + PNGs.
- Or just use the compiled `out/main.pdf` produced by Tectonic.

## Local run
```bash
pip install -r requirements.txt
python pipeline.py
```

## Notes
- This pipeline avoids heavyweight dependencies (FinBERT, ccxt) to run reliably in CI.
- You can plug in your richer intraday HAR or FinBERT by editing `pipeline.py`.
