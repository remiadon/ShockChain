"""
Build the numeric feature matrix X_num and continuous regression targets y.

Pipeline:
  1. Download daily OHLC from Yahoo (10 tickers) and macro series from FRED.
  2. Merge to a daily polars frame (forward-fill weekly/monthly FRED series).
  3. Derive 16 market-state features.
  4. Standardize each via a trailing 252-day z-score (min_periods=60).
  5. Compute forward moves at t+1/+5/+15 for SPX, VIX, DXY, WTI, US10Y and
     emit BOTH the raw moves and vol-normalized moves (move / trailing_60d_vol)
     as regression targets.
  6. Align each event in events_verified.csv to the next available trading day
     and assemble training_dataset.parquet + X_num.npy.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
EVENTS_CSV = OUTPUT_DIR / "events_verified.csv"
RAW_PARQUET = OUTPUT_DIR / "market_data_raw.parquet"
TRAIN_PARQUET = OUTPUT_DIR / "training_dataset.parquet"
XNUM_NPY = OUTPUT_DIR / "X_num.npy"

START_DATE = "2014-06-01"
END_DATE = "2025-03-01"

YF_TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "DXY": "DX-Y.NYB",
    "WTI": "CL=F",
    "US10Y": "^TNX",
    "GOLD": "GC=F",
    "TLT": "TLT",
    "HYG": "HYG",
    "LQD": "LQD",
}

FRED_SERIES = {
    "T10Y2Y": "T10Y2Y",
    "REAL_YIELD": "DFII10",
    "FED_FUNDS": "FEDFUNDS",
}

MARKET_FEATURES = [
    "vix_level", "vix_pct_1y", "spx_rvol_20d", "vix_term_slope",
    "hy_oas", "ig_oas", "hy_ig_diff",
    "us10y_level", "t10y2y", "real_yield", "fed_funds",
    "dxy_level", "spx_ret_20d", "gold_ret_20d", "wti_level", "cross_corr_20d",
]

TARGETS = {
    "SPX": "ret",     # pct change
    "VIX": "diff",    # absolute change
    "DXY": "ret",
    "WTI": "ret",
    "US10Y": "bps",   # absolute change in bps (multiply by 10)
}
HORIZONS = [1, 5, 15]


def download_yfinance() -> pl.DataFrame:
    print(f"Downloading {len(YF_TICKERS)} tickers from Yahoo Finance...")
    raw = yf.download(
        list(YF_TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if isinstance(raw.columns, __import__("pandas").MultiIndex):
        closes = {}
        for name, ticker in YF_TICKERS.items():
            if (ticker, "Close") in raw.columns:
                closes[name] = raw[(ticker, "Close")]
            else:
                closes[name] = raw[ticker]["Close"]
        import pandas as pd
        pdf = pd.DataFrame(closes)
    else:
        import pandas as pd
        pdf = raw[["Close"]].rename(columns={"Close": list(YF_TICKERS.keys())[0]})

    pdf = pdf.reset_index().rename(columns={"Date": "date"})
    df = pl.from_pandas(pdf).with_columns(pl.col("date").cast(pl.Date))
    return df.sort("date")


def download_fred(api_key: str) -> pl.DataFrame:
    print(f"Downloading {len(FRED_SERIES)} series from FRED...")
    fred = Fred(api_key=api_key)
    import pandas as pd

    frames = []
    for name, sid in FRED_SERIES.items():
        s = fred.get_series(sid, observation_start=START_DATE, observation_end=END_DATE)
        s.index.name = "date"
        s.name = name
        frames.append(s)
    pdf = pd.concat(frames, axis=1).reset_index()
    df = pl.from_pandas(pdf).with_columns(pl.col("date").cast(pl.Date))
    return df.sort("date")


def merge_market_data(yf_df: pl.DataFrame, fred_df: pl.DataFrame) -> pl.DataFrame:
    merged = yf_df.join(fred_df, on="date", how="left").sort("date")
    fill_cols = list(YF_TICKERS.keys()) + list(FRED_SERIES.keys())
    merged = merged.with_columns([pl.col(c).forward_fill() for c in fill_cols])
    return merged


def rolling_pct_rank(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        win = values[lo:i + 1]
        win = win[~np.isnan(win)]
        v = values[i]
        if not np.isnan(v) and len(win) > 0:
            out[i] = (win < v).sum() / len(win)
    return out


def rolling_mean_pairwise_corr(returns: np.ndarray, window: int) -> np.ndarray:
    """returns shape (n, k); compute mean of all pairwise rolling corrs."""
    n, k = returns.shape
    pairs = list(combinations(range(k), 2))
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        if i - lo + 1 < window:
            continue
        win = returns[lo:i + 1]
        if np.isnan(win).any():
            continue
        cors = []
        for a, b in pairs:
            ca = win[:, a] - win[:, a].mean()
            cb = win[:, b] - win[:, b].mean()
            denom = np.sqrt((ca ** 2).sum() * (cb ** 2).sum())
            if denom > 0:
                cors.append((ca * cb).sum() / denom)
        if cors:
            out[i] = float(np.mean(cors))
    return out


def compute_market_state(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("SPX").pct_change().alias("spx_ret"),
        pl.col("GOLD").pct_change().alias("gold_ret"),
        pl.col("TLT").pct_change().alias("tlt_ret"),
        pl.col("WTI").pct_change().alias("wti_ret"),
    ])

    df = df.with_columns([
        pl.col("VIX").alias("vix_level"),
        (pl.col("spx_ret").rolling_std(window_size=20) * np.sqrt(252)).alias("spx_rvol_20d"),
        (pl.col("VIX") - pl.col("VIX3M")).alias("vix_term_slope"),
        # FRED's ICE BofA OAS series only return 2023+ via the public API after
        # a 2024 licensing change. Proxy with HYG/LQD ETF 252d drawdown:
        # drawdown rises when spreads widen, so direction matches OAS.
        (1 - pl.col("HYG") / pl.col("HYG").rolling_max(window_size=252, min_samples=20)).alias("hy_oas"),
        (1 - pl.col("LQD") / pl.col("LQD").rolling_max(window_size=252, min_samples=20)).alias("ig_oas"),
        (
            (1 - pl.col("HYG") / pl.col("HYG").rolling_max(window_size=252, min_samples=20))
            - (1 - pl.col("LQD") / pl.col("LQD").rolling_max(window_size=252, min_samples=20))
        ).alias("hy_ig_diff"),
        pl.col("US10Y").alias("us10y_level"),
        pl.col("T10Y2Y").alias("t10y2y"),
        pl.col("REAL_YIELD").alias("real_yield"),
        pl.col("FED_FUNDS").alias("fed_funds"),
        pl.col("DXY").alias("dxy_level"),
        ((pl.col("SPX") / pl.col("SPX").shift(20)) - 1).alias("spx_ret_20d"),
        ((pl.col("GOLD") / pl.col("GOLD").shift(20)) - 1).alias("gold_ret_20d"),
        pl.col("WTI").alias("wti_level"),
    ])

    vix_pct = rolling_pct_rank(df["VIX"].to_numpy().astype(float), 252)
    df = df.with_columns(pl.Series("vix_pct_1y", vix_pct))

    rets = np.column_stack([
        df["spx_ret"].to_numpy().astype(float),
        df["tlt_ret"].to_numpy().astype(float),
        df["gold_ret"].to_numpy().astype(float),
        df["wti_ret"].to_numpy().astype(float),
    ])
    cross = rolling_mean_pairwise_corr(rets, 20)
    df = df.with_columns(pl.Series("cross_corr_20d", cross))

    return df


def standardize_trailing(df: pl.DataFrame, cols: list[str], window: int = 252, min_periods: int = 60) -> pl.DataFrame:
    exprs = []
    for c in cols:
        m = pl.col(c).rolling_mean(window_size=window, min_samples=min_periods)
        s = pl.col(c).rolling_std(window_size=window, min_samples=min_periods)
        # When the trailing window has zero variance (e.g., fed_funds pinned
        # at 0 for years) the z-score is undefined — emit 0 (no signal).
        z = pl.when(s > 1e-12).then((pl.col(c) - m) / s).otherwise(0.0)
        exprs.append(z.alias(f"{c}_z"))
    return df.with_columns(exprs)


def compute_targets(df: pl.DataFrame) -> pl.DataFrame:
    spx_ret = pl.col("SPX").pct_change()
    vix_diff = pl.col("VIX").diff()
    dxy_ret = pl.col("DXY").pct_change()
    wti_ret = pl.col("WTI").pct_change()
    us10y_diff_bps = pl.col("US10Y").diff() * 10.0

    df = df.with_columns([
        spx_ret.rolling_std(window_size=60, min_samples=20).alias("_vol_SPX"),
        vix_diff.rolling_std(window_size=60, min_samples=20).alias("_vol_VIX"),
        dxy_ret.rolling_std(window_size=60, min_samples=20).alias("_vol_DXY"),
        wti_ret.rolling_std(window_size=60, min_samples=20).alias("_vol_WTI"),
        us10y_diff_bps.rolling_std(window_size=60, min_samples=20).alias("_vol_US10Y"),
    ])

    raw_exprs = []
    for h in HORIZONS:
        raw_exprs.append((pl.col("SPX").shift(-h) / pl.col("SPX") - 1).alias(f"SPX_{h}d_raw"))
        raw_exprs.append((pl.col("VIX").shift(-h) - pl.col("VIX")).alias(f"VIX_{h}d_raw"))
        raw_exprs.append((pl.col("DXY").shift(-h) / pl.col("DXY") - 1).alias(f"DXY_{h}d_raw"))
        raw_exprs.append((pl.col("WTI").shift(-h) / pl.col("WTI") - 1).alias(f"WTI_{h}d_raw"))
        raw_exprs.append(((pl.col("US10Y").shift(-h) - pl.col("US10Y")) * 10.0).alias(f"US10Y_{h}d_raw"))
    df = df.with_columns(raw_exprs)

    z_exprs = []
    for ind in TARGETS:
        vol = pl.col(f"_vol_{ind}")
        for h in HORIZONS:
            # Divide by sqrt(h) so each horizon has unit variance under a
            # random-walk null. Then ŷ=0 has RMSE≈1 across all horizons.
            denom = vol * float(np.sqrt(h))
            z = pl.when(denom > 1e-12).then(pl.col(f"{ind}_{h}d_raw") / denom).otherwise(None)
            z_exprs.append(z.cast(pl.Float32).alias(f"{ind}_{h}d_z"))
    df = df.with_columns(z_exprs)

    df = df.drop([f"_vol_{ind}" for ind in TARGETS])
    return df


def align_events(events: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    events = events.with_columns(pl.col("date").str.to_date(strict=False).alias("event_date")).sort("event_date")
    market_dates = market.select(pl.col("date").alias("market_date")).sort("market_date")
    aligned = events.join_asof(
        market_dates, left_on="event_date", right_on="market_date", strategy="forward",
    )
    return aligned


def main():
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    import os
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError(f"FRED_API_KEY not set in {ENV_PATH}")

    yf_df = download_yfinance()
    fred_df = download_fred(fred_key)
    market = merge_market_data(yf_df, fred_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market.write_parquet(RAW_PARQUET)
    print(f"Saved raw market data -> {RAW_PARQUET} ({market.shape[0]} rows, {market.shape[1]} cols)")

    market = compute_market_state(market)
    market = standardize_trailing(market, MARKET_FEATURES)
    market = compute_targets(market)

    z_cols = [f"{c}_z" for c in MARKET_FEATURES]
    target_raw_cols = [f"{ind}_{h}d_raw" for ind in TARGETS for h in HORIZONS]
    target_z_cols = [f"{ind}_{h}d_z" for ind in TARGETS for h in HORIZONS]
    target_cols = target_raw_cols + target_z_cols

    keep = ["date"] + z_cols + target_cols
    market_slim = market.select(keep)

    events = pl.read_csv(EVENTS_CSV).with_row_index("event_id")
    n_events = events.height
    aligned = align_events(events, market_slim)
    matched = aligned.filter(pl.col("market_date").is_not_null()).height
    print(f"\nEvents: {n_events}; matched to a trading day: {matched}; unmatched: {n_events - matched}")

    aligned = aligned.join(market_slim, left_on="market_date", right_on="date", how="left")

    all_feature_cols = z_cols + target_cols
    n_before_drop = aligned.height
    aligned = aligned.drop_nulls(subset=all_feature_cols)
    dropped = n_before_drop - aligned.height
    print(f"Dropped {dropped} events with null features/targets after fills")

    final = aligned.select([
        pl.col("event_id"),
        pl.col("headline"),
        pl.col("event_date").alias("date"),
        pl.col("market_date"),
        pl.col("impact"),
        pl.col("category"),
        *[pl.col(c) for c in z_cols],
        *[pl.col(c) for c in target_cols],
    ])
    final.write_parquet(TRAIN_PARQUET)
    print(f"Saved training dataset -> {TRAIN_PARQUET} ({final.shape[0]} rows, {final.shape[1]} cols)")

    X_num = final.select(z_cols).to_numpy().astype(np.float32)
    np.save(XNUM_NPY, X_num)
    print(f"Saved X_num -> {XNUM_NPY} (shape {X_num.shape})")

    print("\n--- 1-day regression target stats ---")
    print(f"  {'indicator':8s} {'kind':5s} {'mean':>9s} {'std':>9s} {'min':>9s} {'p05':>9s} {'p95':>9s} {'max':>9s}")
    for ind in TARGETS:
        for kind in ("raw", "z"):
            col = f"{ind}_1d_{kind}"
            stats = final.select([
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"),
                pl.col(col).quantile(0.05).alias("p05"),
                pl.col(col).quantile(0.95).alias("p95"),
                pl.col(col).max().alias("max"),
            ]).row(0, named=True)
            print(
                f"  {ind:8s} {kind:5s} "
                f"{stats['mean']:>9.4f} {stats['std']:>9.4f} {stats['min']:>9.4f} "
                f"{stats['p05']:>9.4f} {stats['p95']:>9.4f} {stats['max']:>9.4f}"
            )

    print("\n--- Features with >5% NaN after alignment ---")
    n = final.height
    nan_report = []
    for c in z_cols + target_cols:
        nulls = final[c].null_count()
        pct = nulls / n * 100
        if pct > 5:
            nan_report.append((c, pct))
    if nan_report:
        for c, p in nan_report:
            print(f"  {c}: {p:.2f}%")
    else:
        print("  none")

    dmin = final["date"].min()
    dmax = final["date"].max()
    print(f"\nFinal dataset date range: {dmin} -> {dmax}")


if __name__ == "__main__":
    main()
