import polars as pl
import polars.selectors as cs
import yfinance_pl as yf
import argparse
from inspect import getmembers, isfunction

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="sp500_targets", help="Output file path")

def classif_targets(
        tickers=('^GSPC', '^VIX', 'DX-Y.NYB', 'WTI', '^TNX'),
        cut_points=(-1, -0.1, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.1, 1),
        interval='1d'
    ):
    _l = list()
    for ticker in tickers:
        tick = yf.Ticker(ticker)
        hist = tick.history(period='10y', interval=interval).select(
            pl.col.date.dt.date(),
            pl.col('close.amount').alias(ticker)
        )
        _l.append(hist)
    df = pl.concat(_l, how='align_full').drop_nulls()
    var = pl.col(tickers).pct_change()
    zscore = (var - var.rolling_mean(60)) / var.rolling_std(60)
    targets = df.select('date',
        zscore.cut(
            [-2, -1, 1, 2], 
            labels=['strong negative', 'negative', 'neutral', 'positive', 'strong positive']
        ).name.keep()
    ).drop_nulls()
    with pl.Config(tbl_cols=10, tbl_formatting='MARKDOWN', tbl_rows=300):
        print(
            "Value counts for each target column:\n",
            targets.unpivot(on=cs.categorical())\
                .group_by(target='variable', cat='value', maintain_order=True).agg(pl.len())
        )
    return targets

def news_headlines():
    df = pl.read_ndjson('hf://datasets/olm/gdelt-news-headlines/**/*.jsonl')
    df = df.group_by('title').agg(date=pl.col.article_date.str.to_date('%Y%m%d').min())
    return df



if __name__ == "__main__":
    args = parser.parse_args()
    current_functions = dict(getmembers(__import__(__name__), isfunction))
    if args.output in current_functions:
        current_functions[args.output]().write_parquet(f"data/{args.output}.parquet")
    else:
        raise ValueError(f"Function {args.output} not found in the current module.")
