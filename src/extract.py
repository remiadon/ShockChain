import polars as pl
import yfinance_pl as yf
import argparse
from inspect import getmembers, isfunction

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="sp500_targets", help="Output file path")

def sp500_targets():
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period='10y', interval='1d').select(pl.col.date.dt.date(), 'volume', price='close.amount')
    sp500_cuts = sp500.with_columns(
        pl.col.price.pct_change().cut([-1, -0.1, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.1, 1])
    )
    targets = sp500_cuts.to_dummies('price') # TODO remove dummy encoding, pytorch can handle categorical labels directly with CrossEntropyLoss and no need for one-hot encoding. Just use the cut labels as categorical targets.
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
