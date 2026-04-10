import polars as pl
import yfinance_pl as yf

def extract_sp500_targets():
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period='10y', interval='1d').select('date', 'volume', price='close.amount')

    sp500_cuts = sp500.with_columns(pl.col.price.pct_change().cut([0.01, 0.03, 0.05, 0.1, 1]))
    targets = sp500_cuts.pivot('price', index='date', aggregate_function=pl.element().is_not_null().first().fill_null(False))
    return targets

if __name__ == "__main__":
    extract_sp500_targets().write_csv("data/sp500_targets.csv")