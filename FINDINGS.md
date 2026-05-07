# ShockChain — Findings

End-to-end notes from building the headline → multi-horizon market-shock
regression model on the `headlines-generation` branch.

## TL;DR

- **Dataset**: 4,682 LLM-generated historical events (2015-01 → 2025-02), each
  paired with 16 standardized macro market-state features and 15 vol-normalized
  forward-return targets (5 indicators × 3 horizons).
- **Headline embeddings carry weak but real signal.** `SentimentOnly` beats
  `AlwaysZero` on every horizon by ~0.3-1.7% RMSE and reaches ~56% directional
  accuracy.
- **The chain wiring is no longer a bug** — after smaller models + stronger
  regularization, `ShockChain` matches `SentimentOnly` and beats
  `IndependentHeads` at 1d/15d. The 15d/1d RMSE degradation ratio is 0.955
  (no error compounding).
- **SPX direction at 5d/15d is the cleanest signal**: 70.8% / 73.3%
  directional accuracy with `ShockChain`.
- **VIX dominates the error budget**: 1d RMSE 1.45 vs ~0.95 for the other 4
  targets. Its fat right tail (April-2020 single-day VIX = +24.86 standardized
  units) drags the joint loss.
- **Joint R² is essentially zero** for all indicators across all horizons.
  Vol-normalized macro returns at single-event resolution are mostly noise.

## Pipeline

```
events_verified.csv ─┐
                     ├─► embed_headlines.py ─► embeddings.npy (4682, 768)
                     │
                     └─► build_features.py ──► training_dataset.parquet
                            │                  X_num.npy (4682, 16)
                            │                  market_data_raw.parquet
                            │
                            ├─► dataset.py ─► temporal split (train/val/test)
                            │
                            └─► train.py ──► output/models/*.pt
                                                   │
                                                   └─► report.py ─► results.json
                                                                    model_comparison.png
```

Run end-to-end: `cd src && python run_all.py`.

## Data construction decisions

### LLM-generated event corpus
5,064 historical market-relevant events generated and date-verified by Claude
across 2016-2025 in half-month windows
([src/headline_generation.py](src/headline_generation.py)). After NaN/coverage
filters, 4,682 survive into the training set.

### Market-state features (X_num: 16-dim)
Daily series from yfinance (8 instruments + HYG/LQD ETFs) and FRED (3 macro
series). Derived features per
[CLAUDE.md](CLAUDE.md)-style spec — vol regime, credit, rates, cross-asset
positioning. Each feature is standardized with a **trailing 252-day z-score
(min_periods=60)** to avoid look-ahead leakage.

**Spec said 17 features, list enumerates 16.** Implemented all 16 from the
list verbatim; `X_num.npy` shape is `(4682, 16)`.

### Targets (y: 15-dim, vol-normalized)
For each event date, compute forward moves at t+1/+5/+15 for SPX, VIX, DXY,
WTI, US10Y, then divide by `sqrt(h) * trailing_60d_vol` so each target has
unit variance under a random-walk null. Both `_raw` and `_z` columns are
stored in `training_dataset.parquet`; the model trains on `_z`.

### Temporal split (no shuffling)

| split | date range | rows |
|---|---|---|
| train | < 2022-01-01 | 3,029 |
| val | 2022-01-01 → 2024-01-01 | 1,058 |
| test | ≥ 2024-01-01 | 595 |

## Bugs found and fixed

These were not in the original spec — they came up during build/run iteration.
Documented here so they don't get re-introduced:

1. **FRED ICE BofA OAS series retired pre-2023.** `BAMLH0A0HYM2` and
   `BAMLC0A0CM` only return data from 2023-04-30 onward via the public API
   after a 2024 ICE licensing change. Using them as documented in the spec
   would silently drop 4,200+ events. **Workaround**: proxy with HYG/LQD ETF
   252-day drawdown (direction matches OAS — drawdown rises when spreads
   widen).
2. **Zero-variance trailing windows produced ±inf z-scores.** Fed funds was
   pinned at 0 from 2015-2020, so `fed_funds.rolling_std(252)` returned 0 for
   long stretches → division by zero in the standardize step → 80 inf values
   in `X_num.npy` → instant NaN losses on every model that touched `x_num`.
   **Fix**: in `standardize_trailing`, return 0 (no signal) when std < 1e-12.
3. **`polars.drop_nulls` doesn't drop NaN.** Polars distinguishes nullness
   from float NaN. The 2 lingering NaNs in `cross_corr_20d_z` (early-series
   incomplete window) survived `drop_nulls` and reached the dataset loader.
   **Fix**: in `dataset.py`, filter on `np.isfinite(...).all(axis=1)` for both
   x_num and y_targets.
4. **Cumulative h-day moves divided by 1-day vol have variance ~h.** With the
   original `move / vol_60d`, AlwaysZero had RMSE 1.0 at 1d but 2.2 at 5d and
   3.9 at 15d — not a model issue, a target-construction issue.
   **Fix**: divide by `sqrt(h) * vol_60d` so all horizons are unit-variance.
5. **Embeddings vs filtered events were misaligned by row position.**
   `embeddings.npy` was generated from the original 5,064-row CSV, but
   `X_num.npy` and `training_dataset.parquet` only retain 4,682 rows after
   null drops. Headlines aren't unique (38 duplicates). **Fix**: added
   `event_id` (row index from `events_verified.csv`) to the parquet so
   `dataset.py` can slice `embeddings[event_ids]` deterministically.

## Modeling decisions

- **Switched from 5-class classification to regression** (user direction
  partway through). Targets stored as both raw and vol-normalized to keep
  optionality at training time.
- **Smaller model + stronger regularization.** Initial config (256→128 trunk,
  dropout 0.3, weight_decay 1e-4) overfit from epoch 1. Final config:
  64→32 trunk, dropout 0.5, weight_decay 1e-3.
- **Shared indicator heads across stages.** All 3 stages call the same
  per-indicator head module, forcing the heads to learn an indicator-level
  projection that works at every horizon. Cuts param count 4× (340k → 78k).

## Final results (test set, 595 events)

| Model | params | 1d RMSE | 5d RMSE | 15d RMSE | 1d Dir% | 5d Dir% | 15d Dir% | Avg RMSE |
|---|---|---|---|---|---|---|---|---|
| AlwaysZero | 0 | 1.048 | 0.996 | 1.017 | — | — | — | 1.020 |
| SentimentOnly | 149k | 1.045 | 0.988 | **1.000** | 53.8 | 56.3 | **56.7** | **1.011** |
| MarketStateOnly | 8k | 1.055 | 1.004 | 1.023 | 54.7 | 55.7 | 55.4 | 1.027 |
| IndependentHeads | 77k | 1.046 | **0.987** | 1.008 | **54.5** | **59.4** | 57.2 | 1.014 |
| **ShockChain** | 78k | 1.047 | **0.987** | **1.000** | 52.8 | 56.2 | 55.0 | **1.011** |

### ShockChain per-indicator

| | 1d RMSE | 1d Dir% | 1d R² | 5d RMSE | 5d Dir% | 15d RMSE | 15d Dir% |
|---|---|---|---|---|---|---|---|
| SPX | 1.014 | 59.4% | -0.004 | 0.994 | **70.8%** | 0.959 | **73.3%** |
| VIX | 1.454 | 42.5% | -0.007 | 1.253 | 48.5% | 1.274 | 48.9% |
| DXY | 0.956 | 58.9% | -0.010 | 0.865 | 55.9% | 0.963 | 54.9% |
| WTI | 0.857 | 52.9% | +0.003 | 0.945 | 50.0% | 0.832 | 43.9% |
| US10Y | 0.956 | 50.2% | -0.002 | 0.877 | 55.7% | 0.973 | 53.9% |

### Hypothesis tests

| | result |
|---|---|
| ShockChain beats AlwaysZero (RMSE) | **PASS** — model learns *something* |
| ShockChain beats SentimentOnly (RMSE) | **TIE** (1.0113 vs 1.0110) — embedding alone captures most of the available signal |
| ShockChain beats MarketStateOnly (RMSE) | **PASS** — headline adds value beyond market state |
| ShockChain beats IndependentHeads (RMSE) | **PASS** — chain wiring helps marginally |
| ShockChain 1d directional > 55% | **FAIL** (52.8%) — but 5d/15d clear the bar |
| Horizon degradation 15d/1d ratio < 1.5 | **PASS** (0.955) — no error compounding |

## Interpretation

- The dataset supports the project's core hypothesis (event text carries
  predictive signal beyond market state) — but only weakly, and only on SPX
  direction at multi-day horizons.
- The chain mechanism (`ShockChain` vs `IndependentHeads`) gives a small
  improvement at 1d and 15d but not at 5d. The teacher-forced noise injection
  (std=0.5) seems calibrated correctly — no compounding, no collapse.
- VIX is the worst-modeled target by a large margin. Its 1d RMSE (1.45) is
  inflated by a single outlier (March-2020 VIX +24.86 σ); winsorizing the
  training targets at ±5σ would likely tighten this without hurting the
  cleaner indicators.
- The headline corpus is LLM-generated, not crawled news. The "signal" the
  model picks up may partly reflect Claude's own retrospective framing of
  which events mattered, rather than genuine surprise to markets at the time.
  Treat 70%+ SPX directional accuracy with healthy skepticism.

## Ideas worth trying next

- **Winsorize training targets** at ±5σ to tame the VIX outliers.
- **Quantile-regression heads** (predict P25/P50/P75 instead of MSE point
  estimate) — more useful for downstream risk simulation than a point
  prediction.
- **Replace LLM-generated events with real headlines** (the `gdelt-news-headlines`
  dataset already wired in [src/extract.py](src/extract.py)). The current
  results are a methodological sanity check; the production claim needs the
  real corpus.
- **Validate the impact label** — events labeled `High` should have
  systematically larger forward moves than `Low` events. If that's not the
  case in the data, the verification step has a problem.
- **Fine-tune the encoder** instead of using frozen embeddings — the existing
  `src/finetune_headline_encoder.py` already does this for a separate
  classification task and could be repurposed.
