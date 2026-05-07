"""
Comparison report: print tables, save JSON, plot bar chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_JSON = OUTPUT_DIR / "results.json"
COMPARISON_PNG = OUTPUT_DIR / "model_comparison.png"

MODEL_ORDER = ["AlwaysZero", "SentimentOnly", "MarketStateOnly", "IndependentHeads", "ShockChain"]


def _row(model_name: str, metrics: dict) -> str:
    parts = [f"{model_name:<19s}"]
    for h in (1, 5, 15):
        m = metrics["per_horizon"][h]
        rmse = m["avg_rmse"]
        dir_pct = m["avg_dir_acc"] * 100
        parts.append(f"{rmse:>7.4f} | {dir_pct:>5.1f}%")
    return " | ".join(parts)


def _print_comparison(results: dict):
    header = (
        f"{'':<19s} | {'1d RMSE | 1d Dir%':>17s} | "
        f"{'5d RMSE | 5d Dir%':>17s} | {'15d RMSE | 15d Dir%':>18s}"
    )
    print("\n=== Model comparison (test set) ===")
    print(header)
    print("-" * len(header))
    for name in MODEL_ORDER:
        if name in results:
            print(_row(name, results[name]))


def _print_shockchain_breakdown(results: dict, indicators: list[str]):
    if "ShockChain" not in results:
        return
    pih = results["ShockChain"]["per_indicator_horizon"]
    print("\n=== ShockChain per-indicator (test set) ===")
    header = (
        f"{'':<7s} | {'1d RMSE':>7s} | {'1d Dir%':>7s} | {'1d R²':>6s} | "
        f"{'5d RMSE':>7s} | {'5d Dir%':>7s} | {'15d RMSE':>8s} | {'15d Dir%':>8s}"
    )
    print(header)
    print("-" * len(header))
    for ind in indicators:
        m1 = pih[ind][1]; m5 = pih[ind][5]; m15 = pih[ind][15]
        print(
            f"{ind:<7s} | {m1['rmse']:>7.4f} | {m1['dir_acc']*100:>6.1f}% | "
            f"{m1['r2']:>6.3f} | {m5['rmse']:>7.4f} | {m5['dir_acc']*100:>6.1f}% | "
            f"{m15['rmse']:>8.4f} | {m15['dir_acc']*100:>7.1f}%"
        )


def _print_horizon_degradation(results: dict):
    if "ShockChain" not in results:
        return
    ph = results["ShockChain"]["per_horizon"]
    ratio = ph[15]["avg_rmse"] / ph[1]["avg_rmse"]
    note = "(close to 1.0 is good; >1.5 suggests error compounding)"
    print(f"\nShockChain horizon degradation 15d/1d RMSE: {ratio:.3f}  {note}")


def _print_assessment(results: dict):
    if "ShockChain" not in results:
        return
    sc = results["ShockChain"]["overall"]["avg_rmse"]
    sc_dir1 = results["ShockChain"]["per_horizon"][1]["avg_dir_acc"] * 100

    def beat(other: str) -> str:
        if other not in results:
            return "n/a"
        other_rmse = results[other]["overall"]["avg_rmse"]
        return "PASS" if sc < other_rmse else "FAIL"

    print("\n=== Assessment ===")
    print(f"  ShockChain beats AlwaysZero (RMSE):       {beat('AlwaysZero')}")
    print(f"  ShockChain beats SentimentOnly (RMSE):    {beat('SentimentOnly')}")
    print(f"  ShockChain beats MarketStateOnly (RMSE):  {beat('MarketStateOnly')}")
    print(f"  ShockChain beats IndependentHeads (RMSE): {beat('IndependentHeads')}")
    print(f"  ShockChain 1d directional > 55%:          {'PASS' if sc_dir1 > 55 else 'FAIL'} ({sc_dir1:.1f}%)")


def _save_json(results: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {RESULTS_JSON}")


def _plot_comparison(results: dict):
    horizons = [1, 5, 15]
    models = [m for m in MODEL_ORDER if m in results]
    n_models = len(models)
    x = np.arange(len(horizons))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(models):
        rmses = [results[name]["per_horizon"][h]["avg_rmse"] for h in horizons]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rmses, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_ylabel("Avg RMSE (vol-normalized targets)")
    ax.set_title("ShockChain vs baselines — test set RMSE by horizon")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart -> {COMPARISON_PNG}")


def generate_report(results: dict, indicators: list[str] | None = None):
    if indicators is None:
        from dataset import INDICATORS as _IND
        indicators = _IND
    _print_comparison(results)
    _print_shockchain_breakdown(results, indicators)
    _print_horizon_degradation(results)
    _print_assessment(results)
    _save_json(results)
    _plot_comparison(results)


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    # JSON loads horizon keys as strings; coerce back to int for printing.
    for name, r in results.items():
        r["per_horizon"] = {int(k): v for k, v in r["per_horizon"].items()}
        for ind, h_map in r["per_indicator_horizon"].items():
            r["per_indicator_horizon"][ind] = {int(k): v for k, v in h_map.items()}
    generate_report(results)
