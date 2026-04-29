"""
Sanity-check the headline embeddings via t-SNE.

Loads output/embeddings.npy and output/events_verified.csv, projects the
embeddings to 2D with t-SNE, and writes a side-by-side scatter plot
(category vs. impact) to output/embedding_tsne.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_CSV = OUTPUT_DIR / "events_verified.csv"
EMBEDDINGS_NPY = OUTPUT_DIR / "embeddings.npy"
TSNE_PNG = OUTPUT_DIR / "embedding_tsne.png"

IMPACT_COLORS = {"Low": "green", "Moderate": "orange", "High": "red"}


def main():
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)

    events = pl.read_csv(INPUT_CSV)
    embeddings = np.load(EMBEDDINGS_NPY)
    assert len(events) == embeddings.shape[0], (
        f"Row mismatch: {len(events)} events vs {embeddings.shape[0]} embeddings"
    )
    print(f"Loaded {len(events)} events, embeddings shape {embeddings.shape}")

    print("Running t-SNE (this can take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    coords = tsne.fit_transform(embeddings)

    categories = events["category"].to_list()
    impacts = events["impact"].to_list()

    fig, (ax_cat, ax_imp) = plt.subplots(1, 2, figsize=(18, 8))

    unique_cats = sorted(set(categories))
    cmap = plt.get_cmap("tab20", len(unique_cats))
    cat_to_color = {cat: cmap(i) for i, cat in enumerate(unique_cats)}
    cat_colors = [cat_to_color[c] for c in categories]
    ax_cat.scatter(coords[:, 0], coords[:, 1], c=cat_colors, s=8, alpha=0.7)
    ax_cat.set_title("t-SNE by category")
    ax_cat.set_xlabel("t-SNE 1")
    ax_cat.set_ylabel("t-SNE 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_to_color[c], markersize=6, label=c)
        for c in unique_cats
    ]
    ax_cat.legend(handles=handles, fontsize=7, loc="best", framealpha=0.85)

    impact_colors = [IMPACT_COLORS.get(i, "gray") for i in impacts]
    ax_imp.scatter(coords[:, 0], coords[:, 1], c=impact_colors, s=8, alpha=0.7)
    ax_imp.set_title("t-SNE by impact")
    ax_imp.set_xlabel("t-SNE 1")
    ax_imp.set_ylabel("t-SNE 2")
    imp_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in IMPACT_COLORS.items()
    ]
    ax_imp.legend(handles=imp_handles, fontsize=9, loc="best", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(TSNE_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE figure -> {TSNE_PNG}")

    cat_counts = events.group_by("category").len().sort("len", descending=True)
    impact_counts = events.group_by("impact").len().sort("len", descending=True)

    print("\nEvents per category:")
    for row in cat_counts.iter_rows(named=True):
        print(f"  {row['category']}: {row['len']}")

    print("\nEvents per impact level:")
    for row in impact_counts.iter_rows(named=True):
        print(f"  {row['impact']}: {row['len']}")


if __name__ == "__main__":
    main()
