"""
Embed event headlines with sentence-transformers/all-mpnet-base-v2.

Reads output/events_verified.csv, encodes the `headline` column into 768-dim
normalized vectors, and writes:
  - output/embeddings.npy                (n_events x 768 numpy array)
  - output/events_with_embeddings.parquet (original columns + emb_0..emb_767)
"""

from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_CSV = OUTPUT_DIR / "events_verified.csv"
EMBEDDINGS_NPY = OUTPUT_DIR / "embeddings.npy"
EVENTS_PARQUET = OUTPUT_DIR / "events_with_embeddings.parquet"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 128
EMBED_DIM = 768


def main():
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)

    events = pl.read_csv(INPUT_CSV)
    headlines = events["headline"].to_list()
    print(f"Loaded {len(headlines)} events from {INPUT_CSV}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        headlines,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Embedding shape: {embeddings.shape}")

    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"Saved embeddings -> {EMBEDDINGS_NPY}")

    emb_df = pl.from_numpy(
        embeddings,
        schema=[f"emb_{i}" for i in range(EMBED_DIM)],
    )
    combined = pl.concat([events, emb_df], how="horizontal")
    combined.write_parquet(EVENTS_PARQUET)
    print(f"Saved events + embeddings -> {EVENTS_PARQUET} ({combined.shape[0]} rows, {combined.shape[1]} cols)")


if __name__ == "__main__":
    main()
