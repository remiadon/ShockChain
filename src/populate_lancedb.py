"""
Populate a LanceDB table with learned headline embeddings.

Schema
------
  id       : int      — row index
  date     : str      — ISO date string (YYYY-MM-DD)
  headline : str      — raw headline text  ← full-text search (BM25)
  vector   : float[]  — mean-pooled encoder embedding  ← ANN vector search

After this step you get hybrid search for free:
    tbl.search("inflation fears", query_type="hybrid")
       .limit(10)
       .to_list()
"""

from __future__ import annotations

import argparse

import dvc.api
import lancedb
import polars as pl
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.finetune_headline_encoder import HeadlineClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--headlines", default="data/news_headlines.parquet")
parser.add_argument("--checkpoint", default="models/headline_classifier.pt")
parser.add_argument("--output", type=str)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_all(
    texts: list[str],
    model: HeadlineClassifier,
    tokenizer,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    model.eval()
    all_embs: list[list[float]] = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="encoding", unit="batch", dynamic_ncols=True):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            _, emb = model(**enc)
        all_embs.extend(emb.cpu().tolist())
    return all_embs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    max_length = dvc.api.params_show()["train"]["max_length"]
    encode_batch_size = int(dvc.api.params_show()["populate_lancedb"]["encode_batch_size"])
    table_name = dvc.api.params_show()["populate_lancedb"]["table_name"]

    # ---- load model ----
    # torch.load unpickles the full model object without calling __init__,
    # so AutoModel.from_pretrained() is never invoked here.
    # HeadlineClassifier must be imported above so pickle can find the class.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    # weights_only=False is required to unpickle the full model object.
    # Safe here because the checkpoint is produced by our own training script.
    model  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.encoder.config._name_or_path)
    print(f"Loaded checkpoint: {args.checkpoint}  (base: {model.encoder.config._name_or_path})")

    # ---- load headlines ----
    df = pl.read_parquet(args.headlines).drop_nulls("title")
    texts = df["title"].to_list()
    dates = df["date"].cast(pl.Utf8).to_list()
    print(f"Encoding {len(texts):,} headlines …")

    # ---- encode ----
    vectors = encode_all(texts, model, tokenizer, max_length, device, encode_batch_size)

    # ---- write LanceDB ----
    db = lancedb.connect(args.output)

    records = [
        {"id": i, "date": dates[i], "headline": texts[i], "vector": vectors[i]}
        for i in range(len(texts))
    ]

    #if table_name in db.table_names():
    #    db.drop_table(table_name)

    tbl = db.create_table(table_name, data=records)
    tbl.create_fts_index("headline", replace=True)

    print(f"\nLanceDB table '{table_name}' created at {args.output}")
    print(f"  Rows    : {tbl.count_rows():,}")
    print(f"  Vector  : dim={len(vectors[0])}  (ANN ready)")
    print(f"  Full-text index: 'headline'  (BM25 ready)")
    print("\nExample hybrid query:")
    print('  tbl.search("fed rate hike", query_type="hybrid").limit(5).to_list()')
