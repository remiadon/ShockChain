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

Storage
-------
  Writes directly to the S3 bucket configured in .dvc/config.
  Credentials and endpoint live in ~/.aws/config under the profile declared
  in the DVC remote config.  Override with --output for local testing:
      python -m src.populate_lancedb --output lance/
"""

from __future__ import annotations

import argparse
import numpy as np
import boto3
import dvc.api
import lancedb
import polars as pl
import torch
import duckdb
from duckdb.sqltypes import VARCHAR
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.finetune_headline_encoder import HeadlineClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--headlines", default="data/news_headlines.parquet")
parser.add_argument("--checkpoint", default="models/headline_classifier.pt")


# ---------------------------------------------------------------------------
# S3 storage options
# ---------------------------------------------------------------------------

def _s3_storage_options() -> dict:
    """
    Use a boto3 Session to resolve the AWS profile declared in params.yaml
    (including endpoint_url, which object_store won't pick up on its own)
    and return a storage_options dict for lancedb.connect().

    get_frozen_credentials() is the documented way to materialise credentials
    from any provider in the chain (env vars, profile, IAM role, etc.).
    The endpoint_url is read via a throw-away S3 client so botocore handles
    all the profile-config lookup without relying on private APIs.
    """
    profile = dvc.api.params_show().get("populate_lancedb", {}).get("aws_profile", "shockchain")
    session = boto3.Session(profile_name=profile)
    creds   = session.get_credentials().get_frozen_credentials()
    client  = session.client("s3")

    return { # FIXME
        "aws_access_key_id":     creds.access_key,
        "aws_secret_access_key": creds.secret_key,
        "aws_region":            session.region_name,
        "aws_endpoint":          client.meta.endpoint_url,
    }


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
    args   = parser.parse_args()
    params = dvc.api.params_show()
    max_length        = params["train"]["max_length"]
    encode_batch_size = int(params["populate_lancedb"]["encode_batch_size"])
    table_name        = params["populate_lancedb"]["table_name"]
    db_uri            = params["populate_lancedb"]["db_uri"]

    # ---- write LanceDB (S3 or local) ----
    storage_options = _s3_storage_options() if db_uri.startswith("s3://") else {}
    db  = lancedb.connect(db_uri, storage_options=storage_options)

    # ---- load model ----
    # torch.save(model, ...) in finetune_headline_encoder.py serialises the full
    # HeadlineClassifier object (encoder weights + head weights + class_names).
    # torch.load therefore restores all of that in one call — no from_pretrained().
    #
    # The tokenizer is NOT part of the model object and must be loaded separately.
    # model.encoder.config._name_or_path records the HuggingFace model ID used
    # during training, so we can reconstruct the exact same tokenizer without
    # hard-coding a name here.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    model  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.encoder.config._name_or_path)
    print(f"Loaded checkpoint: {args.checkpoint}  (base: {model.encoder.config._name_or_path})")

    # ---- load headlines ----
    df    = pl.read_parquet(args.headlines).drop_nulls("title").sample(n=1_000_000, seed=42).sort('date')  # shuffle then sort to preserve temporal continuity while avoiding any ordering bias in the original dataset
    texts = df["title"].to_list()
    dates = df["date"].cast(pl.Utf8).to_list()
    print(f"Encoding {len(texts):,} headlines …")

    # ---- encode ----
    vectors = encode_all(texts, model, tokenizer, max_length, device, encode_batch_size)


    records = [
        {"id": i, "date": dates[i], "headline": texts[i], "vector": vectors[i]}
        for i in range(len(texts))
    ]

    if table_name in db.table_names():
        db.drop_table(table_name)

    tbl = db.create_table(table_name, data=records)
    tbl.create_fts_index("headline", replace=True)

    print(f"\nLanceDB table '{table_name}' written to {db_uri}")
    print(f"  Rows    : {tbl.count_rows():,}")
    print(f"  Vector  : dim={len(vectors[0])}  (ANN ready)")
    print(f"  Full-text index: 'headline'  (BM25 ready)")

