from __future__ import annotations

import argparse

import duckdb
import torch
from duckdb.sqltypes import VARCHAR
from transformers import AutoTokenizer

from src.finetune_headline_encoder import HeadlineClassifier
from src.populate_lancedb import encode_all

LANCE_PATH = "lance/headlines.lance"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the Lance extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL lance")
    con.execute("LOAD lance;")
    return con


def register_embed_udf(
    con: duckdb.DuckDBPyConnection,
    model: HeadlineClassifier,
    tokenizer,
    max_length: int,
) -> None:
    """
    Register embed(text) as a scalar UDF on this connection.

    The UDF runs the full encoder pipeline (tokenise → forward → mean-pool)
    and returns a FLOAT[dim] vector, matching the schema written by
    populate_lancedb.py.  Must be called once per connection before any query
    that references embed().
    """
    device  = next(model.parameters()).device
    dim     = model.encoder.config.hidden_size

    def embed(text: str) -> list[float]:
        return encode_all([text], model, tokenizer, max_length, device, 1)[0]

    con.create_function("embed", embed, [VARCHAR], f"FLOAT[{dim}]")



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over headlines")
    parser.add_argument("query", help="Natural-language query to embed and search")
    parser.add_argument("--checkpoint", default="models/headline_classifier.pt")
    parser.add_argument("--lance-path", default=LANCE_PATH)
    args = parser.parse_args()

    import dvc.api
    max_length = dvc.api.params_show()["train"]["max_length"]

    device    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    model     = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.encoder.config._name_or_path)

    con = get_connection()
    register_embed_udf(con, model, tokenizer, max_length)

    print(con.execute(args.query).pl())
