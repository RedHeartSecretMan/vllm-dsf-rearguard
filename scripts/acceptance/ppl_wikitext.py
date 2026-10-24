"""Prepare the wikitext-2 test split for ``scripts.acceptance.ppl_score`` as a token JSONL file.

The published PPL run joined the detokenized non-empty documents with a blank
line, tokenized the text once with the model's tokenizer and scored
non-overlapping 2048-token windows. This script reproduces that preparation and
also writes the word and byte counts needed for word and byte perplexity::

    python3 -m scripts.acceptance.ppl_wikitext MODEL_DIR --output runtime/wikitext-2-test.jsonl
    python3 -m scripts.acceptance.ppl_score runtime/wikitext-2-test.jsonl --request-form echo \\
        --top-k 1 --output runtime/ppl-wikitext-2-test.json

It needs ``datasets``, ``transformers`` and ``lm_eval`` (for the detokenizer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize_documents(rows: list[str], detokenize) -> tuple[str, int, int]:
    """Return the scored text, the raw word count and the raw byte count.

    Word and byte counts are taken from the raw non-empty rows, matching the
    published run, while the model sees the detokenized text.
    """
    documents = [row for row in rows if row]
    text = "\n\n".join(detokenize(row) for row in documents)
    raw = "\n".join(documents)
    return text, len(re.findall(r"\s+", raw)) + 1, len(raw.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a fresh evidence path")
    from datasets import load_dataset
    from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
    from transformers import AutoTokenizer

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text, words, byte_count = normalize_documents(
        list(dataset["text"]), lambda row: wikitext_detokenizer({"page": row})
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, use_fast=True
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    record = {
        "tokens": tokens,
        "dataset": "wikitext-2-raw-v1:test",
        "word_count": words,
        "byte_count": byte_count,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k != "tokens"}
                   | {"token_count": len(tokens)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
