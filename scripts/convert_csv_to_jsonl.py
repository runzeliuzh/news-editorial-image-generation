#!/usr/bin/env python3
"""
Convert MediaEval NewsImages test CSV to JSONL.

This version accepts the raw test CSV directly. It first normalizes the input
text to UTF-8 using a small encoding fallback routine, then reads the CSV and
exports JSONL.

Input CSV columns:
  article_id, article_url, article_title, image_id

Output JSONL fields:
  article_id, image_id, article_url, article_title
"""

import argparse
import csv
import io
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional


REQUIRED_COLUMNS = {"article_id", "article_url", "article_title", "image_id"}


def convert_csv_to_utf8(src_path: str, dst_path: Optional[str] = None) -> Tuple[str, str]:
    src = Path(src_path)
    raw = src.read_bytes()

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    text = None
    chosen = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            chosen = enc
            break
        except UnicodeDecodeError:
            continue

    # latin-1 can decode any byte sequence, so text is always set after the loop.
    assert text is not None and chosen is not None

    if dst_path is not None:
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="")
        print(f"[DONE] Converted {src} -> {dst}")
        print(f"[INFO] Source decode used: {chosen}")

    return text, chosen


def default_utf8_copy_path(input_path: str) -> str:
    src = Path(input_path)
    return str(src.with_name(f"{src.stem}_utf8{src.suffix}"))


def read_test_csv(path: str, write_utf8_copy: bool = True) -> List[Dict[str, str]]:
    utf8_copy_path = default_utf8_copy_path(path) if write_utf8_copy else None
    text, chosen = convert_csv_to_utf8(path, utf8_copy_path)
    print(f"[INFO] Reading normalized CSV text using source decode: {chosen}")

    rows = []
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")

    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing)}. "
            f"Found columns: {reader.fieldnames}"
        )

    for i, row in enumerate(reader, start=2):
        article_id = str(row.get("article_id", "")).strip()
        image_id = str(row.get("image_id", "")).strip()
        article_url = str(row.get("article_url", "")).strip()
        article_title = str(row.get("article_title", "")).strip()

        if not article_id or not article_title:
            print(f"[WARN] Skip row {i}: empty article_id or article_title")
            continue

        rows.append(
            {
                "article_id": article_id,
                "image_id": image_id,
                "article_url": article_url,
                "article_title": article_title,
            }
        )

    return rows


def write_jsonl(rows: List[Dict[str, str]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw or UTF-8 test CSV path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample N rows. Use 0 to export all rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --sample > 0.")
    parser.add_argument("--first", type=int, default=0, help="Take the first N rows instead of random sampling. Use 0 to disable.")
    parser.add_argument(
        "--no-write-utf8-copy",
        action="store_true",
        help="Do not write a normalized *_utf8.csv copy next to the input file.",
    )

    args = parser.parse_args()

    rows = read_test_csv(args.input, write_utf8_copy=not args.no_write_utf8_copy)
    print(f"Loaded valid rows: {len(rows)}")

    if args.first > 0:
        rows = rows[: args.first]
        print(f"Selected first rows: {len(rows)}")
    elif args.sample > 0:
        random.seed(args.seed)
        rows = random.sample(rows, min(args.sample, len(rows)))
        print(f"Random sampled rows: {len(rows)} with seed={args.seed}")
    else:
        print("Exporting all rows")

    write_jsonl(rows, args.output)
    print(f"Wrote JSONL: {args.output}")

    print("Preview:")
    for row in rows[:5]:
        print(f'  {row["article_id"]}: {row["article_title"][:100]}')


if __name__ == "__main__":
    main()