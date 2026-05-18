#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split an SFT JSONL file into train/validation JSONL files reproducibly.

Expected input format:
Each line is one JSON object. Extra fields are preserved.

Example:
python split_sft_train_val.py \
  --input qwen3b_plan_sft.jsonl \
  --train-out qwen3b_plan_sft.train.jsonl \
  --val-out qwen3b_plan_sft.val500.jsonl \
  --val-size 500 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="Input SFT JSONL file.")
    ap.add_argument("--train-out", required=True, help="Output train JSONL file.")
    ap.add_argument("--val-out", required=True, help="Output validation JSONL file.")
    ap.add_argument("--val-size", type=int, default=500, help="Number of validation rows.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffle.")
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )

    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno} in {path}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} in {path} is not a JSON object.")

            rows.append(obj)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_sft_format(rows: List[Dict[str, Any]]) -> None:
    """Light sanity check. Does not modify data."""
    bad = 0

    for r in rows:
        messages = r.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            bad += 1
            continue

        roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
        if roles != ["system", "user", "assistant"]:
            bad += 1

    if bad:
        raise ValueError(
            f"Found {bad} rows that do not look like messages=[system,user,assistant]. "
            "Please check the input SFT JSONL."
        )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if args.val_size <= 0:
        raise ValueError("--val-size must be positive.")

    for out_path in [train_out, val_out]:
        if out_path.exists() and not args.allow_overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_path}\n"
                f"Use --allow-overwrite if you intentionally want to overwrite it."
            )

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in input file: {input_path}")

    check_sft_format(rows)

    if args.val_size >= len(rows):
        raise ValueError(
            f"--val-size={args.val_size} must be smaller than total rows={len(rows)}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size :]

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    print(json.dumps(
        {
            "input": str(input_path),
            "train_out": str(train_out),
            "val_out": str(val_out),
            "seed": args.seed,
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()