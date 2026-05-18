#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Qwen2.5-3B-Instruct LoRA/QLoRA on title-only visual-planner SFT data.

Expected JSONL format:
Each row must contain:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{...five-key JSON...}"}
  ],
  ... extra metadata allowed ...
}

This script:
- trains only on assistant tokens, not system/user prompt tokens
- supports optional validation JSONL
- supports QLoRA 4-bit training
- supports checkpoint resume
- saves final LoRA adapter in --out-dir
"""

from __future__ import annotations
import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from datasets import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data / model
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--eval-file", default="", help="Optional validation JSONL file.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Training
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--num-train-epochs", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Resume
    ap.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Path to checkpoint dir, or 'true' to auto-resume from latest checkpoint.",
    )

    return ap.parse_args()


def validate_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = example.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Each row must contain messages=[system,user,assistant].")

    expected_roles = ["system", "user", "assistant"]

    for msg, expected_role in zip(messages, expected_roles):
        if not isinstance(msg, dict):
            raise ValueError("Each message must be a dict.")

        role = msg.get("role")
        content = msg.get("content")

        if role != expected_role:
            raise ValueError(f"Expected role={expected_role}, got role={role}.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty content for role={expected_role}.")

    return messages


def build_tokenized_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    """
    Assistant-only supervised loss.

    full_text:
      system + user + assistant

    prompt_text:
      system + user + assistant reply prefix (add_generation_prompt=True)

    labels:
      -100 for all prompt_text tokens (system, user, and reply prefix)
      token ids for assistant answer tokens only
    """
    messages = validate_messages(example)

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt["input_ids"]), len(input_ids))

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]

    supervised_token_count = sum(1 for x in labels if x != -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "supervised_token_count": supervised_token_count,
    }


class DataCollatorForMaskedCausalLM:
    def __init__(self, tokenizer: AutoTokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    p = Path(out_dir)
    if not p.exists():
        return None

    checkpoints = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-")[-1])
                checkpoints.append((step, str(child)))
            except ValueError:
                pass

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def load_and_tokenize_jsonl(
    path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    split_name: str,
):
    raw = load_dataset("json", data_files=path, split="train")

    tokenized = raw.map(
        lambda ex: build_tokenized_example(ex, tokenizer, max_seq_length),
        remove_columns=raw.column_names,
        desc=f"Tokenizing {split_name} data with assistant-only labels",
    )

    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda ex: ex["supervised_token_count"] > 0,
        desc=f"Filtering {split_name} rows with no assistant labels",
    )
    after = len(tokenized)

    if after < before:
        print(
            f"WARNING: filtered {before - after} / {before} {split_name} rows "
            f"because the assistant answer was fully truncated.",
            flush=True,
        )

    tokenized = tokenized.remove_columns(["supervised_token_count"])

    return tokenized


def print_supervised_token_sanity(dataset, name: str, max_check: int = 50) -> None:
    n = min(max_check, len(dataset))
    if n == 0:
        print(f"WARNING: {name} dataset is empty.", flush=True)
        return

    total = 0
    min_count = None
    max_count = None

    for i in range(n):
        labels = dataset[i]["labels"]
        count = sum(1 for x in labels if x != -100)
        total += count
        min_count = count if min_count is None else min(min_count, count)
        max_count = count if max_count is None else max(max_count, count)

    print(
        f"{name} sanity check over first {n} rows: "
        f"avg_supervised_tokens={total / n:.1f}, "
        f"min={min_count}, max={max_count}",
        flush=True,
    )


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    fp16_enabled = (not args.bf16) and torch.cuda.is_available()

    kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=fp16_enabled,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        disable_tqdm=True,
    )

    if has_eval:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
        )
    else:
        kwargs.update(eval_strategy="no")

    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        # Older transformers use evaluation_strategy instead of eval_strategy.
        if "eval_strategy" in kwargs:
            v = kwargs.pop("eval_strategy")
            kwargs["evaluation_strategy"] = v
        return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quant_config,
        torch_dtype=None if quant_config is not None else dtype,
    )

    model.config.use_cache = False

    if args.load_in_4bit:
        print("Preparing model for k-bit QLoRA training...", flush=True)
        model = prepare_model_for_kbit_training(model)
    else:
        # Important when using PEFT + gradient checkpointing without k-bit preparation.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("Adding LoRA adapters...", flush=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading and tokenizing train dataset...", flush=True)
    train_dataset = load_and_tokenize_jsonl(
        path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        split_name="train",
    )

    eval_dataset = None
    if args.eval_file:
        print("Loading and tokenizing eval dataset...", flush=True)
        eval_dataset = load_and_tokenize_jsonl(
            path=args.eval_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            split_name="eval",
        )

    print(f"Train examples: {len(train_dataset)}", flush=True)
    print(f"Eval examples: {len(eval_dataset) if eval_dataset is not None else 0}", flush=True)

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after tokenization/filtering.")

    print_supervised_token_sanity(train_dataset, "train")
    if eval_dataset is not None:
        print_supervised_token_sanity(eval_dataset, "eval")

    training_args = build_training_arguments(args, has_eval=eval_dataset is not None)
    collator = DataCollatorForMaskedCausalLM(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume_arg = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "true":
            resume_arg = find_latest_checkpoint(args.out_dir)
        else:
            resume_arg = args.resume_from_checkpoint

        if resume_arg:
            print(f"Resuming from checkpoint: {resume_arg}", flush=True)
        else:
            print("No checkpoint found; starting from scratch.", flush=True)

    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    print("Saving final LoRA adapter...", flush=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    meta = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "eval_file": args.eval_file or None,
        "out_dir": args.out_dir,
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "loss_masking": "assistant_only",
        "load_in_4bit": bool(args.load_in_4bit),
        "bf16": bool(args.bf16),
        "fp16": bool((not args.bf16) and torch.cuda.is_available()),
        "seed": args.seed,
    }

    Path(args.out_dir, "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Done. Final LoRA adapter saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()