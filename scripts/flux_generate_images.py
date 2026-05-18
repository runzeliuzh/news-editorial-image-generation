#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean FLUX image generation for MediaEval NewsImages-style outputs.

This version is intentionally simple:
- uses the original Qwen `sdxl_prompt` as the FLUX prompt;
- optionally adds ONE fixed style phrase;
- does NOT use human-face detection;
- does NOT rewrite, replace, or repair semantic prompt content;
- does NOT use SDXL negative_prompt for FLUX generation.

We used FLUX.1-schnell settings on 16GB VRAM:
  --sequential-cpu-offload --steps 4 --guidance-scale 0.0
  --max-sequence-length 256 --width 736 --height 416
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
from PIL import Image
from tqdm import tqdm


STYLE_PROFILES = {
    "none": "",
    "editorial": (
        "clean text-free editorial illustration, non-photorealistic news image, "
        "complete simplified people only, no realistic faces, no portraits, no close-up faces, "
        "no readable letters, no logos"
    ),
    "symbolic": (
        "symbolic text-free editorial illustration, non-photorealistic news graphic, "
        "complete simplified people only, no realistic faces, no portraits, no close-up faces, "
        "no readable letters, no logos"
    ),
    "flat_vector": (
        "flat vector text-free editorial illustration, clean news graphic, "
        "complete simplified people only, no realistic faces, no portraits, no close-up faces, "
        "no readable letters, no logos"
    ),
    "silhouette": (
        "faceless silhouette text-free editorial illustration, simplified figures, "
        "complete simplified people only, no realistic faces, no portraits, no close-up faces, "
        "no readable letters, no logos"
    ),
}



# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    return " ".join(str(x or "").replace("\n", " ").replace("\t", " ").split())


def slugify(text: Any, max_len: int = 120) -> str:
    text = str(text if text is not None else "")
    out = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s[:max_len] if s else "item"


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------

def row_has_exported_prompts(row: Dict[str, Any]) -> bool:
    prompts = row.get("sdxl_prompts")
    return isinstance(prompts, list) and len(prompts) > 0


def extract_raw_target(row: Dict[str, Any], target_source: str) -> Optional[Dict[str, Any]]:
    if target_source == "teacher":
        target = row.get("teacher_target")
    elif target_source == "student":
        target = row.get("student_target")
    else:
        target = row.get("student_target")
        if not isinstance(target, dict) or not clean_text(target.get("sdxl_prompt", "")):
            target = row.get("teacher_target")
    return target if isinstance(target, dict) else None


def collect_prompts_from_row(
    row: Dict[str, Any],
    input_mode: str,
    target_source: str,
    variants: Optional[Set[str]],
) -> List[Dict[str, str]]:
    """
    Return usable prompts from one row.

    For exported inference output, use:
      --input-mode exported --variants chosen
    """
    out: List[Dict[str, str]] = []

    mode = input_mode
    if mode == "auto":
        mode = "exported" if row_has_exported_prompts(row) else "raw_target"

    if mode == "exported":
        prompts = row.get("sdxl_prompts", [])
        if not isinstance(prompts, list):
            return out
        for p in prompts:
            if not isinstance(p, dict):
                continue
            variant = clean_text(p.get("variant", "chosen")) or "chosen"
            if variants is not None and variant not in variants:
                continue
            pos = clean_text(p.get("sdxl_prompt", ""))
            neg = clean_text(p.get("negative_prompt", ""))
            if pos:
                out.append({"variant": variant, "sdxl_prompt": pos, "negative_prompt": neg})
        return out

    target = extract_raw_target(row, target_source)
    if not target:
        return out

    pos = clean_text(target.get("sdxl_prompt", ""))
    neg = clean_text(target.get("negative_prompt", ""))
    if not pos:
        return out

    variant = target_source if target_source in {"teacher", "student"} else "raw"
    if variants is not None and variant not in variants:
        return out

    out.append({"variant": variant, "sdxl_prompt": pos, "negative_prompt": neg})
    return out


# ---------------------------------------------------------------------
# Prompt style only
# ---------------------------------------------------------------------

def build_flux_prompt(pos: str, style_profile: str, style_position: str) -> str:
    """
    Build the final FLUX prompt.

    This function does not rewrite the original prompt.
    It only adds one fixed style phrase, or returns the raw prompt if
    --style-profile none.

    Prefixing helps avoid the style phrase being truncated by CLIP when the
    original prompt is long. The semantic content of the original prompt is not
    rewritten or modified.
    """
    pos = clean_text(pos)
    style = clean_text(STYLE_PROFILES.get(style_profile, ""))

    if not style:
        return pos

    if style.lower() in pos.lower():
        return pos

    if style_position == "suffix":
        return f"{pos}, {style}"
    return f"{style}, {pos}"


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def resize_for_submission(img: Image.Image, target_w: int = 460, target_h: int = 260) -> Image.Image:
    """
    Center-crop to target aspect ratio if needed, then resize to 460x260.
    This avoids stretching.
    """
    img = img.convert("RGB")
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    if abs(src_ratio - tgt_ratio) > 0.01:
        if src_ratio > tgt_ratio:
            new_w = int(src_h * tgt_ratio)
            left = (src_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, src_h))
        else:
            new_h = int(src_w / tgt_ratio)
            top = (src_h - new_h) // 2
            img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def validate_dimensions(width: int, height: int) -> None:
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"FLUX width/height should be divisible by 16. Got {width}x{height}.")


# ---------------------------------------------------------------------
# FLUX loading and generation
# ---------------------------------------------------------------------

def load_flux(
    model_name: str,
    device: str,
    dtype_name: str,
    cpu_offload: bool,
    sequential_cpu_offload: bool,
    local_files_only: bool,
):
    from diffusers import FluxPipeline

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype_map[dtype_name],
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)

    # Best-effort memory helpers.
    for fn_name in ("enable_vae_slicing", "enable_attention_slicing"):
        try:
            getattr(pipe, fn_name)()
        except Exception:
            pass

    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    if device.startswith("cuda") and sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif device.startswith("cuda") and cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


@torch.inference_mode()
def generate_flux_image(
    pipe,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)

    out = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        max_sequence_length=max_sequence_length,
    )
    return out.images[0]


# ---------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------

def record_failure(
    failed_records: List[Dict[str, Any]],
    row_idx: int,
    row: Dict[str, Any],
    reason: str,
    error: str = "",
    candidate_id: str = "",
    variant: str = "",
    sample_idx: int = 0,
    seed: Optional[int] = None,
    pos: str = "",
    neg: str = "",
) -> None:
    failed_records.append({
        "article_index": row_idx,
        "article_id": str(row.get("article_id", "")),
        "image_id": str(row.get("image_id", "")),
        "article_title": clean_text(row.get("article_title", "")),
        "reason": reason,
        "error": error,
        "candidate_id": candidate_id,
        "variant": variant,
        "sample_index": sample_idx,
        "seed": seed,
        "positive_prompt": pos,
        "negative_prompt_ignored_by_flux": neg,
    })


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="JSONL with exported prompts or raw target prompts")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--input-mode", default="auto", choices=["auto", "exported", "raw_target"])
    parser.add_argument("--target-source", default="auto", choices=["auto", "teacher", "student"])
    parser.add_argument("--variants", nargs="*", default=None)

    parser.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--cpu-offload", action="store_true", help="Balanced memory-saving offload")
    parser.add_argument("--sequential-cpu-offload", action="store_true", help="Maximum memory-saving offload; slower but best for 16GB VRAM")

    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="For FLUX.1-schnell, use 0.0")
    parser.add_argument("--max-sequence-length", type=int, default=256, help="For FLUX.1-schnell, use <=256")

    parser.add_argument("--style-profile", default="symbolic",
                        choices=["none", "editorial", "symbolic", "flat_vector", "silhouette"],
                        help="Only style control. It adds one fixed style phrase and does not rewrite the prompt.")
    parser.add_argument("--style-position", default="prefix", choices=["prefix", "suffix"],
                        help="Where to add the fixed style phrase. Prefix is less likely to be truncated.")

    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--group-name", default="NewsWeavers")
    parser.add_argument("--approach-name", default="Qwen3BLoraFLUX")
    parser.add_argument("--submission-root", default="", help="Defaults to output-dir if omitted")
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not raise an error if fewer final PNGs are produced than input rows")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.images_per_prompt != 1:
        raise ValueError("MediaEval submission requires one image per article. Use --images-per-prompt 1.")

    if not args.approach_name.strip():
        raise ValueError("--approach-name cannot be empty")

    if not args.group_name.strip():
        raise ValueError("--group-name cannot be empty")

    validate_dimensions(args.width, args.height)

    if args.cpu_offload and args.sequential_cpu_offload:
        raise ValueError("Use either --cpu-offload or --sequential-cpu-offload, not both.")

    if "schnell" in args.model.lower():
        if args.max_sequence_length > 256:
            raise ValueError("FLUX.1-schnell requires --max-sequence-length <= 256.")
        if abs(args.guidance_scale) > 1e-8:
            warnings.warn("FLUX.1-schnell is normally used with --guidance-scale 0.0.")
        if args.steps > 4:
            warnings.warn("FLUX.1-schnell is normally used with 1-4 steps.")

    if args.device.startswith("cuda") and not (args.cpu_offload or args.sequential_cpu_offload):
        warnings.warn("CUDA selected without CPU offload. FLUX may OOM on 16GB VRAM.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    out_dir = Path(args.output_dir)
    cand_dir = out_dir / "candidates"
    sub_dir = out_dir / "submission_460x260"
    submission_root = Path(args.submission_root) if args.submission_root else out_dir
    mediaeval_dir = submission_root / f"{args.group_name}_{args.approach_name}"

    ensure_dir(cand_dir)
    ensure_dir(sub_dir)
    ensure_dir(mediaeval_dir)

    rows = list(read_jsonl(args.input))
    if args.max_records > 0:
        rows = rows[: args.max_records]

    allowed_variants = set(args.variants) if args.variants else None

    print(f"Loaded rows: {len(rows)}")
    print(f"Input mode: {args.input_mode}")
    print(f"Target source: {args.target_source}")
    print(f"Requested variants: {sorted(allowed_variants) if allowed_variants else 'all relevant variants'}")
    print(f"Candidates: {cand_dir}")
    print(f"Debug 460x260 folder: {sub_dir}")
    print(f"MediaEval run folder: {mediaeval_dir}")
    print(f"Loading FLUX: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"CPU offload: {args.cpu_offload}")
    print(f"Sequential CPU offload: {args.sequential_cpu_offload}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps: {args.steps}, guidance_scale: {args.guidance_scale}, max_sequence_length: {args.max_sequence_length}")
    print(f"Style profile: {args.style_profile}")
    print(f"Style position: {args.style_position}")

    pipe = load_flux(
        model_name=args.model,
        device=args.device,
        dtype_name=args.dtype,
        cpu_offload=args.cpu_offload,
        sequential_cpu_offload=args.sequential_cpu_offload,
        local_files_only=args.local_files_only,
    )

    manifest: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    seen_article_ids: Set[str] = set()
    seen_submission_hashes: Dict[str, str] = {}

    generated = 0
    skipped_existing = 0
    failed = 0
    skipped_prompt = 0

    for row_idx, row in enumerate(tqdm(rows, desc="articles")):
        article_id = str(row.get("article_id", "")).strip()
        image_id = str(row.get("image_id", ""))
        title = clean_text(row.get("article_title", ""))

        if not article_id or article_id == "None":
            failed += 1
            record_failure(failed_records, row_idx, row, reason="missing_article_id")
            continue

        if article_id in seen_article_ids:
            failed += 1
            record_failure(failed_records, row_idx, row, reason="duplicate_article_id")
            continue
        seen_article_ids.add(article_id)

        prompt_rows = collect_prompts_from_row(row, args.input_mode, args.target_source, allowed_variants)
        if len(prompt_rows) != 1:
            failed += 1
            skipped_prompt += 1
            record_failure(
                failed_records,
                row_idx,
                row,
                reason="not_exactly_one_usable_prompt",
                error=f"usable_prompts={len(prompt_rows)}",
            )
            continue

        p = prompt_rows[0]
        variant = str(p.get("variant", "raw"))
        pos = clean_text(p.get("sdxl_prompt", ""))
        neg = clean_text(p.get("negative_prompt", ""))

        article_slug = slugify(article_id)
        sample_idx = 0
        seed = stable_seed(args.seed, article_id, title, variant, sample_idx)

        candidate_id = f"{article_slug}__{variant}__s{sample_idx}"
        image_path = cand_dir / f"{candidate_id}.png"
        debug_submission_path = sub_dir / f"{candidate_id}.png"
        mediaeval_submission_path = mediaeval_dir / f"{article_slug}_{args.group_name}_{args.approach_name}.png"

        flux_prompt_used = build_flux_prompt(
            pos=pos,
            style_profile=args.style_profile,
            style_position=args.style_position,
        )

        try:
            need_generate = True

            if args.skip_existing and image_path.exists():
                try:
                    with Image.open(image_path) as test_img:
                        test_img.verify()
                    skipped_existing += 1
                    need_generate = False
                except Exception:
                    print(f"Existing candidate is unreadable; regenerating: {image_path}", flush=True)
                    need_generate = True

            if need_generate:
                img = generate_flux_image(
                    pipe=pipe,
                    prompt=flux_prompt_used,
                    seed=seed,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                )
                img.save(image_path, format="PNG")
                generated += 1
                print(f"Saved candidate: {image_path}", flush=True)

            with Image.open(image_path) as img_for_sub:
                submission_img = resize_for_submission(img_for_sub)
                submission_img.save(debug_submission_path, format="PNG")
                submission_img.save(mediaeval_submission_path, format="PNG")

            img_hash = sha256_file(mediaeval_submission_path)
            if img_hash in seen_submission_hashes:
                prev_article = seen_submission_hashes[img_hash]
                warnings.warn(f"Duplicate image content detected: article_id={article_id} duplicates article_id={prev_article}")
            else:
                seen_submission_hashes[img_hash] = article_id

            manifest.append({
                "candidate_id": candidate_id,
                "article_index": row_idx,
                "article_id": article_id,
                "image_id": image_id,
                "article_title": title,
                "prompt_variant": variant,
                "sample_index": sample_idx,
                "seed": seed,
                "positive_prompt": pos,
                "negative_prompt_ignored_by_flux": neg,
                "flux_prompt_used": flux_prompt_used,
                "style_profile": args.style_profile,
                "style_position": args.style_position,
                "image_path": str(image_path),
                "debug_submission_path": str(debug_submission_path),
                "mediaeval_submission_path": str(mediaeval_submission_path),
                "flux_model": args.model,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "max_sequence_length": args.max_sequence_length,
                "cpu_offload": args.cpu_offload,
                "sequential_cpu_offload": args.sequential_cpu_offload,
            })

        except RuntimeError as e:
            failed += 1
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            err = str(e)
            warnings.warn(f"FAILED article_id={article_id}, candidate={candidate_id}: {err}")
            record_failure(
                failed_records,
                row_idx,
                row,
                reason="generation_runtime_error",
                error=err,
                candidate_id=candidate_id,
                variant=variant,
                sample_idx=sample_idx,
                seed=seed,
                pos=pos,
                neg=neg,
            )

        except Exception as e:
            failed += 1
            err = str(e)
            warnings.warn(f"FAILED article_id={article_id}, candidate={candidate_id}: {err}")
            record_failure(
                failed_records,
                row_idx,
                row,
                reason="generation_error",
                error=err,
                candidate_id=candidate_id,
                variant=variant,
                sample_idx=sample_idx,
                seed=seed,
                pos=pos,
                neg=neg,
            )

    write_jsonl(out_dir / "candidates_manifest.jsonl", manifest)
    write_jsonl(out_dir / "failed_articles.jsonl", failed_records)

    with open(out_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    submission_png_count = len(list(mediaeval_dir.glob("*.png")))
    expected = len(rows)
    incomplete = submission_png_count != expected

    if incomplete:
        msg = (
            f"Incomplete MediaEval folder: found {submission_png_count} PNGs, "
            f"but input has {expected} rows. Check failed_articles.jsonl."
        )
        warnings.warn(msg)

    summary = {
        "group_name": args.group_name,
        "approach_name": args.approach_name,
        "mediaeval_dir": str(mediaeval_dir),
        "input_rows": len(rows),
        "manifest_rows": len(manifest),
        "submission_png_count": submission_png_count,
        "failed_records": len(failed_records),
        "generated_new_candidates": generated,
        "skipped_existing_candidates": skipped_existing,
        "skipped_prompt_records": skipped_prompt,
        "is_complete": not incomplete,
        "prompt_method_note": (
            "FLUX prompt = original Qwen sdxl_prompt plus one optional fixed style phrase. "
            "No human-face detection, no semantic prompt rewrite, and negative_prompt is ignored."
        ),
    }

    with open(out_dir / "submission_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Generated new candidates: {generated}")
    print(f"Skipped existing candidates: {skipped_existing}")
    print(f"Failed/skipped records: {failed}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Submission PNG count: {submission_png_count}")
    print(f"Saved candidates: {cand_dir}")
    print(f"Saved debug 460x260 PNGs: {sub_dir}")
    print(f"Saved MediaEval run folder: {mediaeval_dir}")
    print(f"Saved manifest: {out_dir / 'candidates_manifest.jsonl'}")
    print(f"Saved failed records: {out_dir / 'failed_articles.jsonl'}")
    print(f"Saved summary: {out_dir / 'submission_summary.json'}")

    if failed_records:
        print("\nWARNING: Some articles failed or had no usable prompt. Handle these before final ZIP submission.")

    if incomplete and not args.allow_incomplete:
        raise RuntimeError("Incomplete MediaEval output. See failed_articles.jsonl and submission_summary.json.")


if __name__ == "__main__":
    main()
