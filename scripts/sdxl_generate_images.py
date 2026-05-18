#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDXL image generation script.

Purpose:
  Generate exactly one 460x260 PNG recommendation per article for one run.
  It can read prompts directly from:
    1) exported rows: row["sdxl_prompts"][...]["sdxl_prompt"]
    2) raw teacher/student rows: row["teacher_target"]["sdxl_prompt"] or row["student_target"]["sdxl_prompt"]

It normally uses the SDXL prompt already present in the input.
If a raw teacher/student row lacks sdxl_prompt but has generation_plan or visual_plan,
it synthesizes a fallback prompt from that row's own visual plan.

MediaEval output folder:
  <submission_root>/<group_name>_<approach_name>/
    <article_id>_<group_name>_<approach_name>.png

It also saves experiment/debug files:
  <output_dir>/candidates/
  <output_dir>/submission_460x260/
  <output_dir>/candidates_manifest.jsonl
  <output_dir>/failed_articles.jsonl
  <output_dir>/generation_config.json
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


DEFAULT_NEGATIVE = (
    "photorealistic face, realistic face, detailed human face, realistic portrait, "
    "face close-up, headshot, exact facial likeness, AI-generated fake portrait, "
    "uncanny human face, photorealistic fake news photo, readable text, legible text, "
    "dense text, small text, letters, words, chart labels, document text, exact logos, "
    "watermark, clutter, unrelated objects"
)


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
    return " ".join(str(x or "").split())


def as_clean_list(x: Any) -> List[str]:
    """Return a clean list of strings from a list or scalar value."""
    if isinstance(x, list):
        return [clean_text(v) for v in x if clean_text(v)]
    if clean_text(x):
        return [clean_text(x)]
    return []


def get_visual_plan(target: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the visual/generation plan from a raw target.

    Normal rows should have generation_plan. Some imperfect planner rows may
    instead have visual_plan. This fallback lets those rows still produce one
    SDXL prompt without manually editing the JSONL.
    """
    generation_plan = target.get("generation_plan")
    if isinstance(generation_plan, dict):
        return generation_plan

    visual_plan = target.get("visual_plan")
    if isinstance(visual_plan, dict):
        return visual_plan

    return {}


def synthesize_prompt_from_visual_plan(row: Dict[str, Any], target: Dict[str, Any]) -> str:
    """
    Build a fallback SDXL prompt from the row's own title and visual plan.

    This is used only when a raw teacher/student target has no top-level
    sdxl_prompt. It does not affect normal rows that already have prompts.
    """
    title = clean_text(row.get("article_title", ""))
    plan = get_visual_plan(target)

    required = as_clean_list(plan.get("required_visual_cues"))
    optional = as_clean_list(plan.get("optional_visual_cues"))

    visual_strategy = clean_text(plan.get("visual_strategy", "")).replace("_", " ")
    framing = clean_text(plan.get("framing", ""))
    abstractness = clean_text(plan.get("abstractness", "")).replace("_", " ")

    cues: List[str] = []
    seen: Set[str] = set()
    for item in required + optional:
        key = item.lower()
        if key not in seen:
            cues.append(item)
            seen.add(key)

    parts = [
        "non-photorealistic editorial illustration",
        "flat vector and cut-paper style",
        "clean symbolic news thumbnail",
    ]

    if title:
        parts.append(f"about {title}")
    if cues:
        parts.append(", ".join(cues))
    if visual_strategy:
        parts.append(f"{visual_strategy} visual strategy")
    if abstractness:
        parts.append(f"{abstractness} composition")
    if framing:
        parts.append(framing)

    parts.extend([
        "balanced layout",
        "muted colors with strong contrast",
        "no readable text",
        "no exact logos",
        "no watermark",
        "avoid realistic human faces",
    ])

    return ", ".join([p for p in parts if clean_text(p)])


def synthesize_negative_from_visual_plan(target: Dict[str, Any]) -> str:
    """
    Build a fallback negative prompt from DEFAULT_NEGATIVE plus row-specific
    forbidden cues, if present.
    """
    plan = get_visual_plan(target)
    forbidden = as_clean_list(plan.get("forbidden_or_misleading_cues"))

    neg = DEFAULT_NEGATIVE
    if forbidden:
        neg = neg + ", " + ", ".join(forbidden)
    return neg


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
    """Used only to warn if two submission PNGs are exactly identical."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dimensions(width: int, height: int) -> None:
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"SDXL width/height must be divisible by 8. Got {width}x{height}.")


def load_sdxl(model_name: str, device: str, scheduler_name: str, local_files_only: bool):
    from diffusers import StableDiffusionXLPipeline
    from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": local_files_only,
    }
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)
    except Exception as e:
        if kwargs.get("variant") == "fp16":
            print(f"fp16 variant load failed; retrying without variant='fp16'. Reason: {e}")
            kwargs.pop("variant", None)
            pipe = StableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)
        else:
            raise

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.vae.enable_slicing()
    except Exception:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.to(device)
    return pipe


@torch.inference_mode()
def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    device: str,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return out.images[0]


def resize_for_submission(img: Image.Image, target_w: int = 460, target_h: int = 260) -> Image.Image:
    """Center-crop only if aspect ratio differs, then resize to 460x260 without stretching."""
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
    """Return prompt dicts with variant, sdxl_prompt, negative_prompt."""
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
            neg = clean_text(p.get("negative_prompt", "")) or DEFAULT_NEGATIVE
            if pos:
                out.append({"variant": variant, "sdxl_prompt": pos, "negative_prompt": neg})
        return out

    target = extract_raw_target(row, target_source)
    if not target:
        return out

    pos = clean_text(target.get("sdxl_prompt", ""))
    neg = clean_text(target.get("negative_prompt", "")) or DEFAULT_NEGATIVE

    # Minimal fallback for imperfect raw planner rows:
    # if sdxl_prompt is missing but generation_plan/visual_plan exists,
    # synthesize one from the row's own title and plan.
    if not pos:
        plan = get_visual_plan(target)
        if plan:
            pos = synthesize_prompt_from_visual_plan(row, target)
            neg = clean_text(target.get("negative_prompt", "")) or synthesize_negative_from_visual_plan(target)

    if not pos:
        return out

    variant = target_source if target_source in {"teacher", "student"} else "raw"
    if variants is not None and variant not in variants:
        return out

    out.append({"variant": variant, "sdxl_prompt": pos, "negative_prompt": neg})
    return out


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
        "negative_prompt": neg,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with exported prompts or raw teacher/student target prompts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-mode", default="auto", choices=["auto", "exported", "raw_target"])
    parser.add_argument("--target-source", default="auto", choices=["auto", "teacher", "student"])
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=520, help="Generate at 520 for MediaEval 460x260")
    parser.add_argument("--width", type=int, default=920, help="Generate at 920 for MediaEval 460x260")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--scheduler", default="dpm", choices=["dpm", "euler", "euler_a"])
    parser.add_argument("--images-per-prompt", type=int, default=1, help="Must remain 1 for MediaEval submission")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="*", default=None, help="Optional. For exported prompts, e.g. --variants chosen")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--group-name", default="NewsWeavers")
    parser.add_argument("--approach-name", default="Qwen3BLoraSDXL")
    parser.add_argument("--submission-root", default="", help="Defaults to output-dir if omitted")
    args = parser.parse_args()

    if args.images_per_prompt != 1:
        raise ValueError("MediaEval submission requires one image per article. Use --images-per-prompt 1.")
    if args.approach_name.strip() == "":
        raise ValueError("--approach-name cannot be empty")
    if args.group_name.strip() == "":
        raise ValueError("--group-name cannot be empty")

    validate_dimensions(args.width, args.height)

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

    print("Loading SDXL...")
    pipe = load_sdxl(args.model, args.device, args.scheduler, args.local_files_only)

    manifest: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    seen_article_ids: Set[str] = set()
    seen_submission_hashes: Dict[str, str] = {}
    generated = skipped_existing = failed = skipped_prompt = 0

    for row_idx, row in enumerate(tqdm(rows, desc="articles")):
        article_id = str(row.get("article_id", "")).strip()
        image_id = str(row.get("image_id", ""))
        title = clean_text(row.get("article_title", ""))

        if not article_id or article_id == "None":
            failed += 1
            record_failure(failed_records, row_idx, row, reason="missing_article_id")
            continue

        if article_id in seen_article_ids:
            # Duplicate IDs would overwrite submission filenames, so this is a data integrity error.
            failed += 1
            record_failure(failed_records, row_idx, row, reason="duplicate_article_id")
            continue
        seen_article_ids.add(article_id)

        prompt_rows = collect_prompts_from_row(
            row=row,
            input_mode=args.input_mode,
            target_source=args.target_source,
            variants=allowed_variants,
        )

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
        neg = clean_text(p.get("negative_prompt", "")) or DEFAULT_NEGATIVE

        article_slug = slugify(article_id)
        sample_idx = 0
        seed = stable_seed(args.seed, article_id, title, variant, sample_idx)
        candidate_id = f"{article_slug}__{variant}__s{sample_idx}"
        image_path = cand_dir / f"{candidate_id}.png"
        debug_submission_path = sub_dir / f"{candidate_id}.png"
        mediaeval_submission_path = mediaeval_dir / f"{article_slug}_{args.group_name}_{args.approach_name}.png"

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
                img = generate_image(
                    pipe=pipe,
                    prompt=pos,
                    negative_prompt=neg,
                    seed=seed,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    device=args.device,
                )
                img.save(image_path, format="PNG")
                generated += 1
                print(f"Saved candidate: {image_path}", flush=True)

            with Image.open(image_path) as img_for_sub:
                submission_img = resize_for_submission(img_for_sub)
                submission_img.save(debug_submission_path, format="PNG")
                submission_img.save(mediaeval_submission_path, format="PNG")

            # Warn if two submitted PNG files are exactly identical. This does not stop nohup runs.
            img_hash = sha256_file(mediaeval_submission_path)
            if img_hash in seen_submission_hashes:
                prev_article = seen_submission_hashes[img_hash]
                warnings.warn(
                    f"Duplicate image content detected: article_id={article_id} duplicates article_id={prev_article}"
                )
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
                "negative_prompt": neg,
                "image_path": str(image_path),
                "debug_submission_path": str(debug_submission_path),
                "mediaeval_submission_path": str(mediaeval_submission_path),
                "sdxl_model": args.model,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "scheduler": args.scheduler,
            })

        except RuntimeError as e:
            failed += 1
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            err = str(e)
            warnings.warn(f"FAILED article_id={article_id}, candidate={candidate_id}: {err}")
            record_failure(
                failed_records, row_idx, row, reason="generation_runtime_error", error=err,
                candidate_id=candidate_id, variant=variant, sample_idx=sample_idx,
                seed=seed, pos=pos, neg=neg,
            )

        except Exception as e:
            failed += 1
            err = str(e)
            warnings.warn(f"FAILED article_id={article_id}, candidate={candidate_id}: {err}")
            record_failure(
                failed_records, row_idx, row, reason="generation_error", error=err,
                candidate_id=candidate_id, variant=variant, sample_idx=sample_idx,
                seed=seed, pos=pos, neg=neg,
            )

    write_jsonl(out_dir / "candidates_manifest.jsonl", manifest)
    write_jsonl(out_dir / "failed_articles.jsonl", failed_records)
    with open(out_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    submission_png_count = len(list(mediaeval_dir.glob("*.png")))
    expected = len(rows)
    if submission_png_count != expected:
        warnings.warn(
            f"Incomplete MediaEval folder: found {submission_png_count} PNGs, "
            f"but input has {expected} rows. Check failed_articles.jsonl."
        )

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
        "duplicate_image_hash_warnings_possible": True,
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


if __name__ == "__main__":
    main()
