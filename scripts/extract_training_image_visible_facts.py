#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-VL image-visible fact extraction for MediaEval NewsImages.

VLM used:
    Qwen/Qwen2.5-VL-3B-Instruct

Purpose:
    Extract conservative image-only visible facts from training images.
    The model does not see the article title/content.
    The output is used as soft visual evidence for pseudo visual-intent construction.

Outputs:
    train_image_visible_facts.jsonl
    train_image_visible_facts.csv
    audit_summary.json
    config.json

Design decisions:
    - Open objects/actions are preserved as open vocabulary.
    - Coarse taxonomy labels are normalized.
    - Invalid taxonomy outputs are mapped to uncertain/mixed.
    - visible_text is a coarse label; visible_text_content stores actual readable text.
    - visual_evidence_reliability is a heuristic reliability score, not a calibrated probability.
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
SCHEMA_VERSION = "qwen_visible_facts_simple_v2"
PROMPT_VERSION = "visible_fact_prompt_v2"

STYLE_TYPES = [
    "photograph",
    "stock_photograph",
    "non_photorealistic_illustration",
    "editorial_cartoon",
    "infographic",
    "map",
    "chart",
    "screenshot",
    "document",
    "collage",
    "poster_or_logo",
    "mixed_style",
    "other",
    "uncertain",
]

MAIN_SUBJECT_TYPES = [
    "person",
    "group",
    "object",
    "institution",
    "landscape_or_environment",
    "map",
    "chart_or_data",
    "document_or_screenshot",
    "abstract_or_symbolic",
    "mixed",
    "other",
    "uncertain",
]

VISUAL_ROLES = [
    "portrait",
    "event_scene",
    "institutional_anchor",
    "object_metaphor",
    "map_based_representation",
    "data_visualization",
    "document_evidence",
    "generic_stock_concept",
    "symbolic_editorial_scene",
    "mixed_role",
    "open_role",
    "uncertain",
]

COMPOSITION_TYPES = [
    "single_dominant_subject",
    "central_person",
    "group_scene",
    "object_centered_closeup",
    "map_centered",
    "left_right_opposition",
    "collage",
    "text_heavy",
    "abstract_layout",
    "mixed_composition",
    "other",
    "uncertain",
]

COARSE_TONES = [
    "neutral",
    "serious",
    "urgent",
    "positive",
    "tense",
    "celebratory",
    "uncertain",
]

VISIBLE_TEXT_LABELS = [
    "none",
    "small_amount",
    "large_amount",
    "uncertain",
]

YES_NO_UNCERTAIN = [
    "yes",
    "no",
    "uncertain",
]

FACE_VISIBILITY_LABELS = [
    "none",
    "small_faces",
    "clear_face",
    "large_closeup_face",
    "uncertain",
]

PERSON_COUNT_LABELS = [
    "0",
    "1",
    "2",
    "3-5",
    "more_than_5",
    "uncertain",
]


def list_images(image_dir: str):
    image_dir = Path(image_dir)
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def image_id_from_path(path: Path):
    return path.stem


def safe_image_metadata(path: Path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size

        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6) if height else None,
        }
    except Exception as e:
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "metadata_error": str(e),
        }


def extract_json_from_text(text: str):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove markdown fences if Qwen adds them.
    text2 = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text2 = re.sub(r"```$", "", text2).strip()

    try:
        return json.loads(text2)
    except Exception:
        pass

    # Last resort: extract the first JSON-looking object.
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def canonicalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_label(value, allowed, default="uncertain", multi_value_policy="uncertain"):
    if value is None:
        return default

    if isinstance(value, list):
        if len(value) == 0:
            return default
        if len(value) == 1:
            value = value[0]
        else:
            return multi_value_policy

    raw = str(value).strip()

    # Exact match first. This preserves labels such as "3-5".
    if raw in allowed:
        return raw

    # If the model copies or joins labels with "|", treat it conservatively.
    if "|" in raw:
        parts = [canonicalize_text(x) for x in raw.split("|") if x.strip()]
        valid_parts = [p for p in parts if p in allowed]
        if len(valid_parts) == 1:
            return valid_parts[0]
        return multi_value_policy

    val = canonicalize_text(raw)

    aliases = {
        "text_heavy": "text_heavy",
        "text-heavy": "text_heavy",

        "photo": "photograph",
        "real_photo": "photograph",
        "real_photograph": "photograph",
        "news_photograph": "photograph",

        "stock_photo": "stock_photograph",
        "stock_image": "stock_photograph",

        "illustration": "non_photorealistic_illustration",
        "non_photorealistic": "non_photorealistic_illustration",
        "non_photorealistic_image": "non_photorealistic_illustration",

        "cartoon": "editorial_cartoon",

        "poster": "poster_or_logo",
        "logo": "poster_or_logo",
        "poster_logo": "poster_or_logo",
        "brand_logo": "poster_or_logo",

        "data_visualisation": "data_visualization",
        "data_visualization": "data_visualization",

        "building": "institution",
        "government_building": "institution",
        "institutional_building": "institution",
        "church": "institution",
        "school": "institution",
        "courthouse": "institution",
        "court": "institution",

        "document_screenshot": "document_or_screenshot",
        "screenshot_or_document": "document_or_screenshot",
        "page": "document_or_screenshot",

        "symbolic_scene": "symbolic_editorial_scene",
        "object_metaphorical": "object_metaphor",
        "map_based": "map_based_representation",
    }

    val = aliases.get(val, val)

    if val in allowed:
        return val

    return default


def normalize_label_list(values, allowed, default="uncertain", max_items=3):
    if values is None:
        return [default]

    if isinstance(values, str):
        raw = values.strip()
        if "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
        elif "," in raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            parts = [raw]
    elif isinstance(values, list):
        parts = values
    else:
        return [default]

    cleaned = []
    for p in parts:
        label = normalize_label(
            p,
            allowed,
            default=None,
            multi_value_policy=None,
        )
        if label and label in allowed and label not in cleaned:
            cleaned.append(label)

    if len(cleaned) > max_items:
        if "mixed_role" in cleaned:
            return ["mixed_role"]
        return [default]

    if not cleaned:
        return [default]

    return cleaned


def normalize_bool(value, default=None):
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    val = str(value).strip().lower()
    if val in ["true", "yes", "1"]:
        return True
    if val in ["false", "no", "0"]:
        return False

    return default


def normalize_open_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if ";" in value:
            return [x.strip() for x in value.split(";") if x.strip()]
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]

    return []


def normalize_visible_text_label(value):
    """
    visible_text should be one of:
        none / small_amount / large_amount / uncertain

    visible_text_content stores actual readable text separately.
    This function also handles older-style outputs where the model accidentally
    put actual text into visible_text.
    """
    if value is None:
        return "uncertain", ""

    raw = str(value).strip()

    if raw in VISIBLE_TEXT_LABELS:
        return raw, ""

    if not raw:
        return "none", ""

    if raw.lower() in ["yes", "no", "true", "false"]:
        return "uncertain", ""

    # Treat non-label content as actual visible text.
    if len(raw) <= 30:
        return "small_amount", raw
    return "large_amount", raw


def normalize_qwen_facts(parsed):
    if not isinstance(parsed, dict):
        return None

    out = dict(parsed)

    out["open_visible_description"] = str(
        out.get("open_visible_description", "")
    ).strip()

    out["open_objects"] = normalize_open_list(out.get("open_objects"))
    out["open_actions"] = normalize_open_list(out.get("open_actions"))

    out["open_scene"] = str(out.get("open_scene", "uncertain")).strip() or "uncertain"

    out["human_presence"] = normalize_bool(out.get("human_presence"), default=None)

    out["person_count_estimate"] = normalize_label(
        out.get("person_count_estimate"),
        PERSON_COUNT_LABELS,
        default="uncertain",
    )

    out["face_visibility"] = normalize_label(
        out.get("face_visibility"),
        FACE_VISIBILITY_LABELS,
        default="uncertain",
    )

    out["main_subject_type"] = normalize_label(
        out.get("main_subject_type"),
        MAIN_SUBJECT_TYPES,
        default="uncertain",
    )

    out["visual_role_candidates"] = normalize_label_list(
        out.get("visual_role_candidates"),
        VISUAL_ROLES,
        default="uncertain",
        max_items=3,
    )

    out["composition_type"] = normalize_label(
        out.get("composition_type"),
        COMPOSITION_TYPES,
        default="uncertain",
    )

    out["style_type_vlm"] = normalize_label(
        out.get("style_type_vlm"),
        STYLE_TYPES,
        default="uncertain",
    )

    out["coarse_tone"] = normalize_label(
        out.get("coarse_tone"),
        COARSE_TONES,
        default="uncertain",
    )

    visible_text_label, fallback_visible_content = normalize_visible_text_label(
        out.get("visible_text")
    )
    out["visible_text"] = visible_text_label

    explicit_visible_content = str(out.get("visible_text_content", "")).strip()
    if explicit_visible_content:
        out["visible_text_content"] = explicit_visible_content
    else:
        out["visible_text_content"] = fallback_visible_content

    out["logo_or_watermark_visible"] = normalize_label(
        out.get("logo_or_watermark_visible"),
        YES_NO_UNCERTAIN,
        default="uncertain",
    )

    out["requires_title_context"] = True

    out["uncertainty_notes"] = str(out.get("uncertainty_notes", "")).strip()

    return out


def build_qwen_prompt():
    prompt = {
        "task": (
            "You are auditing a news image, but you do NOT know the article title "
            "or article content. Describe ONLY visible facts in the image."
        ),
        "do_not_infer": [
            "news topic",
            "specific event",
            "identity of people",
            "political meaning",
            "legal meaning",
            "cause or consequence",
            "what the article is about",
        ],
        "rules": [
            "Return valid JSON only. No markdown.",
            "For choose_one fields, output exactly one label from the allowed list.",
            "For visual_role_candidates, output 1 to 3 labels from the allowed list.",
            "Do not copy the allowed label list as the answer.",
            "Do not join labels with the pipe symbol.",
            "Use uncertain when unclear.",
            "Open objects and actions can use open vocabulary.",
            "visible_text must be one of: none, small_amount, large_amount, uncertain.",
            "If readable text is visible, put the actual text in visible_text_content; otherwise use an empty string.",
        ],
        "allowed_labels": {
            "main_subject_type": MAIN_SUBJECT_TYPES,
            "visual_role_candidates": VISUAL_ROLES,
            "composition_type": COMPOSITION_TYPES,
            "style_type_vlm": STYLE_TYPES,
            "coarse_tone": COARSE_TONES,
            "visible_text": VISIBLE_TEXT_LABELS,
            "logo_or_watermark_visible": YES_NO_UNCERTAIN,
            "face_visibility": FACE_VISIBILITY_LABELS,
            "person_count_estimate": PERSON_COUNT_LABELS,
        },
        "json_schema": {
            "open_visible_description": "one concise sentence describing only visible content",
            "open_objects": ["open-vocabulary visible object names"],
            "open_actions": ["coarse visible actions if meaningful and visible"],
            "open_scene": "visible scene or environment, or uncertain",

            "human_presence": True,
            "person_count_estimate": "choose_one",
            "face_visibility": "choose_one",

            "main_subject_type": "choose_one",
            "visual_role_candidates": ["choose_from_allowed_labels"],
            "composition_type": "choose_one",
            "style_type_vlm": "choose_one",
            "coarse_tone": "choose_one",

            "visible_text": "choose_one",
            "visible_text_content": "actual readable text if visible, otherwise empty string",
            "logo_or_watermark_visible": "choose_one",

            "requires_title_context": True,
            "uncertainty_notes": "briefly state what cannot be determined from the image alone",
        },
    }

    return json.dumps(prompt, ensure_ascii=False, indent=2)


def load_qwen(model_name: str, max_pixels: int):
    print(f"Loading Qwen VLM: {model_name}")
    print(f"Processor max_pixels: {max_pixels}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
    )

    model.eval()
    return model, processor


@torch.no_grad()
def qwen_visible_facts(image_path: Path, model, processor, max_new_tokens: int):
    prompt = build_qwen_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed_raw = extract_json_from_text(raw_response)
    parsed_ok = parsed_raw is not None
    parsed_normalized = normalize_qwen_facts(parsed_raw) if parsed_ok else None

    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_ok": parsed_ok,
        "parsed_visible_facts_raw": parsed_raw,
        "parsed_visible_facts": parsed_normalized,
    }


def compute_visual_evidence_reliability(parsed_facts):
    """
    Heuristic reliability score for how useful the extracted image-visible facts are.
    This is NOT a calibrated model confidence.
    """
    if not parsed_facts:
        return 0.0, "ignore"

    score = 1.0

    main_type = parsed_facts.get("main_subject_type", "uncertain")
    if main_type in ["uncertain", "other"]:
        score -= 0.25
    elif main_type == "mixed":
        score -= 0.10

    roles = parsed_facts.get("visual_role_candidates", [])
    if not roles or "uncertain" in roles:
        score -= 0.25
    if "mixed_role" in roles or "open_role" in roles:
        score -= 0.10

    comp = parsed_facts.get("composition_type", "uncertain")
    if comp in ["uncertain", "other"]:
        score -= 0.15
    elif comp == "mixed_composition":
        score -= 0.10

    style_type = parsed_facts.get("style_type_vlm", "uncertain")
    if style_type == "uncertain":
        score -= 0.10

    desc = parsed_facts.get("open_visible_description", "")
    if not desc or len(desc) < 10:
        score -= 0.20

    score = max(0.0, min(1.0, score))

    if score >= 0.70:
        usage = "strong"
    elif score >= 0.40:
        usage = "soft"
    else:
        usage = "ignore"

    return float(score), usage


def build_csv_rows(records):
    rows = []

    for r in records:
        if r.get("error"):
            rows.append({
                "image_id": r.get("image_id"),
                "image_path": r.get("image_path"),
                "error": r.get("error"),
            })
            continue

        qwen = r.get("qwen_visible_facts", {}) or {}
        parsed = qwen.get("parsed_visible_facts") or {}
        meta = r.get("image_metadata", {}) or {}

        rows.append({
            "image_id": r.get("image_id"),
            "image_path": r.get("image_path"),

            "width": meta.get("width"),
            "height": meta.get("height"),
            "aspect_ratio": meta.get("aspect_ratio"),
            "file_size_bytes": meta.get("file_size_bytes"),

            "qwen_parsed_ok": qwen.get("parsed_ok"),

            "open_visible_description": parsed.get("open_visible_description"),
            "open_objects": "; ".join(parsed.get("open_objects", [])) if isinstance(parsed.get("open_objects"), list) else "",
            "open_actions": "; ".join(parsed.get("open_actions", [])) if isinstance(parsed.get("open_actions"), list) else "",
            "open_scene": parsed.get("open_scene"),

            "human_presence": parsed.get("human_presence"),
            "person_count_estimate": parsed.get("person_count_estimate"),
            "face_visibility": parsed.get("face_visibility"),

            "main_subject_type": parsed.get("main_subject_type"),
            "visual_role_candidates": "; ".join(parsed.get("visual_role_candidates", [])) if isinstance(parsed.get("visual_role_candidates"), list) else "",
            "composition_type": parsed.get("composition_type"),
            "style_type_vlm": parsed.get("style_type_vlm"),
            "coarse_tone": parsed.get("coarse_tone"),

            "visible_text": parsed.get("visible_text"),
            "visible_text_content": parsed.get("visible_text_content"),
            "logo_or_watermark_visible": parsed.get("logo_or_watermark_visible"),

            "uncertainty_notes": parsed.get("uncertainty_notes"),

            "visual_evidence_reliability": r.get("visual_evidence_reliability"),
            "image_anchor_usage": r.get("image_anchor_usage"),
        })

    return rows


def summarize(records):
    def parsed_facts(r):
        return (r.get("qwen_visible_facts", {}) or {}).get("parsed_visible_facts") or {}

    valid_records = [r for r in records if not r.get("error")]

    summary = {
        "num_records": len(records),
        "num_valid_records": len(valid_records),
        "num_errors": len(records) - len(valid_records),
        "qwen_model": QWEN_MODEL_NAME,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,

        "qwen_parsed_ok": Counter([
            str((r.get("qwen_visible_facts", {}) or {}).get("parsed_ok"))
            for r in valid_records
        ]),

        "main_subject_type_distribution": Counter([
            parsed_facts(r).get("main_subject_type", "missing")
            for r in valid_records
        ]),

        "visual_role_candidate_distribution": Counter([
            role
            for r in valid_records
            for role in parsed_facts(r).get("visual_role_candidates", ["missing"])
        ]),

        "composition_type_distribution": Counter([
            parsed_facts(r).get("composition_type", "missing")
            for r in valid_records
        ]),

        "style_type_vlm_distribution": Counter([
            parsed_facts(r).get("style_type_vlm", "missing")
            for r in valid_records
        ]),

        "visible_text_distribution": Counter([
            parsed_facts(r).get("visible_text", "missing")
            for r in valid_records
        ]),

        "face_visibility_distribution": Counter([
            parsed_facts(r).get("face_visibility", "missing")
            for r in valid_records
        ]),

        "image_anchor_usage_distribution": Counter([
            r.get("image_anchor_usage", "missing")
            for r in valid_records
        ]),
    }

    return {
        k: dict(v) if isinstance(v, Counter) else v
        for k, v in summary.items()
    }


def write_outputs(jsonl_path, csv_path, summary_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    rows = build_csv_rows(records)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = summarize(records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def check_resume_compatibility(output_dir: Path, args):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as f:
        old = json.load(f)

    problems = []

    if old.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version mismatch: {old.get('schema_version')} != {SCHEMA_VERSION}")

    if old.get("prompt_version") != PROMPT_VERSION:
        problems.append(f"prompt_version mismatch: {old.get('prompt_version')} != {PROMPT_VERSION}")

    if old.get("qwen_model") != QWEN_MODEL_NAME:
        problems.append(f"qwen_model mismatch: {old.get('qwen_model')} != {QWEN_MODEL_NAME}")

    if old.get("max_pixels") != args.max_pixels:
        problems.append(f"max_pixels mismatch: {old.get('max_pixels')} != {args.max_pixels}")

    if problems:
        msg = "\n".join(problems)
        raise RuntimeError(
            "Resume is unsafe because the existing output directory was created "
            "with a different configuration:\n"
            f"{msg}\n\n"
            "Use a new output_dir or delete the old output_dir before rerunning."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/news_images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen_image_visible_facts",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Qwen-VL processes image size in units related to 28x28 patches.
    # 768 * 28 * 28 is a stable default for consumer GPUs.
    parser.add_argument("--max_pixels", type=int, default=768 * 28 * 28)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "train_image_visible_facts.jsonl"
    csv_path = output_dir / "train_image_visible_facts.csv"
    summary_path = output_dir / "audit_summary.json"
    config_path = output_dir / "config.json"

    if args.resume:
        check_resume_compatibility(output_dir, args)

    config = {
        "created_at": datetime.now().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "qwen_model": QWEN_MODEL_NAME,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "style_types": STYLE_TYPES,
        "main_subject_types": MAIN_SUBJECT_TYPES,
        "visual_roles": VISUAL_ROLES,
        "composition_types": COMPOSITION_TYPES,
        "coarse_tones": COARSE_TONES,
        "visible_text_labels": VISIBLE_TEXT_LABELS,
        "face_visibility_labels": FACE_VISIBILITY_LABELS,
        "person_count_labels": PERSON_COUNT_LABELS,
        "note": (
            "Only Qwen/Qwen2.5-VL-3B-Instruct is used. "
            "No CLIP, OCR, face detector, GroundingDINO, SAM, or fallback VLM is used. "
            "The output contains image-only visible facts and heuristic reliability scores. "
            "All taxonomy labels are soft evidence, not hard generation constraints."
        ),
    }

    # Always write config at start. In non-resume mode, this overwrites old config.
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    images = list_images(image_dir)
    if args.limit:
        images = images[:args.limit]

    print(f"Found images: {len(images)}")
    print(f"Image dir: {image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Qwen VLM: {QWEN_MODEL_NAME}")
    print(f"Schema version: {SCHEMA_VERSION}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"max_pixels: {args.max_pixels}")

    done = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
        print(f"Resume mode: {len(done)} images already processed.")

    model, processor = load_qwen(QWEN_MODEL_NAME, args.max_pixels)

    mode = "a" if args.resume else "w"

    with open(jsonl_path, mode, encoding="utf-8") as fout:
        for path in tqdm(images, desc="Extracting Qwen visible facts"):
            image_id = image_id_from_path(path)

            if image_id in done:
                continue

            metadata = safe_image_metadata(path)

            record = {
                "image_id": image_id,
                "image_path": str(path),
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "image_metadata": metadata,
                "models": {
                    "vlm": QWEN_MODEL_NAME,
                },
            }

            if metadata.get("width") is None:
                record["error"] = metadata.get("metadata_error", "failed_to_open_image")
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            try:
                record["qwen_visible_facts"] = qwen_visible_facts(
                    image_path=path,
                    model=model,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                record["qwen_visible_facts"] = {
                    "prompt_version": PROMPT_VERSION,
                    "prompt": build_qwen_prompt(),
                    "raw_response": "",
                    "parsed_ok": False,
                    "parsed_visible_facts_raw": None,
                    "parsed_visible_facts": None,
                    "error": str(e),
                }

            parsed_facts = record["qwen_visible_facts"].get("parsed_visible_facts")
            reliability, usage = compute_visual_evidence_reliability(parsed_facts)

            record["visual_evidence_reliability"] = reliability
            record["image_anchor_usage"] = usage
            record["requires_title_context"] = True
            record["method_note"] = (
                "Image-only visible fact extraction. "
                "Do not treat this as article semantics. "
                "Use with article title/text later to construct text-conditioned visual intent. "
                "All taxonomy labels are soft evidence, not hard generation constraints. "
                "visual_evidence_reliability is heuristic, not calibrated confidence."
            )

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    write_outputs(jsonl_path, csv_path, summary_path)

    print("\nDone.")
    print(f"Saved JSONL:   {jsonl_path}")
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved config:  {config_path}")


if __name__ == "__main__":
    main()