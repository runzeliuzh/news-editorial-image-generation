#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT data preparation pipeline for plan-based Qwen3B LoRA SFT.

Goal
----
Create high-quality SFT examples for:
  input = title-only planning prompt.
  output = five-key JSON:
           title_understanding, generation_plan, sdxl_prompt, negative_prompt, ranking_caption

Method
------
1. Join MediaEval training title rows with VLM-parsed paired-image visible facts by image_id.
2. Ask a stronger teacher, e.g. Qwen2.5-7B-Instruct, to create a five-key silver target.
3. Repair/normalize/fuzzy-map small schema mistakes.
4. Validate schema and basic semantic quality.
5. Optionally sample rows for manual audit.
6. Build final SFT JSONL as the 'ideal' example answer.

Important
---------
- The teacher sees title + VLM parse.
- The student SFT input sees title only.
- The final assistant target contains only the five keys expected by the inference script.


Example
-------
# 1) create silver labels
python prepare_sft_data.py make-silver \
  --articles train_titles_all.jsonl \
  --vlm train_image_visible_facts.jsonl \
  --out qwen7b_teacher_silver.jsonl \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --load-in-4bit \
  --local-files-only \
  --max-new-tokens 1800 \
  --retries 2

# 2) validate/filter
python prepare_sft_data.py validate-filter \
  --input qwen7b_teacher_silver.jsonl \
  --valid-out qwen7b_teacher_silver.valid.jsonl \
  --bad-out qwen7b_teacher_silver.bad.jsonl

# 3) sample 100 for manual audit
python prepare_sft_data.py sample-audit \
  --input qwen7b_teacher_silver.valid.jsonl \
  --out manual_audit_100.jsonl \
  --n 100 \
  --seed 42

# 4) build SFT JSONL
python prepare_sft_data.py make-sft \
  --input qwen7b_teacher_silver.valid.jsonl \
  --out qwen3b_plan_sft_data.jsonl
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_NEGATIVE = (
    "photorealistic face, realistic face, detailed human face, realistic portrait, "
    "face close-up, headshot, exact facial likeness, AI-generated fake portrait, "
    "uncanny human face, photorealistic fake news photo, readable text, legible text, "
    "dense text, small text, letters, words, chart labels, document text, exact logos, "
    "watermark, clutter, unrelated objects, distorted anatomy, graphic gore, "
    "misleading literal scene"
)


FINAL_TOP_KEYS = [
    "title_understanding",
    "generation_plan",
    "sdxl_prompt",
    "negative_prompt",
    "ranking_caption",
]

TITLE_KEYS = [
    "headline_core",
    "named_entities",
    "entity_roles",
    "main_event_or_issue",
    "implied_context",
    "relation",
    "visual_risks",
    "what_should_not_be_emphasized",
]

PLAN_KEYS = [
    "image_role",
    "required_visual_cues",
    "optional_visual_cues",
    "forbidden_or_misleading_cues",
    "visual_strategy",
    "framing",
    "abstractness",
]

VALID_VISUAL_STRATEGIES = [
    "direct_depiction",
    "symbolic_editorial",
    "location_anchor",
    "person_or_group_scene",
    "object_still_life",
    "document_or_data_visual",
    "map_or_geography",
    "event_scene",
    "conceptual_composite",
    "other",
]

VALID_ABSTRACTNESS = [
    "literal",
    "semi_symbolic",
    "symbolic",
    "other",
]

LIST_TITLE_KEYS = {
    "named_entities",
    "entity_roles",
    "visual_risks",
    "what_should_not_be_emphasized",
}

LIST_PLAN_KEYS = {
    "required_visual_cues",
    "optional_visual_cues",
    "forbidden_or_misleading_cues",
}


# ---------------------------------------------------------------------
# Exact student prompt copied/aligned with current inference script
# ---------------------------------------------------------------------


def student_system_prompt() -> str:
    return (
        "You are a title-grounded news visual planner for SDXL news/editorial image generation. "
        "You will receive only one news article title. Do not use article bodies or external retrieval. "
        "Use general world knowledge only to recognize entities and context; do not invent unsupported facts. "
        "Return exactly one valid JSON object and no markdown. "
        "The JSON must use exactly these top-level keys: "
        "title_understanding, generation_plan, sdxl_prompt, negative_prompt, ranking_caption. "
        "The field title_understanding explains the title's main event, entities, issue, and uncertainty. "
        "The field generation_plan is the visual plan derived from title_understanding: image role, required cues, optional cues, forbidden or misleading cues, strategy, framing, and abstractness. "
        "Do not use alternative top-level keys such as visualPlan, visual_plan, prompt, SDXL Prompt, "
        "SDXLPrompt, negativePrompt, Negative Prompt, caption, headlineCore. "
        "The field sdxl_prompt means a Stable Diffusion XL text-to-image prompt derived from title_understanding and generation_plan. "
        "It must include the plan's required visual cues, reflect its visual_strategy/framing/abstractness, and avoid the plan's forbidden_or_misleading_cues. "
        "Choose among non-photorealistic or clearly synthetic styles according to the title and generation_plan; the image should look suitable for a news illustration, not like a realistic event photograph. "
        "The field negative_prompt must list things to avoid in Stable Diffusion XL generation, including title-specific misleading details. "
        "The field negative_prompt should be derived from title_understanding and generation_plan, especially the forbidden_or_misleading_cues. "
        "The field ranking_caption is a short plain-language caption describing the intended generated image for later evaluation; it is not an SDXL prompt. "
        "Write sdxl_prompt and negative_prompt as natural-language comma-separated phrases with spaces, not snake_case tags or variable-like tokens. "
        "Write sdxl_prompt as a concise visual prompt, not a long paragraph, and put the most important title-specific cues early. "
        "Write negative_prompt as concise comma-separated avoid terms. "
        "Always avoid readable text, exact logos, watermarks, fake documents, chart labels, graphic harm, "
        "misleading literal scenes, and exact realistic public-figure likenesses unless the title clearly requires a generic non-likeness crowd/person scene. "
        "Keep all reasoning inside the JSON fields."
    )

def schema_instruction(title: str) -> str:
    # This is a schema/template, not a completed article-output example.
    return f"""Article title:
{title}

Return exactly one valid JSON object following this schema. Fill every field with title-supported content.

{{
  "title_understanding": {{
    "headline_core": "short interpretation of the title",
    "named_entities": ["entity names if any"],
    "entity_roles": ["brief roles if any"],
    "main_event_or_issue": "main issue or event described by the title",
    "implied_context": "relevant context inferred only from the title and general knowledge",
    "relation": "relationship among entities/issues if any",
    "visual_risks": ["risks such as wrong likeness, misleading literal scene, readable text"],
    "what_should_not_be_emphasized": ["things that should not dominate the image"]
  }},
  "generation_plan": {{
    "image_role": "editorial illustration for a news thumbnail",
    "required_visual_cues": ["2-5 title-supported visual cues"],
    "optional_visual_cues": ["0-4 optional supporting cues"],
    "forbidden_or_misleading_cues": ["things that should not appear"],
    "visual_strategy": "direct_depiction | symbolic_editorial | location_anchor | person_or_group_scene | object_still_life | document_or_data_visual | map_or_geography | event_scene | conceptual_composite | other",
    "framing": "brief composition/framing description",
    "abstractness": "literal | semi_symbolic | symbolic | other"
  }},
  "sdxl_prompt": "title-specific visual subject and cues from generation_plan, title-appropriate clearly synthetic/non-photorealistic editorial style, clear composition, no readable text, no exact logos, no watermark",
  "negative_prompt": "title- and plan-specific misleading detail, forbidden cue, readable text, exact logos, watermark, fake documents, chart labels, exact realistic likeness",
  "ranking_caption": "one concise caption describing what the generated image should show"
}}

Rules:
- Output JSON only.
- Use snake_case keys exactly as shown.
- sdxl_prompt and negative_prompt must be non-empty strings.
- generation_plan must be derived from title_understanding, not from unsupported assumptions.
- required_visual_cues must be supported by the title.
- sdxl_prompt must operationalize generation_plan: include the required_visual_cues, reflect visual_strategy/framing/abstractness, and avoid the forbidden_or_misleading_cues.
- negative_prompt must operationalize title_understanding.visual_risks and generation_plan.forbidden_or_misleading_cues.
- negative_prompt should include title-specific misleading details, not only generic terms.
- Keep sdxl_prompt concise and visual, not a long paragraph; put important title-specific cues early.
- Choose a clearly synthetic or non-photorealistic editorial/news style appropriate to the title.
- Write sdxl_prompt and negative_prompt as natural-language comma-separated phrases with spaces; do not use snake_case, tag-style tokens, or underscores inside field values.
- Keep negative_prompt concise, comma-separated, and do not repeat phrases.
- ranking_caption should be one short human-readable sentence describing the intended image meaning; do not include SDXL style/control terms.
"""

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def clean_text(x: Any) -> str:
    return " ".join(str(x or "").replace("\n", " ").replace("\t", " ").split())


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, dict):
                # Used when old teacher emits named entity objects.
                text = clean_text(v.get("text", ""))
                if text:
                    out.append(text)
            elif clean_text(v):
                out.append(clean_text(v))
        return out
    if isinstance(x, str):
        s = clean_text(x)
        if not s:
            return []
        if "," in s or ";" in s:
            return [clean_text(p) for p in re.split(r"[,;]", s) if clean_text(p)]
        return [s]
    return [clean_text(x)] if clean_text(x) else []


def ensure_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return clean_text(x)
    if isinstance(x, list):
        return clean_text(", ".join(clean_text(v) for v in x if clean_text(v)))
    if isinstance(x, dict):
        return clean_text(json.dumps(x, ensure_ascii=False))
    return clean_text(str(x))


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON in {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            yield row


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_article_rows(path: str, title_col: str, article_id_col: str, image_col: str) -> Iterable[Dict[str, str]]:
    if path.lower().endswith(".jsonl"):
        for idx, row in enumerate(read_jsonl(path), 1):
            title = clean_text(row.get(title_col, ""))
            if not title:
                continue
            yield {
                "article_id": clean_text(row.get(article_id_col, idx)),
                "image_id": clean_text(row.get(image_col, "")),
                "article_title": title,
            }
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if title_col not in fields:
                raise ValueError(f"Missing title column {title_col}. Available columns: {fields}")
            for idx, row in enumerate(reader, 1):
                title = clean_text(row.get(title_col, ""))
                if not title:
                    continue
                yield {
                    "article_id": clean_text(row.get(article_id_col, idx)),
                    "image_id": clean_text(row.get(image_col, "")),
                    "article_title": title,
                }


# ---------------------------------------------------------------------
# VLM parse loading
# ---------------------------------------------------------------------

def extract_vlm_parse(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("qwen_visible_facts")
    if isinstance(q, dict):
        for k in ("parsed_visible_facts", "visible_facts", "parsed"):
            if isinstance(q.get(k), dict):
                return q[k]

    for k in ("parsed_visible_facts", "visible_facts", "parse"):
        if isinstance(row.get(k), dict):
            return row[k]

    parse_keys = {
        "open_visible_description", "open_objects", "open_actions", "open_scene",
        "human_presence", "person_count_estimate", "face_visibility",
        "main_subject_type", "visual_role_candidates", "composition_type",
        "style_type_vlm", "coarse_tone", "visible_text", "visible_text_content",
        "logo_or_watermark_visible", "requires_title_context", "uncertainty_notes",
        "visual_evidence_reliability", "image_anchor_usage",
    }
    if any(k in row for k in parse_keys):
        return {k: row.get(k) for k in parse_keys if row.get(k) not in (None, "", [], {})}
    return None


def compress_vlm_parse(vlm: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "open_visible_description", "open_objects", "open_actions", "open_scene",
        "human_presence", "person_count_estimate", "face_visibility",
        "main_subject_type", "visual_role_candidates", "composition_type",
        "style_type_vlm", "coarse_tone", "visible_text", "visible_text_content",
        "logo_or_watermark_visible", "requires_title_context", "uncertainty_notes",
        "visual_evidence_reliability", "image_anchor_usage",
    ]
    return {k: vlm.get(k) for k in keep if vlm.get(k) not in (None, "", [], {})}


def load_vlm_by_image_id(path: str) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    total = usable = 0
    for row in read_jsonl(path):
        total += 1
        image_id = clean_text(row.get("image_id", ""))
        if not image_id:
            continue
        parsed = extract_vlm_parse(row)
        if isinstance(parsed, dict) and parsed:
            by_id[image_id] = compress_vlm_parse(parsed)
            usable += 1
    print(f"Loaded VLM rows: total={total}; usable={usable}; keyed_by_image_id={len(by_id)}", flush=True)
    return by_id


# ---------------------------------------------------------------------
# Robust JSON parsing / repair
# ---------------------------------------------------------------------

def strip_markdown_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_candidate(text: str) -> List[str]:
    """
    Return plausible JSON candidates from model output.
    Prefer extracted {...}, but also keep the raw stripped text.
    """
    s = strip_markdown_fence(text)
    candidates: List[str] = []

    def add(x: str) -> None:
        x = (x or "").strip()
        if x and x not in candidates:
            candidates.append(x)

    add(s)

    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        add(s[obj_start:obj_end + 1])

    list_start = s.find("[")
    list_end = s.rfind("]")
    if list_start >= 0 and list_end > list_start:
        add(s[list_start:list_end + 1])

    return candidates


def strip_json_comments(text: str) -> str:
    """
    Remove // and # comments outside quoted strings.
    More careful than line.split("//"), so URLs or text inside strings are preserved.
    """
    text = text or ""
    out = []
    in_str = False
    quote = ""
    escape = False
    i = 0

    while i < len(text):
        ch = text[i]

        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            i += 1
            continue

        if ch in ("'", '"'):
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "#":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def repair_invalid_json_escapes(text: str) -> str:
    """
    JSON only allows escapes like \\n, \\t, \\", \\\\, \\uXXXX.
    Remove backslashes before unsupported escape chars.
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', "", text or "")


def repair_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text or "")


def repair_overescaped_json_quotes(text: str) -> str:
    """
    Repair outputs like:
      {\\\"title_understanding\\\": ...}
    into:
      {"title_understanding": ...}

    Only applies when escaped quotes are frequent.
    """
    if not text:
        return ""
    if text.count('\\"') >= 8:
        return text.replace('\\"', '"')
    return text


def repair_extra_quote_after_json_string(text: str) -> str:
    """
    Repair common model mistake:
      "sdxl_prompt": "some value"",
    into:
      "sdxl_prompt": "some value",

    Conservative: only targets one-line string values right before comma,
    closing brace/bracket, or newline.
    """
    if not text:
        return ""
    return re.sub(r'(:\s*"[^"\n\r]*")"\s*([,\}\]\n\r])', r"\1\2", text)


def repair_unescaped_label_quotes(text: str) -> str:
    """
    Conservative repair for a few common malformed inner-quote patterns.
    Do not try to solve arbitrary broken quotes here.
    """
    if not text:
        return ""

    # Example:
    # "large screen with "LIVE" text" -> "large screen with LIVE text"
    # Keep this conservative to avoid damaging valid JSON.
    text = re.sub(
        r'"([^"\n\r]{1,80})"\s+(sign|label|banner|poster|text)"',
        r'"\1 \2"',
        text,
        flags=re.IGNORECASE,
    )

    # Example from your smoke result:
    # ["company", ""event"] -> ["company", "event"]
    text = re.sub(r'""([^"\n\r]{1,40})"', r'"\1"', text)

    return text


def repair_tabs_or_spaces_in_key_names(text: str) -> str:
    """
    Repair key names with accidental tabs/newlines/leading/trailing spaces:
      "\ttitle_understanding": ...
    Also fixes a common space typo:
      "what_should_not_be emphasized"
    """
    if not text:
        return ""

    def fix_key(m: re.Match) -> str:
        key = m.group(1)
        fixed = re.sub(r"[\t\r\n]+", "", key).strip()
        return f'"{fixed}":'

    text = re.sub(r'"([^"\n\r]{1,100})"\s*:', fix_key, text)
    text = text.replace("what_should_not_be emphasized", "what_should_not_be_emphasized")
    return text


def _unwrap_dict_like(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return None


def _try_json_dict(text: str) -> Optional[Dict[str, Any]]:
    try:
        return _unwrap_dict_like(json.loads(text))
    except Exception:
        pass

    try:
        return _unwrap_dict_like(ast.literal_eval(text))
    except Exception:
        return None


def _add_unique(items: List[str], value: str) -> None:
    value = (value or "").strip()
    if value and value not in items:
        items.append(value)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust parser for Qwen teacher JSON-like output.

    Handles:
    - markdown ```json fences
    - extra text around JSON
    - singleton-list outputs: [{...}]
    - over-escaped JSON quotes: {\\\"key\\\": ...}
    - comments
    - invalid JSON escapes
    - trailing commas
    - accidental extra quote after string value
    - simple Python dicts via ast.literal_eval

    It does not fabricate missing fields. If parsing still fails, return None.
    Later canonicalization/validation should decide whether the recovered object
    is usable for SFT.
    """
    if not text or not isinstance(text, str):
        return None

    candidates: List[str] = []
    for cand in extract_json_candidate(text):
        _add_unique(candidates, cand)

    # Also repair whole text first, then extract again. This helps when braces/quotes
    # are hidden behind over-escaping.
    repaired_whole = text
    for fn in (
        strip_markdown_fence,
        strip_json_comments,
        repair_overescaped_json_quotes,
        repair_invalid_json_escapes,
        repair_unescaped_label_quotes,
        repair_extra_quote_after_json_string,
        repair_tabs_or_spaces_in_key_names,
        repair_trailing_commas,
    ):
        repaired_whole = fn(repaired_whole)

    for cand in extract_json_candidate(repaired_whole):
        _add_unique(candidates, cand)

    seen = set()

    for cand in list(candidates):
        variants: List[str] = []

        def add_variant(x: str) -> None:
            x = (x or "").strip()
            if x and x not in variants:
                variants.append(x)

        add_variant(cand)

        # Individual repairs.
        for fn in (
            strip_json_comments,
            repair_overescaped_json_quotes,
            repair_invalid_json_escapes,
            repair_unescaped_label_quotes,
            repair_extra_quote_after_json_string,
            repair_tabs_or_spaces_in_key_names,
            repair_trailing_commas,
        ):
            add_variant(fn(cand))

        # Combined repair pipeline.
        combined = cand
        for fn in (
            strip_json_comments,
            repair_overescaped_json_quotes,
            repair_invalid_json_escapes,
            repair_unescaped_label_quotes,
            repair_extra_quote_after_json_string,
            repair_tabs_or_spaces_in_key_names,
            repair_trailing_commas,
        ):
            combined = fn(combined)
        add_variant(combined)

        for v in variants:
            if v in seen:
                continue
            seen.add(v)

            obj = _try_json_dict(v)
            if obj is not None:
                return obj

    return None


def canonical_key(k: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def fuzzy_map_key(key: Any, expected: List[str], min_ratio: float = 0.72) -> Optional[str]:
    ck = canonical_key(key)
    if not ck:
        return None
    canon = {canonical_key(k): k for k in expected}
    if ck in canon:
        return canon[ck]

    # Safe common aliases.
    aliases = {
        "visualplan": "generation_plan",
        "visualplanning": "generation_plan",
        "prompt": "sdxl_prompt",
        "sdxlprompt": "sdxl_prompt",
        "positiveprompt": "sdxl_prompt",
        "negativeprompt": "negative_prompt",
        "caption": "ranking_caption",
        "rankingcaption": "ranking_caption",
        "headlinecore": "headline_core",
        "mainissue": "main_event_or_issue",
        "main_event": "main_event_or_issue",
        "main_event_or_topic": "main_event_or_issue",
        "causalorlogicalrelation": "relation",
        "causalrelation": "relation",
        "what_is_not_the_point": "what_should_not_be_emphasized",
        "whatshouldnotbeemphasized": "what_should_not_be_emphasized",
        "requiredcues": "required_visual_cues",
        "optionalcues": "optional_visual_cues",
        "forbiddencues": "forbidden_or_misleading_cues",
        "misleadingcues": "forbidden_or_misleading_cues",
        "strategy": "visual_strategy",
    }
    if ck in aliases and aliases[ck] in expected:
        return aliases[ck]

    # Partial/fuzzy recovery.
    partial = []
    for ec, ek in canon.items():
        if len(ck) >= 4 and (ec.startswith(ck) or ec.endswith(ck) or ck in ec):
            partial.append((len(ck) / max(len(ec), 1), ek))
        elif len(ec) >= 4 and ec in ck:
            partial.append((len(ec) / max(len(ck), 1), ek))
    if partial:
        partial.sort(reverse=True)
        if len(partial) == 1 or partial[0][0] >= partial[1][0] + 0.10:
            if partial[0][0] >= 0.60:
                return partial[0][1]

    matches = get_close_matches(ck, list(canon.keys()), n=2, cutoff=min_ratio)
    if matches:
        best = matches[0]
        if len(matches) == 1 or SequenceMatcher(None, ck, best).ratio() >= SequenceMatcher(None, ck, matches[1]).ratio() + 0.08:
            return canon[best]
    return None


def normalize_keys(d: Dict[str, Any], expected: List[str], notes: List[str], scope: str) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        nk = fuzzy_map_key(k, expected)
        if nk is None:
            notes.append(f"unmapped_key:{scope}:{k}")
            continue
        if nk in out and clean_text(out[nk]):
            notes.append(f"duplicate_mapped_key_ignored:{scope}:{k}->{nk}")
            continue
        out[nk] = v
        if str(k) != nk:
            notes.append(f"fuzzy_key:{scope}:{k}->{nk}")
    return out


# ---------------------------------------------------------------------
# Teacher prompt
# ---------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = """You are a title-grounded news visual planner for SDXL news/editorial image generation.

You will receive one news article title and VLM-parsed visible facts from one paired training image.
Use the article title as the main evidence. Use the VLM parse only as weak visual supervision from one possible editorial image.
Do not use article bodies or external retrieval. Use general world knowledge only to recognize entities and context; do not invent unsupported facts.

Return exactly one valid JSON object with exactly these five top-level keys:
title_understanding, generation_plan, sdxl_prompt, negative_prompt, ranking_caption.

title_understanding explains the title's main event, entities, issue, relation, implied context, visual risks, and uncertainty.

generation_plan is the visual plan derived from title_understanding, with weak paired-image VLM guidance:
image role, required cues, optional cues, forbidden or misleading cues, strategy, framing, and abstractness.

required_visual_cues must be supported by the title/title_understanding only.
The paired-image VLM parse may guide optional_visual_cues, visual_strategy, framing, abstractness, and risky artifacts to avoid.

The paired image is only one possible editorial choice, not the only correct image.
After understanding the title, consider why the paired image may have been used editorially for this title.
Use that paired-image reflection only for broad optional_visual_cues, visual_strategy, framing, abstractness, and risky artifacts to avoid.
Do not output this reflection as an extra field.
Do not reconstruct or copy the exact paired image.

VLM objects, scenes, and actions may become optional_visual_cues only if they are broad, safe, and title-compatible.
VLM layout, composition, person count, or shot type may inform visual_strategy, framing, and abstractness.
Do not put image-only exact clothing, exact colors, exact composition, exact person count, exact faces, readable text, logos, or incidental background into required_visual_cues unless the title supports them.

sdxl_prompt is a Stable Diffusion XL text-to-image prompt derived from title_understanding and generation_plan.
It must include the plan's required visual cues, reflect its visual_strategy/framing/abstractness, and avoid the plan's forbidden_or_misleading_cues.
Choose exactly one non-photorealistic or clearly synthetic editorial/news illustration style according to the title and generation_plan.
The image should look suitable for a news illustration, not like a realistic event photograph.
Possible single style phrases include synthetic editorial illustration, non-photorealistic news thumbnail illustration, stylized vector editorial art, flat editorial illustration, cut-paper collage, symbolic digital illustration, or map-like editorial illustration.

negative_prompt lists things to avoid in Stable Diffusion XL generation.
It should be derived from title_understanding and generation_plan, especially visual_risks and forbidden_or_misleading_cues.
It should include title-specific misleading details and common SDXL/editorial risks such as readable text, legible text, exact logos, watermark, fake documents, chart labels, photorealistic face, realistic portrait, exact facial likeness, fake news photo, and misleading literal scene.

ranking_caption is a short plain-language caption describing the intended generated image for later evaluation; it is not an SDXL prompt.

Write sdxl_prompt and negative_prompt as natural-language comma-separated phrases with spaces, not snake_case tags or variable-like tokens.
Write sdxl_prompt as a concise visual prompt, not a long paragraph, and put the most important title-specific cues early.
Write negative_prompt as concise comma-separated avoid terms.
Keep all reasoning inside the JSON fields.

Output JSON only.
No markdown, comments, explanations, rationale, or extra top-level keys.
Use normal JSON double quotes for every key and string value.
Do not use single quotes.
Do not put tabs, leading spaces, trailing spaces, or newlines inside JSON key names.
Do not wrap the whole JSON object inside a quoted string.
Use snake_case keys exactly.
"""


TEACHER_USER_TEMPLATE = """Create one structured JSON output for SDXL news/editorial image generation.

ARTICLE TITLE:
{article_title}

VLM-PARSED VISIBLE FACTS FROM ONE PAIRED TRAINING IMAGE:
{vlm_parse_json}

Use the title as the main evidence. Use VLM facts only as weak visual hints from one possible paired image.

Before writing the JSON, follow this construction process:
1. Understand the title: main issue/event, explicitly named entities, entity roles, relation, implied context, uncertainty, and visual risks.
2. Consider why the paired image may have been used editorially for this title.
3. Use that paired-image reflection only for broad optional cues, framing, visual strategy, abstractness, and risky artifacts to avoid.
4. Keep required_visual_cues supported by the title/title_understanding only.
5. Do not copy the paired image. Do not use image-only exact clothing, exact colors, exact composition, exact person count, exact faces, logos, readable text, or incidental background as required cues.
6. Write the generation_plan.
7. Write a non-empty positive SDXL prompt from the title understanding and generation plan, must choose an explicit non-photorealistic or clearly synthetic editorial/news illustration style.
8. Write a negative prompt from the title understanding and generation plan.
9. Write one short ranking caption.

Required field meanings:
- title_understanding: understand the title, named entities, main issue/event, context, relation, visual risks, and what not to emphasize.
- generation_plan: plan an editorial news image. required_visual_cues must be title-supported; optional_visual_cues may use broad safe VLM hints; forbidden_or_misleading_cues should include title risks, SDXL risks, and VLM-observed risky artifacts.
- sdxl_prompt: positive Stable Diffusion XL prompt for generating the image; concise comma-separated visual phrases; title-specific subject first; synthetic/non-photorealistic editorial style; clear framing.
- negative_prompt: Stable Diffusion XL negative prompt; concise comma-separated avoid terms, including readable text, exact logos, watermark, fake documents, chart labels, photorealistic face, exact facial likeness, fake news photo, and title-specific misleading details.
- ranking_caption: one short plain-language sentence describing the intended image meaning.

Allowed visual_strategy values:
direct_depiction, symbolic_editorial, location_anchor, person_or_group_scene, object_still_life, document_or_data_visual, map_or_geography, event_scene, conceptual_composite, other.

Allowed abstractness values:
literal, semi_symbolic, symbolic, other.


Return exactly one valid JSON object following this schema. 

{{
  "title_understanding": {{
    "headline_core": "short interpretation of the title",
    "named_entities": ["entity names if any"],
    "entity_roles": ["brief roles if any"],
    "main_event_or_issue": "main issue or event described by the title",
    "implied_context": "relevant context inferred only from the title and general knowledge",
    "relation": "relationship among entities/issues if any",
    "visual_risks": ["risks such as wrong likeness, misleading literal scene, readable text"],
    "what_should_not_be_emphasized": ["things that should not dominate the image"]
  }},
    "generation_plan": {{
    "image_role": "editorial illustration for a news thumbnail",
    "required_visual_cues": ["2-5 title-supported visual cues"],
    "optional_visual_cues": ["0-4 broad safe title-compatible cues, possibly inspired by VLM facts"],
    "forbidden_or_misleading_cues": ["title risks, SDXL/editorial risks, or VLM-observed risky artifacts"],
    "visual_strategy": "direct_depiction | symbolic_editorial | location_anchor | person_or_group_scene | object_still_life | document_or_data_visual | map_or_geography | event_scene | conceptual_composite | other",
    "framing": "brief composition/framing description",
    "abstractness": "literal | semi_symbolic | symbolic | other"
  }},
  "sdxl_prompt": "title-specific visual subject and cues from generation_plan, title-appropriate clearly synthetic/non-photorealistic editorial style, clear composition, no readable text, no exact logos, no watermark",
  "negative_prompt": "title- and plan-specific misleading detail, forbidden cue, readable text, exact logos, watermark, fake documents, chart labels, exact realistic likeness",
  "ranking_caption": "one concise caption describing what the generated image should show"
}}

Output rules:
- Output JSON only.
- Return exactly one valid JSON object.
- Use exactly these five top-level keys: title_understanding, generation_plan, sdxl_prompt, negative_prompt, ranking_caption.
- Use normal JSON double quotes for every key and string value.
- Do not use single quotes.
- Do not put tabs, leading spaces, trailing spaces, or newlines inside JSON key names.
- Do not wrap the whole JSON object inside a quoted string.
- Do not output markdown, comments, explanations, or extra top-level keys.
- Do not output the paired-image reflection as an extra field.
- Do not copy placeholder text from the schema.
- sdxl_prompt must be a non-empty positive SDXL prompt.
- negative_prompt must be non-empty.
- visual_strategy must be exactly one allowed value, not the full list.
- abstractness must be exactly one allowed value, not the full list.
- Write sdxl_prompt and negative_prompt as natural-language comma-separated phrases with spaces, not snake_case tags or variable-like tokens.
- Keep sdxl_prompt concise and visual, not a long paragraph.
- Keep negative_prompt concise, comma-separated, and do not repeat phrases.
"""



def build_teacher_messages(article_title: str, vlm_parse: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE.format(
                article_title=clean_text(article_title),
                vlm_parse_json=json.dumps(vlm_parse or {}, ensure_ascii=False, indent=2),
            ),
        },
    ]


# ---------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------

def load_teacher(model_name: str, load_in_4bit: bool, local_files_only: bool, bf16: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if bf16 else torch.float16
    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        kwargs["torch_dtype"] = dtype if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def model_input_device(model):
    import torch
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return next(emb.parameters()).device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_teacher_json(tokenizer, model, messages: List[Dict[str, str]], args) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    import torch

    current_messages = list(messages)
    last_text = ""
    with torch.inference_mode():
        for attempt in range(args.retries + 1):
            prompt = tokenizer.apply_chat_template(
                current_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            device = model_input_device(model)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            if args.temperature > 0:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                })
            else:
                gen_kwargs["do_sample"] = False

            out = model.generate(**gen_kwargs)
            gen = out[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            last_text = text
            obj = safe_json_loads(text)
            if obj is not None:
                return obj, text, True

            if attempt < args.retries:
                current_messages.append({"role": "assistant", "content": text[:4000]})
                current_messages.append({
                    "role": "user",
                   "content": (
                "The previous answer was not valid JSON. Return exactly one valid JSON object. "
                "Use normal JSON double quotes for all keys and string values. "
                "Do not use single quotes. Do not put tabs, leading spaces, trailing spaces, or newlines inside key names. "
                "Do not wrap the whole JSON object inside a quoted string. "
                "Do not use markdown. "
                "sdxl_prompt must be non-empty and must include an explicit non-photorealistic/editorial illustration style phrase. "
                "visual_strategy must be exactly one allowed value, not a list. "
                "abstractness must be exactly one allowed value. "
                "Use only these top-level keys: "
                "title_understanding, generation_plan, sdxl_prompt, negative_prompt, ranking_caption."
            ),
                })

    return None, last_text, False


# ---------------------------------------------------------------------
# Canonicalization and validation
# ---------------------------------------------------------------------

SCHEMA_PLACEHOLDERS = [
    "short interpretation of the title",
    "entity names explicitly from the title",
    "brief roles of the named entities",
    "main issue or event described by the title",
    "relevant context inferred only from the title and general knowledge",
    "relationship among entities/issues if any",
    "risks such as wrong likeness, misleading literal scene, readable text",
    "things that should not dominate the image",
    "2-5 title-supported visual cues",
    "2-5 title-supported concrete semantic cues",
    "2-5 concrete cues supported by the title",
    "0-4 broad safe title-compatible cues",
    "0-4 broad safe title-compatible cues, possibly inspired by VLM facts",
    "0-4 safe title-compatible supporting cues",
    "title risks, SDXL/editorial risks, or VLM-observed risky artifacts",
    "title risks, generic SDXL risks, or VLM-observed risky artifacts",
    "brief composition/framing description",
    "concise natural-language comma-separated SDXL positive prompt",
    "concise natural-language SDXL positive prompt with explicit non-photorealistic/editorial style",
    "concise comma-separated SDXL negative prompt",
    "one concise plain-language caption",
]

def normalized_for_similarity(s: Any) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def looks_like_schema_copy(x: Any, threshold: float = 0.82) -> bool:
    t = normalized_for_similarity(x)
    if not t:
        return False
    for ph in SCHEMA_PLACEHOLDERS:
        p = normalized_for_similarity(ph)
        if not p:
            continue
        if p in t:
            return True
        if SequenceMatcher(None, t, p).ratio() >= threshold:
            return True
        for part in re.split(r"[,;]", t):
            part = part.strip()
            if len(part) >= 8 and SequenceMatcher(None, part, p).ratio() >= threshold:
                return True
    return False


def has_bad_repetition(text: Any) -> bool:
    s = clean_text(text)
    if re.search(r"(.)\1{12,}", s):
        return True
    words = re.findall(r"\b\w+\b", s.lower())
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1
    if len(words) >= 9:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if any(c >= 3 for c in Counter(trigrams).values()):
            return True
    return False


def dedupe_comma_phrases(text: Any, max_items: int = 32) -> str:
    raw = clean_text(text)
    if not raw:
        return ""
    parts = [clean_text(p).strip(" .") for p in re.split(r"[,;]\s*", raw) if clean_text(p).strip(" .")]
    out, seen = [], set()
    for p in parts:
        key = re.sub(r"\s+", " ", p.lower())
        if key not in seen:
            out.append(p)
            seen.add(key)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def extract_entity_texts_and_roles(named_entities: Any) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    roles: List[str] = []
    if isinstance(named_entities, list):
        for ent in named_entities:
            if isinstance(ent, dict):
                text = clean_text(ent.get("text", ""))
                if text:
                    texts.append(text)
                role = clean_text(ent.get("role_in_title", "")) or clean_text(ent.get("role", ""))
                if role:
                    roles.append(role)
            elif clean_text(ent):
                texts.append(clean_text(ent))
    elif clean_text(named_entities):
        texts.append(clean_text(named_entities))
    return texts, roles

def normalize_allowed_value(value: Any, allowed: List[str], notes: List[str], field: str) -> str:
    raw = ensure_str(value)
    if not raw:
        return ""
    if raw in allowed:
        return raw

    mapped = fuzzy_map_key(raw, allowed, min_ratio=0.70)
    if mapped:
        notes.append(f"fuzzy_value:{field}:{raw}->{mapped}")
        return mapped

    return raw

def canonicalize_target(obj: Optional[Dict[str, Any]], title: str) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize a teacher target into the exact final five-key schema.

    This function is intentionally conservative:
    - repairs small key-name/schema mistakes;
    - fills only safe non-content defaults;
    - does not fabricate core visual-plan content.

    Missing required_visual_cues, visual_strategy, framing, abstractness, or
    sdxl_prompt should remain empty and be rejected later by validate_target().
    """
    notes: List[str] = []
    src = copy.deepcopy(obj) if isinstance(obj, dict) else {}

    # Normalize only the five expected top-level keys.
    # Other extra top-level fields are handled by normalize_keys() as unmapped keys.
    top = normalize_keys(src, FINAL_TOP_KEYS, notes, "top")

    # -----------------------------
    # title_understanding
    # -----------------------------
    raw_tu = top.get("title_understanding")
    if isinstance(raw_tu, dict):
        tu_src = normalize_keys(
            raw_tu,
            TITLE_KEYS + ["causal_or_logical_relation", "what_is_not_the_point"],
            notes,
            "title_understanding",
        )
    else:
        notes.append("missing_or_non_dict:title_understanding")
        tu_src = {}

    entity_texts, entity_roles_from_objects = extract_entity_texts_and_roles(
        tu_src.get("named_entities")
    )

    title_understanding = {
        "headline_core": ensure_str(tu_src.get("headline_core")),
        "named_entities": entity_texts,
        "entity_roles": as_list(tu_src.get("entity_roles")) or entity_roles_from_objects,
        "main_event_or_issue": ensure_str(tu_src.get("main_event_or_issue")),
        "implied_context": ensure_str(tu_src.get("implied_context")),
        "relation": (
            ensure_str(tu_src.get("relation"))
            or ensure_str(tu_src.get("causal_or_logical_relation"))
        ),
        "visual_risks": as_list(tu_src.get("visual_risks")),
        "what_should_not_be_emphasized": (
            as_list(tu_src.get("what_should_not_be_emphasized"))
            or as_list(tu_src.get("what_is_not_the_point"))
        ),
    }

    # Mild fallbacks are acceptable for interpretive/context fields,
    # but every synthesized value is recorded in notes.
    if not title_understanding["headline_core"]:
        notes.append("filled_missing:title_understanding.headline_core")
        title_understanding["headline_core"] = title

    if not title_understanding["main_event_or_issue"]:
        notes.append("filled_missing:title_understanding.main_event_or_issue")
        title_understanding["main_event_or_issue"] = title

    if not title_understanding["implied_context"]:
        notes.append("filled_missing:title_understanding.implied_context")
        title_understanding["implied_context"] = "inferred from the title only"

    if not title_understanding["visual_risks"]:
        notes.append("filled_missing:title_understanding.visual_risks")
        title_understanding["visual_risks"] = [
            "readable text",
            "exact logos",
            "misleading literal scene",
        ]

    if not title_understanding["what_should_not_be_emphasized"]:
        notes.append("filled_missing:title_understanding.what_should_not_be_emphasized")
        title_understanding["what_should_not_be_emphasized"] = [
            "unsupported details",
            "readable text",
            "exact logos",
        ]

    # -----------------------------
    # generation_plan
    # -----------------------------
    raw_gp = top.get("generation_plan")
    if isinstance(raw_gp, dict):
        gp_src = normalize_keys(raw_gp, PLAN_KEYS, notes, "generation_plan")
    else:
        notes.append("missing_or_non_dict:generation_plan")
        gp_src = {}

    generation_plan = {
        "image_role": ensure_str(gp_src.get("image_role")),
        "required_visual_cues": as_list(gp_src.get("required_visual_cues")),
        "optional_visual_cues": as_list(gp_src.get("optional_visual_cues")),
        "forbidden_or_misleading_cues": as_list(gp_src.get("forbidden_or_misleading_cues")),
        "visual_strategy": ensure_str(gp_src.get("visual_strategy")),
        "framing": ensure_str(gp_src.get("framing")),
        "abstractness": ensure_str(gp_src.get("abstractness")),
    }

    generation_plan["visual_strategy"] = normalize_allowed_value(
    generation_plan["visual_strategy"],
    VALID_VISUAL_STRATEGIES,
    notes,
    "generation_plan.visual_strategy",
)

    generation_plan["abstractness"] = normalize_allowed_value(
        generation_plan["abstractness"],
        VALID_ABSTRACTNESS,
        notes,
        "generation_plan.abstractness",
    )



    # Safe constant default.
    # This does not define visual content, so it is okay to fill.
    if not generation_plan["image_role"]:
        notes.append("filled_missing:generation_plan.image_role")
        generation_plan["image_role"] = "editorial illustration for a news thumbnail"

    # Conservative risk fallback.
    # Do NOT fill required_visual_cues / visual_strategy / framing / abstractness here.
    # Missing core plan fields should fail validate_target().
    if not generation_plan["forbidden_or_misleading_cues"]:
        notes.append("filled_missing:generation_plan.forbidden_or_misleading_cues")
        generation_plan["forbidden_or_misleading_cues"] = [
            "readable text",
            "exact logos",
            "watermark",
            "fake documents",
            "photorealistic face",
            "exact realistic likeness",
        ]

    # -----------------------------
    # top-level prompt fields
    # -----------------------------
    sdxl_prompt = ensure_str(top.get("sdxl_prompt"))

    negative_prompt = dedupe_comma_phrases(
        ensure_str(top.get("negative_prompt")) or DEFAULT_NEGATIVE,
        max_items=28,
    )
    if not negative_prompt:
        notes.append("filled_missing:negative_prompt")
        negative_prompt = DEFAULT_NEGATIVE

    ranking_caption = ensure_str(top.get("ranking_caption"))
    if not ranking_caption:
        notes.append("filled_missing:ranking_caption")
        ranking_caption = f"{title}: title-grounded editorial illustration"

    target = {
        "title_understanding": title_understanding,
        "generation_plan": generation_plan,
        "sdxl_prompt": sdxl_prompt,
        "negative_prompt": negative_prompt,
        "ranking_caption": ranking_caption,
    }
    return target, notes


def validate_target(target: Dict[str, Any], title: str, strict: bool = True) -> Tuple[bool, List[str], List[str]]:
    """Return ok, errors, warnings."""
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(target, dict):
        return False, ["target_not_dict"], []

    keys = list(target.keys())
    if keys != FINAL_TOP_KEYS:
        # We canonicalize order before writing, so this should pass.
        if set(keys) != set(FINAL_TOP_KEYS):
            errors.append(f"wrong_top_level_keys:{keys}")
        else:
            warnings.append("top_key_order_was_normalized")

    tu = target.get("title_understanding")
    gp = target.get("generation_plan")
    if not isinstance(tu, dict):
        errors.append("missing_title_understanding_dict")
    if not isinstance(gp, dict):
        errors.append("missing_generation_plan_dict")

    if isinstance(tu, dict):
        for k in TITLE_KEYS:
            if k not in tu:
                errors.append(f"missing_title_key:{k}")
        if not clean_text(tu.get("headline_core")):
            errors.append("empty_headline_core")
        if looks_like_schema_copy(json.dumps(tu, ensure_ascii=False)):
            errors.append("title_understanding_looks_like_schema_copy")



    if isinstance(gp, dict):

        for k in PLAN_KEYS:
            if k not in gp:
                errors.append(f"missing_plan_key:{k}")
        required = as_list(gp.get("required_visual_cues"))
        if len(required) < 2:
            errors.append("too_few_required_visual_cues")
        if len(required) > 6:
            warnings.append("too_many_required_visual_cues")
        if not clean_text(gp.get("visual_strategy")):
            errors.append("empty_visual_strategy")
        if not clean_text(gp.get("framing")):
            errors.append("empty_framing")
        if not clean_text(gp.get("abstractness")):
            errors.append("empty_abstractness")
        if looks_like_schema_copy(json.dumps(gp, ensure_ascii=False)):
            errors.append("generation_plan_looks_like_schema_copy")

        strategy = clean_text(gp.get("visual_strategy"))
        abstractness = clean_text(gp.get("abstractness"))

        if strategy and strategy not in VALID_VISUAL_STRATEGIES:
            errors.append(f"invalid_visual_strategy:{strategy}")

        if abstractness and abstractness not in VALID_ABSTRACTNESS:
            errors.append(f"invalid_abstractness:{abstractness}")

    pos = clean_text(target.get("sdxl_prompt"))
    neg = clean_text(target.get("negative_prompt"))
    cap = clean_text(target.get("ranking_caption"))

    if len(pos) < 40:
        errors.append("sdxl_prompt_too_short")
    if len(neg) < 10:
        errors.append("negative_prompt_too_short")
    if not cap:
        errors.append("missing_ranking_caption")
    if looks_like_schema_copy(pos):
        errors.append("sdxl_prompt_looks_like_schema_copy")
    if looks_like_schema_copy(cap):
        errors.append("ranking_caption_looks_like_schema_copy")
    if has_bad_repetition(pos):
        errors.append("sdxl_prompt_bad_repetition")
    if has_bad_repetition(neg):
        errors.append("negative_prompt_bad_repetition")
    if has_bad_repetition(cap):
        errors.append("ranking_caption_bad_repetition")

    low_pos = pos.lower()
    low_neg = neg.lower()

    # Semantic-ish quality checks. These are intentionally conservative.
    style_terms = [
        "non-photorealistic", "synthetic", "editorial illustration", "illustration",
        "vector", "cut-paper", "symbolic", "stylized", "news thumbnail",
    ]
    if not any(t in low_pos for t in style_terms):
        errors.append("sdxl_prompt_lacks_synthetic_editorial_style")

    # Do not reject phrases such as "no readable text" or "without exact logos".
    # We only reject risky terms if they are not clearly negated/forbidden in the positive prompt.
    risky_positive_terms = [
        "readable text", "legible text", "exact logo", "exact logos", "watermark", "watermarks",
        "photorealistic portrait", "realistic portrait", "exact facial likeness",
        "fake document", "fake documents", "newspaper headline", "newspaper headlines",
    ]

    def risky_term_is_negated(text: str, term: str) -> bool:
        # Accept phrases such as:
        #   "no readable text", "without exact logos", "avoid fake documents",
        #   "no watermark", "watermark-free".
        variants = {term}
        if term.endswith("s"):
            variants.add(term[:-1])
        else:
            variants.add(term + "s")

        for v in variants:
            pattern = re.compile(r"\b" + re.escape(v) + r"\b")
            for m in pattern.finditer(text):
                before = text[max(0, m.start() - 56):m.start()]
                after = text[m.end():min(len(text), m.end() + 24)]
                if re.search(r"\b(no|not|without|avoid|avoiding|exclude|excluding|forbid|forbidden|free of)\b", before):
                    return True
                if re.search(r"^\s*[- ]?free\b", after):
                    return True
        return False

    for term in risky_positive_terms:
        if term in low_pos and not risky_term_is_negated(low_pos, term):
            errors.append(f"risky_positive_prompt_term:{term}")

    for term in ["readable text", "exact logos", "watermark"]:
        if term not in low_neg:
            warnings.append(f"negative_prompt_missing_common_risk:{term}")

    return len(errors) == 0, errors, warnings


def canonical_json_for_assistant(target: Dict[str, Any]) -> str:
    ordered = {k: target[k] for k in FINAL_TOP_KEYS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def repair_level_is_heavy(notes: List[str], max_fuzzy_keys: int = 4) -> bool:
    """Heuristic for excluding teacher rows that needed too much schema repair."""
    if not notes:
        return False
    fuzzy = sum(1 for n in notes if str(n).startswith("fuzzy_key:"))
    unmapped = sum(1 for n in notes if str(n).startswith("unmapped_key:"))
    duplicate = sum(1 for n in notes if str(n).startswith("duplicate_mapped_key_ignored:"))
    # Dropping teacher-only keys is fine; lots of fuzzy/unmapped/duplicate keys is suspicious.
    return fuzzy > max_fuzzy_keys or unmapped > 2 or duplicate > 1


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

def load_done_keys(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    for row in read_jsonl(str(p)):
        aid = clean_text(row.get("article_id", ""))
        iid = clean_text(row.get("image_id", ""))
        if aid:
            done.add((aid, iid))
    return done


def cmd_inspect_vlm(args) -> None:
    by_id = load_vlm_by_image_id(args.vlm)
    print(f"Unique image_id with VLM parse: {len(by_id)}")
    for i, (image_id, vlm) in enumerate(by_id.items()):
        if i >= args.show:
            break
        print("-" * 80)
        print(image_id)
        print(json.dumps(vlm, ensure_ascii=False, indent=2))


def cmd_make_silver(args) -> None:
    tokenizer, model = load_teacher(
        args.teacher_model,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        bf16=args.bf16,
    )
    vlm_by_id = load_vlm_by_image_id(args.vlm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(args.out) if args.resume else set()
    mode = "a" if args.resume and out_path.exists() else "w"

    meta = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "method": f"{args.teacher_model} teacher sees title + VLM parsed visible facts; student SFT input is title-only; assistant target uses five-key schema.",
        "teacher_model": args.teacher_model,
        "final_top_keys": FINAL_TOP_KEYS,
        "default_negative": DEFAULT_NEGATIVE,
    }
    Path(args.out + ".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter()
    with open(out_path, mode, encoding="utf-8") as f:
        for idx, row in enumerate(read_article_rows(args.articles, args.title_col, args.article_id_col, args.image_col), 1):
            if args.limit and counts["processed"] >= args.limit:
                break
            aid = clean_text(row.get("article_id", ""))
            iid = clean_text(row.get("image_id", ""))
            if aid and (aid, iid) in done:
                counts["skipped_resume"] += 1
                continue

            title = clean_text(row["article_title"])
            vlm = vlm_by_id.get(iid, {})

            messages = build_teacher_messages(title, vlm)
            raw_obj, raw_text, parse_ok = generate_teacher_json(tokenizer, model, messages, args)
            target, notes = canonicalize_target(raw_obj, title)
            valid, errors, warnings = validate_target(target, title)

            out_row = {
                **row,
                "vlm_parse_used": bool(vlm),
                "vlm_parse": vlm if args.save_vlm_parse else None,
                "teacher_target": target,
                "parse_ok": parse_ok,
                "is_valid": valid,
                "validation_errors": errors,
                "validation_warnings": warnings,
                "normalization_notes": notes,
                "teacher_model": args.teacher_model,
            }
            if not args.save_vlm_parse:
                out_row.pop("vlm_parse", None)
            if args.save_raw:
                out_row["teacher_raw_text"] = raw_text

            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

            counts["processed"] += 1
            counts["parse_ok"] += int(parse_ok)
            counts["valid"] += int(valid)
            counts["vlm_used"] += int(bool(vlm))
            for e in errors:
                counts[f"error:{e}"] += 1
            for w in warnings:
                counts[f"warning:{w}"] += 1

            if counts["processed"] % 10 == 0:
                print(
                    f"Processed {counts['processed']}; valid={counts['valid']}; "
                    f"parse_ok={counts['parse_ok']}; vlm_used={counts['vlm_used']}",
                    flush=True,
                )

    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts),
        "valid_rate": round(counts["valid"] / counts["processed"], 6) if counts["processed"] else 0.0,
        "parse_ok_rate": round(counts["parse_ok"] / counts["processed"], 6) if counts["processed"] else 0.0,
        "vlm_used_rate": round(counts["vlm_used"] / counts["processed"], 6) if counts["processed"] else 0.0,
    }
    Path(args.out + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote silver file: {args.out}")


def cmd_validate_filter(args) -> None:
    valid_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    counts = Counter()

    for row in read_jsonl(args.input):
        title = clean_text(row.get("article_title", ""))
        target, notes = canonicalize_target(row.get("teacher_target"), title)
        ok, errors, warnings = validate_target(target, title)

        row["teacher_target"] = target
        row["is_valid"] = ok
        row["validation_errors"] = errors
        row["validation_warnings"] = warnings
        row["normalization_notes"] = as_list(row.get("normalization_notes")) + notes

        counts["total"] += 1
        counts["valid"] += int(ok)
        counts["bad"] += int(not ok)
        counts["vlm_used"] += int(bool(row.get("vlm_parse_used")))
        for e in errors:
            counts[f"error:{e}"] += 1
        for w in warnings:
            counts[f"warning:{w}"] += 1

        if ok:
            valid_rows.append(row)
        else:
            bad_rows.append(row)

    write_jsonl(args.valid_out, valid_rows)
    write_jsonl(args.bad_out, bad_rows)

    report = {
        "counts": dict(counts),
        "valid_rate": round(counts["valid"] / counts["total"], 6) if counts["total"] else 0.0,
        "valid_out": args.valid_out,
        "bad_out": args.bad_out,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_audit(args) -> None:
    counts = Counter()
    errors = Counter()
    warnings = Counter()
    strategies = Counter()
    abstractness = Counter()
    recovery_notes = Counter()

    for row in read_jsonl(args.input):
        counts["total"] += 1
        if row.get("is_valid"):
            counts["valid"] += 1
        if row.get("vlm_parse_used"):
            counts["vlm_used"] += 1
        for e in as_list(row.get("validation_errors")):
            errors[e] += 1
        for w in as_list(row.get("validation_warnings")):
            warnings[w] += 1
        for n in as_list(row.get("normalization_notes")):
            recovery_notes[n] += 1

        tgt = row.get("teacher_target") if isinstance(row.get("teacher_target"), dict) else {}
        gp = tgt.get("generation_plan") if isinstance(tgt.get("generation_plan"), dict) else {}
        strategies[clean_text(gp.get("visual_strategy")) or "NA"] += 1
        abstractness[clean_text(gp.get("abstractness")) or "NA"] += 1

    report = {
        "counts": dict(counts),
        "valid_rate": round(counts["valid"] / counts["total"], 6) if counts["total"] else 0.0,
        "top_errors": errors.most_common(30),
        "top_warnings": warnings.most_common(30),
        "visual_strategy": strategies.most_common(),
        "abstractness": abstractness.most_common(),
        "top_normalization_notes": recovery_notes.most_common(30),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_sample_audit(args) -> None:
    rows = list(read_jsonl(args.input))
    rng = random.Random(args.seed)
    if args.only_valid:
        rows = [r for r in rows if r.get("is_valid", True)]
    rng.shuffle(rows)
    sample = rows[: args.n]

    # Produce a compact file convenient for manual checking.
    audit_rows = []
    for r in sample:
        audit_rows.append({
            "article_id": r.get("article_id"),
            "image_id": r.get("image_id"),
            "article_title": r.get("article_title"),
            "vlm_parse_used": r.get("vlm_parse_used"),
            "teacher_target": r.get("teacher_target"),
            "validation_errors": r.get("validation_errors", []),
            "validation_warnings": r.get("validation_warnings", []),
            "manual_check": {
                "schema_ok": None,
                "title_grounding_ok": None,
                "image_overcopying": None,
                "sdxl_prompt_ok": None,
                "negative_prompt_ok": None,
                "keep_for_sft": None,
                "notes": "",
            },
        })
    write_jsonl(args.out, audit_rows)
    print(f"Wrote manual audit sample: {args.out}; rows={len(audit_rows)}")


def cmd_make_sft(args) -> None:
    rows: List[Dict[str, Any]] = []
    counts = Counter()

    for row in read_jsonl(args.input):
        # Final SFT export always keeps only rows marked valid by the silver/validation step.
        if not bool(row.get("is_valid", True)):
            counts["skipped_invalid"] += 1
            continue

        title = clean_text(row.get("article_title", ""))
        target, notes = canonicalize_target(row.get("teacher_target"), title)
        ok, errors, warnings = validate_target(target, title)

        if not title:
            counts["skipped_missing_title"] += 1
            continue
        if not ok:
            counts["skipped_failed_revalidation"] += 1
            continue
        if args.require_vlm and not bool(row.get("vlm_parse_used")):
            counts["skipped_no_vlm_parse"] += 1
            continue

        combined_notes = list(dict.fromkeys(as_list(row.get("normalization_notes")) + notes))
        if repair_level_is_heavy(combined_notes, max_fuzzy_keys=args.max_fuzzy_keys):
            counts["skipped_heavy_schema_repair"] += 1
            continue

        assistant_content = canonical_json_for_assistant(target)

        rows.append({
            "messages": [
                {"role": "system", "content": student_system_prompt()},
                {"role": "user", "content": schema_instruction(title)},
                {"role": "assistant", "content": assistant_content},
            ],
            "article_id": row.get("article_id"),
            "image_id": row.get("image_id"),
            "article_title": title,
            "student_input_uses_vlm": False,
            "teacher_used_vlm_parse": bool(row.get("vlm_parse_used")),
            "source_validation_warnings": warnings,
            "source_normalization_notes": combined_notes,
        })
        counts["written"] += 1

    write_jsonl(args.out, rows)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "method": "Input messages match current inference prompt. Assistant target is strict five-key schema.",
        "num_rows": counts["written"],
        "final_top_keys": FINAL_TOP_KEYS,
        "student_input_uses_vlm": False,
    }
    Path(args.out + ".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    print(f"Wrote SFT JSONL: {args.out}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run 4 Qwen3B LoRA SFT data preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect-vlm")
    p.add_argument("--vlm", required=True)
    p.add_argument("--show", type=int, default=3)
    p.set_defaults(func=cmd_inspect_vlm)

    p = sub.add_parser("make-silver")
    p.add_argument("--articles", required=True)
    p.add_argument("--vlm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--title-col", default="article_title")
    p.add_argument("--article-id-col", default="article_id")
    p.add_argument("--image-col", default="image_id")
    p.add_argument("--max-new-tokens", type=int, default=1800)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--save-vlm-parse", action="store_true")
    p.set_defaults(func=cmd_make_silver)

    p = sub.add_parser("validate-filter")
    p.add_argument("--input", required=True)
    p.add_argument("--valid-out", required=True)
    p.add_argument("--bad-out", required=True)
    p.set_defaults(func=cmd_validate_filter)

    p = sub.add_parser("audit")
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("sample-audit")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-invalid", dest="only_valid", action="store_false")
    p.set_defaults(only_valid=True)

    p.set_defaults(func=cmd_sample_audit)

    p = sub.add_parser("make-sft")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-vlm", action="store_true", help="Keep only rows where a paired VLM parse was available.")
    p.add_argument("--max-fuzzy-keys", type=int, default=4, help="Maximum allowed fuzzy key repairs before considering a row heavily repaired.")
    p.set_defaults(func=cmd_make_sft)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
