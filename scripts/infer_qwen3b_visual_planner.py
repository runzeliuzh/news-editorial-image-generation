#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Qwen2.5-3B-Instruct visual planner inference on article titles.

Methodological goal:
  - Zero-shot mode: base Qwen2.5-3B-Instruct only.
  - LoRA mode: same base model plus optional --adapter.
  - No few-shot article/output examples.
  - Title-only input.
  - Same prompt, decoding, validation-aware retry, and fallback logic for both modes.
  - Last-resort fallback only after retries fail, so every row can still receive an SDXL prompt.
  - Record success/fallback rates for fair LoRA vs zero-shot comparison.

Output is compatible with downstream SDXL scripts expecting:
  row["student_target"]["sdxl_prompt"]
  row["student_target"]["negative_prompt"]

"""
from __future__ import annotations

import argparse
import copy
import ast
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_NEGATIVE = (
    "photorealistic face, realistic face, detailed human face, realistic portrait, "
    "face close-up, headshot, exact facial likeness, AI-generated fake portrait, "
    "uncanny human face, photorealistic fake news photo, readable text, legible text, "
    "dense text, small text, letters, words, chart labels, document text, exact logos, "
    "watermark, clutter, unrelated objects, distorted anatomy, graphic gore, "
    "misleading literal scene"
)

REQUIRED_TOP_LEVEL_KEYS = [
    "title_understanding",
    "generation_plan",
    "sdxl_prompt",
    "negative_prompt",
    "ranking_caption",
]

MIN_SDXL_PROMPT_CHARS = 30
MIN_NEGATIVE_PROMPT_CHARS = 10

def clean_text(x: Any) -> str:
    return " ".join(str(x or "").replace("\n", " ").replace("\t", " ").split())


def dedupe_comma_phrases(text: Any, max_items: int = 32) -> str:
    """De-duplicate comma/semicolon separated prompt phrases while preserving order."""
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


def has_repeated_comma_phrases(text: Any) -> bool:
    parts = [clean_text(p).strip(" .").lower() for p in re.split(r"[,;]\s*", clean_text(text)) if clean_text(p).strip(" .")]
    return len(parts) >= 4 and len(set(parts)) < len(parts)


def has_bad_token_repetition(text: Any, window: int = 3, max_repeats: int = 3) -> bool:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", clean_text(text).lower())
    if len(toks) < window * max_repeats:
        return False
    ngrams = [tuple(toks[i:i+window]) for i in range(len(toks)-window+1)]
    from collections import Counter
    return any(c >= max_repeats for c in Counter(ngrams).values())


def postprocess_target_fields(target: Dict[str, Any]) -> Dict[str, Any]:
    """Light deterministic cleanup of text fields. Does not add semantic content."""
    if not isinstance(target, dict):
        return target
    for k in ("sdxl_prompt", "negative_prompt", "ranking_caption"):
        if k in target:
            target[k] = clean_text(target.get(k, ""))
    if clean_text(target.get("negative_prompt", "")):
        target["negative_prompt"] = dedupe_comma_phrases(target["negative_prompt"], max_items=24)
    if isinstance(target.get("generation_plan"), dict):
        gp = target["generation_plan"]
        for key in ("required_visual_cues", "optional_visual_cues", "forbidden_or_misleading_cues"):
            gp[key] = as_clean_list(gp.get(key))
    return target


def trim_words(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:max_words])


def sanitize_title_for_prompt(title: str) -> str:
    """Conservative title cleanup for deterministic fallback only."""
    title = clean_text(title)
    title = title.replace('"', "").replace("'", "")
    title = title.replace("{", " ").replace("}", " ")
    title = title.replace("[", " ").replace("]", " ")
    title = title.replace("|", " ")
    title = re.sub(r"\s+", " ", title).strip(" ,.;:")
    return title


def build_fallback_prompt(title: str, max_title_words: int = 18) -> str:
    """Fixed title-only fallback prompt, aligned with TemplateSDXL baseline style."""
    safe_title = trim_words(sanitize_title_for_prompt(title), max_title_words)
    if not safe_title:
        safe_title = "news event"
    return clean_text(
        f"{safe_title}. "
        "non-photorealistic editorial illustration, news thumbnail, "
        "flat vector cut-paper style, symbolic visual metaphor, "
        "clear central subject, simple background, balanced composition, "
        "muted contrast, no readable text, no exact logos, no watermark"
    ).strip(" ,.;:")


def make_title_template_fallback_target(title: str) -> Dict[str, Any]:
    """Deterministic title-only fallback after Qwen fails to produce a usable positive prompt."""
    return {
        "title_understanding": {
            "headline_core": title,
            "named_entities": [],
            "entity_roles": [],
            "main_event_or_issue": title,
            "implied_context": "title-only deterministic fallback",
            "relation": "",
            "visual_risks": ["readable text", "exact logos", "realistic face"],
            "what_should_not_be_emphasized": ["unsupported details", "readable text", "exact logos"],
        },
        "generation_plan": {
            "image_role": "editorial illustration for a news thumbnail",
            "required_visual_cues": [title],
            "optional_visual_cues": [],
            "forbidden_or_misleading_cues": ["readable text", "exact logos", "realistic public-figure likeness"],
            "visual_strategy": "other",
            "framing": "centered clearly synthetic news/editorial image composition",
            "abstractness": "other",
        },
        "sdxl_prompt": build_fallback_prompt(title),
        "negative_prompt": DEFAULT_NEGATIVE,
        "ranking_caption": title,
    }


def make_plan_based_fallback_target(title: str, obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Fallback that uses Qwen's generation_plan but force-synthesizes the SDXL prompt.

    This is used only after Qwen failed to write a usable positive sdxl_prompt.
    Therefore we must not preserve a bad/copied/too-short sdxl_prompt from the
    model; we always rebuild the positive prompt from the usable Qwen plan and
    use the baseline DEFAULT_NEGATIVE for SDXL control.
    """
    target, notes = normalize_target(obj, title, synthesize_missing=False)
    target["sdxl_prompt"] = synthesize_prompt_from_plan(title, target)
    target["negative_prompt"] = DEFAULT_NEGATIVE
    if not clean_text(target.get("ranking_caption", "")):
        cues = as_clean_list(target.get("generation_plan", {}).get("required_visual_cues"))
        cue_text = ", ".join(cues[:3]) if cues else "title-grounded editorial scene"
        target["ranking_caption"] = f"{title}: {cue_text}"
    notes.append("force_synthesized_sdxl_prompt_from_qwen_generation_plan")
    notes.append("filled_negative_prompt_with_baseline_template_negative")
    return target, notes


def as_clean_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [clean_text(v) for v in x if clean_text(v)]
    if clean_text(x):
        return [clean_text(x)]
    return []


def repair_invalid_json_escapes(text: str) -> str:
    r"""Repair invalid JSON backslash escapes such as \£ or \&."""
    if not text:
        return ""
    return re.sub(r'\\(?!["\\/bfnrtu])', '', text)


def repair_unescaped_label_quotes(text: str) -> str:
    """Repair short unescaped quoted label phrases, e.g. "For Sale" sign."""
    if not text:
        return ""
    return re.sub(
        r'"([^"\n]{1,80})"\s+(sign|label|banner|poster|text)"',
        r'"\1 \2"',
        text,
        flags=re.IGNORECASE,
    )


def repair_doubled_inner_quotes(text: str) -> str:
    """Repair Qwen mistakes like ""readable information"" inside JSON values."""
    if not text:
        return ""
    return re.sub(r'""([^"\n{}\[\]:,]{1,120})""', r'"\1"', text)


def repair_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    if not text:
        return ""
    return re.sub(r",\s*([}\]])", r"\1", text)


def strip_json_comments(text: str) -> str:
    """Remove simple // comments that Qwen sometimes inserts in JSON-like output."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        if "//" in line and "http://" not in line and "https://" not in line:
            line = line.split("//", 1)[0].rstrip()
        lines.append(line)
    return "\n".join(lines)


def _unwrap_dict_like(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return None


def _try_json_dict_with_notes(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    try:
        return _unwrap_dict_like(json.loads(text)), []
    except Exception:
        pass
    # Conservative fallback for Python-like dicts with single quotes/trailing commas.
    try:
        obj = _unwrap_dict_like(ast.literal_eval(text))
        if obj is not None:
            return obj, ["json_repair:ast_literal_eval"]
        return None, []
    except Exception:
        return None, []


def _try_json_dict(text: str) -> Optional[Dict[str, Any]]:
    obj, _notes = _try_json_dict_with_notes(text)
    return obj


def _json_repair_variants_with_notes(text: str) -> List[Tuple[str, List[str]]]:
    """Generate conservative JSON-repair variants and record what was changed."""
    variants: List[Tuple[str, List[str]]] = []
    seen = set()

    def add(x: str, notes: List[str]) -> None:
        if x is not None and x not in seen:
            variants.append((x, notes))
            seen.add(x)

    add(text, [])

    stripped = strip_json_comments(text)
    if stripped != text:
        add(stripped, ["json_repair:strip_comments"])

    # Apply single repairs.
    for base, base_notes in list(variants):
        repairs = [
            ("invalid_escapes", repair_invalid_json_escapes),
            ("unescaped_label_quotes", repair_unescaped_label_quotes),
            ("doubled_inner_quotes", repair_doubled_inner_quotes),
            ("trailing_commas", repair_trailing_commas),
        ]
        for name, fn in repairs:
            x = fn(base)
            if x != base:
                add(x, base_notes + [f"json_repair:{name}"])

    # Apply all repairs in a stable order.
    for base, base_notes in list(variants):
        x = base
        notes = list(base_notes)
        for name, fn in (
            ("strip_comments", strip_json_comments),
            ("invalid_escapes", repair_invalid_json_escapes),
            ("unescaped_label_quotes", repair_unescaped_label_quotes),
            ("doubled_inner_quotes", repair_doubled_inner_quotes),
            ("trailing_commas", repair_trailing_commas),
        ):
            y = fn(x)
            if y != x:
                notes.append(f"json_repair:{name}")
                x = y
        if x != base:
            add(x, notes)

    return variants


def safe_json_loads_with_notes(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse one JSON object from model output, with conservative repair notes."""
    if not text:
        return None, ["json_parse:empty_text"]

    raw = text.strip()
    notes0: List[str] = []
    cleaned = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    cleaned2 = re.sub(r"```$", "", cleaned).strip()
    if cleaned2 != raw:
        notes0.append("json_extract:removed_markdown_fence")
    raw = cleaned2

    candidates: List[Tuple[str, List[str]]] = [(raw, notes0)]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj_text = raw[start:end + 1]
        if obj_text != raw:
            candidates.append((obj_text, notes0 + ["json_extract:object_substring"]))

    # Also try singleton-list outputs such as [ { ... } ].
    lstart = raw.find("[")
    lend = raw.rfind("]")
    if lstart >= 0 and lend > lstart:
        arr_text = raw[lstart:lend + 1]
        if all(arr_text != c for c, _ in candidates):
            candidates.append((arr_text, notes0 + ["json_extract:list_substring"]))

    for cand, cand_notes in candidates:
        for rc, repair_notes in _json_repair_variants_with_notes(cand):
            obj, parse_backend_notes = _try_json_dict_with_notes(rc)
            if obj is not None:
                final_notes = cand_notes + repair_notes + parse_backend_notes
                if cand.lstrip().startswith("["):
                    final_notes.append("json_repair:unwrapped_singleton_list_if_needed")
                if rc != cand and not repair_notes:
                    final_notes.append("json_repair:unknown")
                return obj, final_notes
    return None, notes0 + ["json_parse:failed_after_repairs"]


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    obj, _notes = safe_json_loads_with_notes(text)
    return obj

def read_titles(args) -> Iterable[Dict[str, str]]:
    """Read title rows from JSONL or CSV."""
    if args.articles.endswith(".jsonl"):
        with open(args.articles, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                yield {
                    "article_id": clean_text(row.get(args.article_id_col, idx)),
                    "image_id": clean_text(row.get(args.image_col, "")),
                    "article_title": clean_text(row.get(args.title_col, "")),
                }
    else:
        import pandas as pd
        df = pd.read_csv(args.articles, dtype=str)
        if args.title_col not in df.columns:
            raise ValueError(f"Missing title column {args.title_col}. Available columns: {list(df.columns)}")
        for idx, row in df.iterrows():
            yield {
                "article_id": clean_text(row.get(args.article_id_col, idx)) if args.article_id_col in df.columns else str(idx),
                "image_id": clean_text(row.get(args.image_col, "")) if args.image_col in df.columns else "",
                "article_title": clean_text(row.get(args.title_col, "")),
            }



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


def build_messages(title: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": student_system_prompt()},
        {"role": "user", "content": schema_instruction(title)},
    ]

EXPECTED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS
EXPECTED_TITLE_UNDERSTANDING_KEYS = [
    "headline_core",
    "named_entities",
    "entity_roles",
    "main_event_or_issue",
    "implied_context",
    "relation",
    "visual_risks",
    "what_should_not_be_emphasized",
]
EXPECTED_GENERATION_PLAN_KEYS = [
    "image_role",
    "required_visual_cues",
    "optional_visual_cues",
    "forbidden_or_misleading_cues",
    "visual_strategy",
    "framing",
    "abstractness",
]

LIST_FIELDS_TITLE = {"named_entities", "entity_roles", "visual_risks", "what_should_not_be_emphasized"}
LIST_FIELDS_PLAN = {"required_visual_cues", "optional_visual_cues", "forbidden_or_misleading_cues"}


def _canonical_key_token(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _key_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _canonical_key_token(a), _canonical_key_token(b)).ratio()


def fuzzy_map_key(key: Any, expected_keys: List[str], min_ratio: float = 0.72) -> Optional[str]:
    """Map misspelled/variant keys to schema keys without a hardcoded alias table.

    The mapping is schema-driven: exact canonical match, safe prefix/suffix/substring
    match for meaningful partial keys, then difflib similarity. It does not create
    semantic content; it only recovers keys Qwen clearly intended to write.
    """
    kt = _canonical_key_token(key)
    if not kt:
        return None

    token_to_key = {_canonical_key_token(k): k for k in expected_keys}
    if kt in token_to_key:
        return token_to_key[kt]

    # Prefix/suffix recovery handles prompt -> sdxl_prompt, entities -> named_entities,
    # roles -> entity_roles, strategy -> visual_strategy, etc.
    partial_hits = []
    for et, ek in token_to_key.items():
        if len(kt) >= 4 and (et.startswith(kt) or et.endswith(kt) or kt in et):
            partial_hits.append((len(kt) / max(len(et), 1), ek))
        elif len(et) >= 4 and et in kt:
            partial_hits.append((len(et) / max(len(kt), 1), ek))
    if partial_hits:
        partial_hits.sort(reverse=True)
        best_score, best_key = partial_hits[0]
        # Avoid overly generic partial mappings such as "title" ->
        # "title_understanding". Short partial matches are accepted only when
        # they cover a meaningful fraction of the expected schema key.
        if best_score >= 0.45 and (len(partial_hits) == 1 or best_score >= partial_hits[1][0] + 0.10):
            return best_key

    expected_tokens = list(token_to_key.keys())
    matches = get_close_matches(kt, expected_tokens, n=2, cutoff=min_ratio)
    if not matches:
        # Slightly looser only for long keys where typos are common.
        if len(kt) >= 10:
            matches = get_close_matches(kt, expected_tokens, n=2, cutoff=0.66)
    if matches:
        best = matches[0]
        if len(matches) == 1 or SequenceMatcher(None, kt, best).ratio() >= SequenceMatcher(None, kt, matches[1]).ratio() + 0.08:
            return token_to_key[best]
    return None


def normalize_keys_by_schema(
    d: Dict[str, Any],
    expected_keys: List[str],
    notes: Optional[List[str]] = None,
    scope: str = "",
) -> Dict[str, Any]:
    """Schema-driven fuzzy key normalization, preserving filled canonical values."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        new_k = fuzzy_map_key(k, expected_keys)
        if not new_k:
            if notes is not None:
                notes.append(f"unmapped_key:{scope}:{k}")
            continue
        if new_k in out and clean_text(out[new_k]):
            if notes is not None:
                notes.append(f"duplicate_mapped_key_ignored:{scope}:{k}->{new_k}")
            continue
        out[new_k] = v
        if notes is not None and str(k) != new_k:
            notes.append(f"fuzzy_key:{scope}:{k}->{new_k}")
    return out


def parse_jsonish_dict_field(x: Any) -> Optional[Dict[str, Any]]:
    """Recover fields like generation_plan when Qwen outputs them as a JSON string."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    obj = safe_json_loads(s)
    return obj if isinstance(obj, dict) else None


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


def ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [clean_text(v) for v in x if clean_text(v)]
    if isinstance(x, str):
        s = clean_text(x)
        if not s:
            return []
        if "," in s or ";" in s:
            return [clean_text(p) for p in re.split(r"[,;]", s) if clean_text(p)]
        return [s]
    return [clean_text(str(x))] if clean_text(str(x)) else []


def normalize_title_understanding_fields(tu: Dict[str, Any], notes: Optional[List[str]] = None) -> Dict[str, Any]:
    tu = normalize_keys_by_schema(tu, EXPECTED_TITLE_UNDERSTANDING_KEYS, notes=notes, scope="title_understanding")
    out: Dict[str, Any] = {}
    for key in EXPECTED_TITLE_UNDERSTANDING_KEYS:
        if key in LIST_FIELDS_TITLE:
            out[key] = ensure_list(tu.get(key))
        else:
            out[key] = ensure_str(tu.get(key))
    return out


def normalize_generation_plan_fields(gp: Dict[str, Any], notes: Optional[List[str]] = None) -> Dict[str, Any]:
    gp = normalize_keys_by_schema(gp, EXPECTED_GENERATION_PLAN_KEYS, notes=notes, scope="generation_plan")
    out: Dict[str, Any] = {}
    for key in EXPECTED_GENERATION_PLAN_KEYS:
        if key in LIST_FIELDS_PLAN:
            out[key] = ensure_list(gp.get(key))
        else:
            out[key] = ensure_str(gp.get(key))
    return out


def normalize_model_schema_keys(obj: Dict[str, Any], notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fuzzy schema-key normalization without a hardcoded typo table."""
    norm = normalize_keys_by_schema(copy.deepcopy(obj), EXPECTED_TOP_LEVEL_KEYS, notes=notes, scope="top")

    tu_obj = parse_jsonish_dict_field(norm.get("title_understanding"))
    if isinstance(tu_obj, dict):
        norm["title_understanding"] = normalize_title_understanding_fields(tu_obj, notes=notes)

    gp_obj = parse_jsonish_dict_field(norm.get("generation_plan"))
    if isinstance(gp_obj, dict):
        norm["generation_plan"] = normalize_generation_plan_fields(gp_obj, notes=notes)

    for key in ("sdxl_prompt", "negative_prompt", "ranking_caption"):
        if key in norm:
            norm[key] = ensure_str(norm.get(key))

    return norm

def get_visual_plan(target: Dict[str, Any]) -> Dict[str, Any]:
    gp = target.get("generation_plan")
    if isinstance(gp, dict):
        return gp
    vp = target.get("visual_plan")
    if isinstance(vp, dict):
        return vp
    return {}


def synthesize_prompt_from_plan(title: str, target: Dict[str, Any]) -> str:
    """Synthesize only when Qwen produced a usable plan but failed the prompt field.

    This fallback is non-photorealistic but style-flexible: it uses Qwen's
    plan labels and framing.
    """
    plan = get_visual_plan(target)
    required = as_clean_list(plan.get("required_visual_cues"))
    optional = as_clean_list(plan.get("optional_visual_cues"))
    strategy = clean_text(plan.get("visual_strategy", "")).replace("_", " ")
    framing = clean_text(plan.get("framing", ""))
    abstractness = clean_text(plan.get("abstractness", "")).replace("_", " ")

    cues: List[str] = []
    seen = set()
    for cue in required + optional:
        key = cue.lower()
        if key not in seen:
            cues.append(cue)
            seen.add(key)

    parts = [
        f"clearly synthetic non-photorealistic news/editorial image about {title}",
    ]
    if cues:
        parts.append(", ".join(cues))
    if strategy:
        parts.append(f"{strategy} visual strategy")
    if abstractness:
        parts.append(f"{abstractness} composition")
    if framing:
        parts.append(framing)
    parts.extend([
        "clear central subject",
        "balanced composition",
        "not a realistic event photograph",
        "no readable text",
        "no exact logos",
        "no watermark",
        "avoid exact realistic public-figure likeness",
    ])
    return ", ".join([p for p in parts if clean_text(p)])


def normalize_target(obj: Optional[Dict[str, Any]], title: str, synthesize_missing: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize schema keys and optionally synthesize missing fields as last-resort fallback."""
    notes: List[str] = []
    if not isinstance(obj, dict):
        obj = {}
        notes.append("input_not_parseable_dict")

    schema_notes: List[str] = []
    norm = normalize_model_schema_keys(obj, notes=schema_notes)
    notes.extend(schema_notes)

    if not isinstance(norm.get("title_understanding"), dict):
        old = clean_text(norm.get("title_understanding", ""))
        norm["title_understanding"] = {
            "headline_core": old or title,
            "named_entities": [],
            "entity_roles": [],
            "main_event_or_issue": title,
            "implied_context": "inferred from the title only",
            "relation": "",
            "visual_risks": ["misleading literal scene", "readable text", "exact logos", "realistic face"],
            "what_should_not_be_emphasized": ["unsupported details", "readable text", "exact logos"],
        }
        notes.append("normalized_title_understanding_to_dict")
    else:
        tu = norm["title_understanding"]
        tu["headline_core"] = clean_text(tu.get("headline_core")) or title
        tu["named_entities"] = as_clean_list(tu.get("named_entities"))
        tu["entity_roles"] = as_clean_list(tu.get("entity_roles"))
        tu["main_event_or_issue"] = clean_text(tu.get("main_event_or_issue")) or title
        tu["implied_context"] = clean_text(tu.get("implied_context")) or "inferred from the title only"
        tu["relation"] = clean_text(tu.get("relation"))
        tu["visual_risks"] = as_clean_list(tu.get("visual_risks")) or ["misleading literal scene", "readable text", "exact logos", "realistic face"]
        tu["what_should_not_be_emphasized"] = as_clean_list(tu.get("what_should_not_be_emphasized")) or ["unsupported details", "readable text", "exact logos"]

    if not isinstance(norm.get("generation_plan"), dict):
        norm["generation_plan"] = {
            "image_role": "editorial illustration for a news thumbnail",
            "required_visual_cues": [title],
            "optional_visual_cues": [],
            "forbidden_or_misleading_cues": ["readable text", "exact logos", "realistic public-figure likeness"],
            "visual_strategy": "other",
            "framing": "centered clearly synthetic news/editorial image composition",
            "abstractness": "other",
        }
        notes.append("synthesized_minimal_generation_plan")
    else:
        plan = norm["generation_plan"]
        if not clean_text(plan.get("image_role", "")):
            plan["image_role"] = "editorial illustration for a news thumbnail"
        if not as_clean_list(plan.get("required_visual_cues")):
            plan["required_visual_cues"] = [title]
        if not isinstance(plan.get("optional_visual_cues"), list):
            plan["optional_visual_cues"] = as_clean_list(plan.get("optional_visual_cues"))
        if not as_clean_list(plan.get("forbidden_or_misleading_cues")):
            plan["forbidden_or_misleading_cues"] = ["readable text", "exact logos", "realistic face"]
        if not clean_text(plan.get("visual_strategy", "")):
            plan["visual_strategy"] = "other"
        if not clean_text(plan.get("framing", "")):
            plan["framing"] = "centered clearly synthetic news/editorial image composition"
        if not clean_text(plan.get("abstractness", "")):
            plan["abstractness"] = "other"

    if synthesize_missing and not clean_text(norm.get("sdxl_prompt", "")):
        norm["sdxl_prompt"] = synthesize_prompt_from_plan(title, norm)
        notes.append("synthesized_missing_sdxl_prompt_from_plan")

    if synthesize_missing and not clean_text(norm.get("negative_prompt", "")):
        forbidden = as_clean_list(norm.get("generation_plan", {}).get("forbidden_or_misleading_cues"))
        norm["negative_prompt"] = DEFAULT_NEGATIVE + ((", " + ", ".join(forbidden)) if forbidden else "")
        notes.append("synthesized_missing_negative_prompt")

    if synthesize_missing and not clean_text(norm.get("ranking_caption", "")):
        cues = as_clean_list(norm.get("generation_plan", {}).get("required_visual_cues"))
        cue_text = ", ".join(cues[:3]) if cues else "title-grounded editorial scene"
        norm["ranking_caption"] = f"{title}: {cue_text}"
        notes.append("synthesized_missing_ranking_caption")

    # Keep only expected top-level keys for downstream clarity.
    final = {k: norm.get(k) for k in REQUIRED_TOP_LEVEL_KEYS}
    final = postprocess_target_fields(final)
    return final, notes


def model_output_has_prompt(obj: Optional[Dict[str, Any]]) -> bool:
    """Backward-compatible strict prompt check: Qwen wrote both positive and negative prompts."""
    return model_output_has_positive_prompt(obj) and model_output_has_negative_prompt(obj)


SCHEMA_PLACEHOLDERS = [
    "short interpretation of the title",
    "entity names if any",
    "brief roles if any",
    "main issue or event described by the title",
    "relevant context inferred only from the title and general knowledge",
    "relationship among entities/issues if any",
    "risks such as wrong likeness, misleading literal scene, readable text",
    "things that should not dominate the image",
    "2-5 title-supported visual cues",
    "0-4 optional supporting cues",
    "things that should not appear",
    "brief composition/framing description",
    "title-specific visual subject and cues",
    "title-appropriate clearly synthetic/non-photorealistic editorial style",
    "title-specific misleading detail",
    "forbidden cue",
    "one concise caption describing what the generated image should show",
    "title-specific visual subject and cues from generation_plan",
"title-appropriate clearly synthetic/non-photorealistic editorial style",
"title- and plan-specific misleading detail",
]


def normalized_for_similarity(s: Any) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def too_similar_to_schema_placeholder(text: Any, threshold: float = 0.82) -> bool:
    """Reject copied or near-copied schema placeholder text."""
    t = normalized_for_similarity(text)
    if not t:
        return False
    for ph in SCHEMA_PLACEHOLDERS:
        p = normalized_for_similarity(ph)
        if not p:
            continue
        if p in t:
            return True
        # Compare whole field and also comma-separated fragments.
        if SequenceMatcher(None, t, p).ratio() >= threshold:
            return True
        for part in re.split(r"[,;]", t):
            part = part.strip()
            if len(part) >= 8 and SequenceMatcher(None, part, p).ratio() >= threshold:
                return True
    return False


def looks_like_copied_schema_text(x: Any) -> bool:
    return too_similar_to_schema_placeholder(x)


def has_bad_char_repetition(text: Any, max_run: int = 12) -> bool:
    return re.search(r"(.)\1{" + str(max_run) + r",}", clean_text(text)) is not None


def has_bad_word_repetition(text: Any, max_repeat: int = 4) -> bool:
    words = re.findall(r"\b\w+\b", clean_text(text).lower())
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= max_repeat:
                return True
        else:
            run = 1
    return False


def field_is_usable_text(x: Any, min_chars: int) -> bool:
    s = clean_text(x)
    if len(s) < min_chars:
        return False
    if has_bad_char_repetition(s) or has_bad_word_repetition(s) or has_bad_token_repetition(s):
        return False
    if too_similar_to_schema_placeholder(s):
        return False
    return True


def model_output_has_positive_prompt(obj: Optional[Dict[str, Any]]) -> bool:
    """True when Qwen wrote a usable positive SDXL prompt."""
    if not isinstance(obj, dict):
        return False
    norm = normalize_model_schema_keys(obj)
    return field_is_usable_text(norm.get("sdxl_prompt", ""), MIN_SDXL_PROMPT_CHARS)


def model_output_has_negative_prompt(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    norm = normalize_model_schema_keys(obj)
    return field_is_usable_text(norm.get("negative_prompt", ""), MIN_NEGATIVE_PROMPT_CHARS)

def model_output_has_title_understanding(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    norm = normalize_model_schema_keys(obj)
    tu = norm.get("title_understanding")
    if not isinstance(tu, dict):
        return False
    if looks_like_copied_schema_text(json.dumps(tu, ensure_ascii=False)):
        return False
    return bool(clean_text(tu.get("headline_core", "")) or clean_text(tu.get("main_event_or_issue", "")))


def model_output_has_generation_plan(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    norm = normalize_model_schema_keys(obj)
    gp = norm.get("generation_plan")
    if not isinstance(gp, dict):
        return False
    if looks_like_copied_schema_text(json.dumps(gp, ensure_ascii=False)):
        return False
    return (
        bool(as_clean_list(gp.get("required_visual_cues")))
        and bool(clean_text(gp.get("visual_strategy", "")))
        and bool(clean_text(gp.get("framing", "")))
        and bool(clean_text(gp.get("abstractness", "")))
    )

def model_output_has_plan_and_prompt(obj: Optional[Dict[str, Any]]) -> bool:
    """Return True when Qwen wrote a usable understanding, visual plan, and positive prompt.

    This is a stricter diagnostic than the main model_success metric. The main
    metric counts model-written positive SDXL prompts; this diagnostic also
    requires title_understanding and generation_plan. Negative prompt quality is
    reported separately by model_output_has_negative_prompt().
    """
    return (
        model_output_has_positive_prompt(obj)
        and model_output_has_title_understanding(obj)
        and model_output_has_generation_plan(obj)
    )


def has_usable_sdxl_prompt(target: Any) -> bool:
    return (
        isinstance(target, dict)
        and bool(clean_text(target.get("sdxl_prompt", "")))
        and bool(clean_text(target.get("negative_prompt", "")))
    )



def canonicalize_and_validate(
    obj: Optional[Dict[str, Any]],
    title: str
) -> Tuple[Dict[str, Any], bool, List[str], List[str]]:
    """Self-contained inference validation.

    We intentionally do not import external teacher-data validators here,
    because inference should not depend on another file that may use different
    schema defaults or style assumptions. Detailed usability checks are handled
    before fallback by model_output_has_* functions.
    """
    if obj is None:
        return {}, False, ["json_parse_failed"], []

    if not isinstance(obj, dict):
        return {}, False, ["target_not_dict"], []

    errors: List[str] = []
    notes: List[str] = ["self_contained_inference_validation"]

    if not clean_text(obj.get("sdxl_prompt", "")):
        errors.append("missing_sdxl_prompt")
    if not clean_text(obj.get("negative_prompt", "")):
        errors.append("missing_negative_prompt")

    ok = len(errors) == 0
    return obj, ok, errors, notes


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quant,
        torch_dtype=None if quant is not None else dtype,
    )

    if clean_text(getattr(args, "adapter", "")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base,
            args.adapter,
            local_files_only=args.local_files_only,
        )
    else:
        model = base

    model.eval()
    return model, tokenizer


def model_input_device(model):
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


def _generate_text(tokenizer, model, current_messages: List[Dict[str, str]], args) -> str:
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

    out_ids = model.generate(**gen_kwargs)
    gen = out_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def make_model_flags(obj: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Model-produced field diagnostics before fallback."""
    return {
        "model_written_sdxl_prompt_ok": model_output_has_positive_prompt(obj),
        "model_written_negative_prompt_ok": model_output_has_negative_prompt(obj),
        "model_title_understanding_ok": model_output_has_title_understanding(obj),
        "model_generation_plan_ok": model_output_has_generation_plan(obj),
        "model_plan_and_prompt_ok": model_output_has_plan_and_prompt(obj),
    }


def generate_one(tokenizer, model, title: str, args) -> Tuple[Dict[str, Any], str, List[str], bool, Dict[str, bool]]:
    """
    Generate one target.

    Main success criterion for fair comparison:
      Qwen itself wrote a usable positive sdxl_prompt before fallback.

    If Qwen omits/invalidates negative_prompt, we fill DEFAULT_NEGATIVE, the
    same negative prompt used by the TemplateSDXL baseline. This keeps SDXL
    generation complete and does not penalize the model for the negative field.

    The stricter visual-planning fields are still recorded separately as
    "success within success" metrics.
    """
    current_messages = build_messages(title)
    last_text = ""
    last_obj: Optional[Dict[str, Any]] = None
    last_parse_ok = False
    last_repair_notes: List[str] = []

    with torch.inference_mode():
        for attempt in range(args.retries + 1):
            text = _generate_text(tokenizer, model, current_messages, args)
            last_text = text
            obj, repair_notes = safe_json_loads_with_notes(text)
            last_obj = obj
            last_parse_ok = obj is not None
            last_repair_notes = repair_notes
            flags = make_model_flags(obj)

            if flags["model_written_sdxl_prompt_ok"]:
                target, notes = normalize_target(obj, title, synthesize_missing=False)
                notes = list(repair_notes) + list(notes)

                # Positive prompt is model-written. For missing/bad negative prompt,
                # use the same TemplateSDXL baseline negative prompt for fairness.
                if not flags["model_written_negative_prompt_ok"]:
                    target["negative_prompt"] = DEFAULT_NEGATIVE
                    notes.append("filled_missing_or_invalid_negative_prompt_with_baseline_template_negative")
                elif has_repeated_comma_phrases((obj or {}).get("negative_prompt", "")):
                    notes.append("deduped_repeated_negative_prompt_phrases")

                # Ranking caption is not part of model success; fill only for downstream compatibility.
                if not clean_text(target.get("ranking_caption", "")):
                    target["ranking_caption"] = f"{title}: title-grounded editorial illustration"
                    notes.append("synthesized_missing_ranking_caption_only")

                notes.append("accepted_model_written_sdxl_prompt")
                return target, text, notes, True, flags

            if attempt < args.retries:
                reason = "not valid JSON" if obj is None else "valid JSON but missing a usable positive sdxl_prompt or copied schema text"
                current_messages.append({"role": "assistant", "content": text[:4000]})
                current_messages.append({
                    "role": "user",
                    "content": (
                        f"The previous answer was {reason}. Rewrite it as exactly one valid JSON object. "
                        "Use exactly these top-level keys: title_understanding, generation_plan, "
                        "sdxl_prompt, negative_prompt, ranking_caption. "
                        "The sdxl_prompt must be a non-empty Stable Diffusion XL text-to-image prompt describing "
                        "visual content, a title-appropriate clearly synthetic or non-photorealistic editorial/news image style, composition, and title-supported cues. "
                        "The negative_prompt should be a concise string listing visual elements to avoid; "
                        "if uncertain, still provide a simple avoid-list. "
                        "Do not repeat the same phrase. Do not use markdown. Do not use alternative key names."
                    ),
                })

    # Last-resort fallback after all retries. This guarantees coverage but is tracked separately.
    # Fallback is transparent and has two subtypes:
    #   1) plan_based_fallback: last Qwen JSON had a usable generation_plan, so synthesize
    #      only the missing SDXL prompt from that plan.
    #   2) title_template_fallback: no usable Qwen plan, so use the deterministic
    #      title-only template fallback aligned with the TemplateSDXL baseline.
    if isinstance(last_obj, dict) and model_output_has_generation_plan(last_obj):
        target, notes = make_plan_based_fallback_target(title, last_obj)
        notes = list(last_repair_notes) + list(notes)
        notes.append("last_resort_fallback_after_retries")
        notes.append("plan_based_fallback_from_qwen_generation_plan")
    else:
        target = make_title_template_fallback_target(title)
        notes = list(last_repair_notes) + ["last_resort_fallback_after_retries", "title_template_fallback"]
    return target, last_text, notes, last_parse_ok, make_model_flags(last_obj)


def load_done_keys(out_path: str) -> set:
    """Load completed (article_id, image_id) keys for safe resume."""
    done = set()
    p = Path(out_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                aid = clean_text(r.get("article_id", ""))
                iid = clean_text(r.get("image_id", ""))
                if aid:
                    done.add((aid, iid))
            except Exception:
                continue
    return done


def write_metadata(args) -> None:
    meta = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "method": "Unified title-only Qwen3B visual planner inference; --adapter toggles LoRA vs zero-shot",
        "base_model": args.base_model,
        "adapter": args.adapter or None,
        "mode": "lora" if args.adapter else "zeroshot",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "retries": args.retries,
        "load_in_4bit": bool(args.load_in_4bit),
        "bf16": bool(args.bf16),
        "local_files_only": bool(args.local_files_only),
        "coverage_policy": "Every row receives an SDXL prompt. If Qwen fails to produce a usable positive SDXL prompt after retries, fallback is used: plan_based_fallback when the last Qwen JSON contains a usable generation_plan; otherwise title_template_fallback aligned with the TemplateSDXL baseline.",
        "success_definition": "model_success means Qwen produced a usable positive sdxl_prompt before fallback. Missing/invalid negative_prompt is filled with the same DEFAULT_NEGATIVE used by the TemplateSDXL baseline and is not counted as model failure.",
        "reported_metrics": "Main image-generation metrics are model_success_rate, fallback_rate, and default_negative_filled_rate_within_success. Diagnostic reasoning metrics include title_understanding_success_rate, generation_plan_success_rate, plan_and_prompt_success_rate, and negative_prompt_success_rate. repair_note_counts records JSON repair and fuzzy key-mapping events.",
        "default_negative": DEFAULT_NEGATIVE,
    }
    Path(args.out + ".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def rate(num: int, den: int) -> float:
    return round((num / den), 6) if den else 0.0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True, help="CSV or JSONL with article_title")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="", help="Optional LoRA adapter directory. Omit for zero-shot.")
    ap.add_argument("--title-col", default="article_title")
    ap.add_argument("--article-id-col", default="article_id")
    ap.add_argument("--image-col", default="image_id")
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=5)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all rows")
    ap.add_argument("--resume", action="store_true", help="Append and skip existing (article_id, image_id) rows")
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false", help="Do not save raw model text")
    ap.set_defaults(save_raw=True)
    args = ap.parse_args()

    model, tokenizer = load_model(args)
    out_parent = Path(args.out).parent
    if out_parent != Path("."):
        out_parent.mkdir(parents=True, exist_ok=True)
    write_metadata(args)

    done_keys = load_done_keys(args.out) if args.resume else set()
    open_mode = "a" if args.resume and Path(args.out).exists() else "w"
    mode = "lora" if args.adapter else "zeroshot"
    print(f"Mode: {mode}; adapter={args.adapter or None}; done_keys={len(done_keys)}", flush=True)

    n = parse_ok_count = usable_count = model_prompt_count = fallback_count = skipped = 0
    model_title_understanding_count = 0
    model_generation_plan_count = 0
    model_plan_and_prompt_count = 0
    model_negative_prompt_count = 0
    default_negative_filled_count = 0
    recovery_counts: Counter = Counter()
    repair_note_counts: Counter = Counter()
    with open(args.out, open_mode, encoding="utf-8") as f:
        for row in read_titles(args):
            if args.limit and n >= args.limit:
                break
            title = row["article_title"]
            aid = clean_text(row.get("article_id", ""))
            iid = clean_text(row.get("image_id", ""))
            if not title:
                continue
            if aid and (aid, iid) in done_keys:
                skipped += 1
                continue

            target, raw_text, gen_notes, raw_parse_ok, model_flags = generate_one(tokenizer, model, title, args)

            validated_target, ok, errors, val_notes = canonicalize_and_validate(target, title)
            # If the external validator/canonicalizer unexpectedly drops the prompt, keep our robust target.
            if has_usable_sdxl_prompt(validated_target):
                final_target = validated_target
            else:
                final_target = target
                val_notes = list(val_notes) + ["kept_prevalidation_target_to_preserve_usable_sdxl_prompt"]

            notes = list(gen_notes) + list(val_notes)
            has_prompt = has_usable_sdxl_prompt(final_target)
            used_fallback = any("fallback" in str(x) or "synthesized_missing_sdxl_prompt" in str(x) for x in notes)

            model_prompt_success = bool(model_flags.get("model_written_sdxl_prompt_ok")) and has_prompt and not used_fallback
            if model_prompt_success:
                recovery_type = "model_written_sdxl_prompt"
            elif used_fallback and has_prompt:
                if any("plan_based_fallback" in str(x) for x in notes):
                    recovery_type = "plan_based_fallback"
                else:
                    recovery_type = "title_template_fallback"
            else:
                recovery_type = "failed_no_prompt"

            n += 1
            parse_ok_count += int(raw_parse_ok)
            usable_count += int(has_prompt)
            model_prompt_count += int(model_prompt_success)
            model_title_understanding_count += int(bool(model_flags.get("model_title_understanding_ok")))
            model_generation_plan_count += int(bool(model_flags.get("model_generation_plan_ok")))
            model_plan_and_prompt_count += int(bool(model_flags.get("model_plan_and_prompt_ok")))
            model_negative_prompt_count += int(bool(model_flags.get("model_written_negative_prompt_ok")))
            default_negative_filled_count += int(model_prompt_success and not model_flags.get("model_written_negative_prompt_ok"))
            fallback_count += int(recovery_type in {"plan_based_fallback", "title_template_fallback"})
            recovery_counts[recovery_type] += 1
            for note in notes:
                note_s = str(note)
                if note_s.startswith(("json_repair:", "json_extract:", "json_parse:", "fuzzy_key:", "unmapped_key:", "duplicate_mapped_key_ignored:")):
                    repair_note_counts[note_s] += 1

            out_row = {
                **row,
                "student_target": final_target if has_prompt else None,
                "parse_ok": raw_parse_ok,
                "model_written_sdxl_prompt_ok": recovery_type == "model_written_sdxl_prompt",
                "model_title_understanding_ok": bool(model_flags.get("model_title_understanding_ok")),
                "model_generation_plan_ok": bool(model_flags.get("model_generation_plan_ok")),
                "model_plan_and_prompt_ok": bool(model_flags.get("model_plan_and_prompt_ok")),
                "model_written_negative_prompt_ok": bool(model_flags.get("model_written_negative_prompt_ok")),
                "negative_prompt_filled_from_baseline_template": recovery_type == "model_written_sdxl_prompt" and not bool(model_flags.get("model_written_negative_prompt_ok")),
                "has_usable_sdxl_prompt": has_prompt,
                "used_fallback": recovery_type in {"plan_based_fallback", "title_template_fallback"},
                "fallback_type": recovery_type if recovery_type in {"plan_based_fallback", "title_template_fallback"} else None,
                "recovery_type": recovery_type,
                "is_valid_final": ok,
                "validation_errors_final": errors,
                "normalization_notes_final": notes,
                "inference_mode": mode,
                "adapter": args.adapter or None,
            }
            if has_prompt:
                out_row["sdxl_prompts"] = [{
                    "variant": "chosen",
                    "sdxl_prompt": final_target["sdxl_prompt"],
                    "negative_prompt": final_target.get("negative_prompt", DEFAULT_NEGATIVE),
                }]
                out_row["ranking_caption"] = clean_text(final_target.get("ranking_caption", row.get("article_title", "")))
            if args.save_raw:
                out_row["student_raw_text"] = raw_text

            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

            if n % 10 == 0:
                print(
                    f"Processed {n}; skipped={skipped}; parse_ok={parse_ok_count}; "
                    f"usable_prompt={usable_count}; model_success={model_prompt_count}; fallback={fallback_count}; "
                    f"recovery_type={dict(recovery_counts)}",
                    flush=True,
                )

    summary = {
        "script": Path(__file__).name,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "adapter": args.adapter or None,
        "rows_processed": n,
        "rows_skipped_resume": skipped,
        "parse_ok": parse_ok_count,
        "has_usable_sdxl_prompt": usable_count,
        "model_written_sdxl_prompt": model_prompt_count,
        "model_title_understanding": model_title_understanding_count,
        "model_generation_plan": model_generation_plan_count,
        "model_plan_and_prompt": model_plan_and_prompt_count,
        "model_written_negative_prompt": model_negative_prompt_count,
        "fallback_total": fallback_count,
        "plan_based_fallback": recovery_counts.get("plan_based_fallback", 0),
        "title_template_fallback": recovery_counts.get("title_template_fallback", 0),
        "negative_prompt_filled_from_baseline_template": default_negative_filled_count,
        "failed_no_prompt": recovery_counts.get("failed_no_prompt", 0),
        "recovery_type_counts": dict(recovery_counts),
        "repair_note_counts": dict(repair_note_counts),
        "model_success_rate": rate(model_prompt_count, n),
        "title_understanding_success_rate": rate(model_title_understanding_count, n),
        "generation_plan_success_rate": rate(model_generation_plan_count, n),
        "plan_and_prompt_success_rate": rate(model_plan_and_prompt_count, n),
        "negative_prompt_success_rate": rate(model_negative_prompt_count, n),
        "fallback_rate": rate(fallback_count, n),
        "default_negative_filled_rate_within_success": rate(default_negative_filled_count, model_prompt_count),
    }
    Path(args.out + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done. mode={mode}; rows={n}; skipped={skipped}; parse_ok={parse_ok_count}; "
        f"usable_prompt={usable_count}; model_success={model_prompt_count}; fallback={fallback_count}; "
        f"recovery_type={dict(recovery_counts)}; wrote={args.out}"
    )


if __name__ == "__main__":
    main()
