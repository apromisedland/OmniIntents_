from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Union


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", flags=re.DOTALL)
_FIRST_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_FIRST_JSON_ARR_RE = re.compile(r"\[.*\]", flags=re.DOTALL)


def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _try_load(candidate: str) -> Optional[Union[Dict[str, Any], Any]]:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Remove trailing commas: { "a": 1, } or [1,2,]
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            return None


def extract_first_json(text: str) -> Optional[Any]:
    """Best-effort extractor for the first JSON object/array from free-form text."""
    if not text:
        return None

    m = _FENCED_JSON_RE.search(text)
    if m:
        parsed = _try_load(m.group(1))
        if parsed is not None:
            return parsed

    m2 = _FIRST_JSON_OBJ_RE.search(text)
    if m2:
        parsed = _try_load(m2.group(0))
        if parsed is not None:
            return parsed

    m3 = _FIRST_JSON_ARR_RE.search(text)
    if m3:
        parsed = _try_load(m3.group(0))
        if parsed is not None:
            return parsed

    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    parsed = extract_first_json(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
