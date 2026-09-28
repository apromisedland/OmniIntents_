"""Strict JSON decoding with bounded recovery from surrounding model text."""

from __future__ import annotations

import json
from typing import Any

from ..errors import ValidationError


def dumps_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _constant(value: str) -> None:
    raise ValidationError(f"Non-finite JSON constant: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_constant, object_pairs_hook=_pairs)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValidationError("Invalid JSON") from error


def extract_first_json(text: str) -> Any:
    if not isinstance(text, str):
        raise ValidationError("Model response must be text")
    decoder = json.JSONDecoder(parse_constant=_constant, object_pairs_hook=_pairs)
    for position, character in enumerate(text):
        if character in "{[":
            try:
                value, _ = decoder.raw_decode(text[position:])
                return value
            except json.JSONDecodeError:
                continue
    return None


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    value = extract_first_json(text)
    return value if isinstance(value, dict) else None


def require_json_object(text: str) -> dict[str, Any]:
    value = extract_first_json_object(text)
    if value is None:
        raise ValidationError("Model response contains no valid JSON object")
    return value


def truncate(text: str, max_chars: int) -> str:
    return text[:max(0, max_chars)]
