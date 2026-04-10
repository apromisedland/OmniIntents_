from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..types import AgentType, Capability


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "yes", "y", "1"}:
            return True
        if t in {"false", "no", "n", "0"}:
            return False
    return default


def validate_intent_json(data: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["Intent JSON must be an object."]
    if not isinstance(data.get("label"), str) or not data["label"].strip():
        errors.append("Missing/invalid 'label'.")
    if not isinstance(data.get("description"), str):
        errors.append("Missing/invalid 'description'.")
    conf = data.get("confidence", 0.5)
    try:
        float(conf)
    except Exception:
        errors.append("Invalid 'confidence' (must be number).")
    # entities optional
    ent = data.get("entities", {})
    if ent is not None and not isinstance(ent, dict):
        errors.append("'entities' must be an object if provided.")
    # requires_* optional
    return len(errors) == 0, errors


def validate_task_plan_json(data: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["Task plan JSON must be an object."]
    if not isinstance(data.get("goal"), str) or not data["goal"].strip():
        errors.append("Missing/invalid 'goal'.")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("Missing/invalid 'steps' (must be non-empty list).")
        return False, errors
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            errors.append(f"Step {i} must be an object.")
            continue
        if "step_id" not in s:
            errors.append(f"Step {i} missing 'step_id'.")
        if not isinstance(s.get("instruction"), str) or not s["instruction"].strip():
            errors.append(f"Step {i} missing/invalid 'instruction'.")
        caps = s.get("required_capabilities", [])
        if caps is not None and not isinstance(caps, list):
            errors.append(f"Step {i} 'required_capabilities' must be list.")
    return len(errors) == 0, errors


def validate_agent_json(data: Any, candidates: Optional[List[AgentType]] = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["Agent JSON must be an object."]
    if not isinstance(data.get("agent_type"), str) or not data["agent_type"].strip():
        errors.append("Missing/invalid 'agent_type'.")
    else:
        try:
            agent = AgentType(data["agent_type"])
            if candidates is not None and agent not in candidates:
                errors.append(f"'agent_type' must be one of candidates: {[c.value for c in candidates]}.")
        except Exception:
            errors.append(f"Unknown agent_type: {data.get('agent_type')!r}.")
    if not isinstance(data.get("rationale", ""), str):
        errors.append("Invalid 'rationale' (must be string).")
    return len(errors) == 0, errors


def parse_capabilities(raw_list: Any) -> List[Capability]:
    if not isinstance(raw_list, list):
        return []
    out: List[Capability] = []
    for c in raw_list:
        try:
            out.append(Capability(str(c)))
        except Exception:
            continue
    return out
