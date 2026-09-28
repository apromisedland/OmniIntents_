"""Validate semantic model contracts before constructing runtime objects."""

from __future__ import annotations

from typing import Any

from ..errors import ValidationError
from ..types import AgentType, Capability, Intent, TaskPlan, TaskStep


def require_fields(data: Any, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("Expected a JSON object")
    if required - data.keys():
        raise ValidationError(f"Missing fields: {sorted(required - data.keys())}")
    if data.keys() - required - (optional or set()):
        raise ValidationError(f"Unexpected fields: {sorted(data.keys() - required - (optional or set()))}")
    return data


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def parse_intents(data: Any) -> tuple[list[Intent], str, str]:
    require_fields(data, {"candidates", "complementarity", "ambiguity"}, {"evidence_summary"})
    if not isinstance(data.get("evidence_summary", ""), str):
        raise ValidationError("evidence_summary must be a string")
    if not isinstance(data["complementarity"], str) or data["complementarity"] not in {"complementary", "non_complementary", "unknown"}:
        raise ValidationError("Invalid complementarity")
    if not isinstance(data["ambiguity"], str) or data["ambiguity"] not in {"ambiguous", "unambiguous", "unknown"}:
        raise ValidationError("Invalid ambiguity")
    if not isinstance(data["candidates"], list) or len(data["candidates"]) > 3:
        raise ValidationError("Expected zero to three intent candidates")
    intents = []
    descriptions = set()
    for value in data["candidates"]:
        require_fields(value, {"specific_label", "description", "confidence"}, {"general_label"})
        intent = Intent(**value)
        if intent.description.casefold() in descriptions:
            raise ValidationError("Duplicate intent candidate")
        descriptions.add(intent.description.casefold())
        intents.append(intent)
    if len(intents) > 1 and data["ambiguity"] == "unambiguous":
        raise ValidationError("Multiple candidates cannot be unambiguous")
    intents.sort(key=lambda intent: -intent.confidence)
    return intents, data["complementarity"], data["ambiguity"]


def parse_capabilities(values: Any) -> list[Capability]:
    if not isinstance(values, list) or not values:
        raise ValidationError("required_capabilities must be a non-empty list")
    try:
        return list(dict.fromkeys(Capability(value) for value in values))
    except (ValueError, TypeError) as error:
        raise ValidationError("Unknown capability") from error


def parse_task_plan(data: Any) -> TaskPlan:
    require_fields(data, {"goal", "steps"})
    goal = nonempty_string(data["goal"], "goal")
    if not isinstance(data["steps"], list) or not data["steps"]:
        raise ValidationError("Task plan requires non-empty steps")
    steps = []
    for position, value in enumerate(data["steps"], 1):
        require_fields(value, {"step_id", "instruction", "required_capabilities"})
        if type(value["step_id"]) is not int or value["step_id"] != position:
            raise ValidationError("step_id must be consecutive integers starting at one")
        steps.append(TaskStep(
            position, nonempty_string(value["instruction"], "instruction"),
            parse_capabilities(value["required_capabilities"]),
        ))
    return TaskPlan(goal, steps)


def parse_agent(data: Any) -> tuple[AgentType, str]:
    require_fields(data, {"agent_type", "rationale"})
    try:
        agent = AgentType(data["agent_type"])
    except (ValueError, TypeError) as error:
        raise ValidationError("Unknown agent type") from error
    return agent, nonempty_string(data["rationale"], "rationale")


def parse_flags(data: Any) -> dict[str, bool]:
    names = {"speech_expression", "visual_expression", "physical_interaction", "creativity"}
    require_fields(data, names)
    if any(type(value) is not bool for value in data.values()):
        raise ValidationError("Explicit decision flags must be JSON booleans")
    return data
