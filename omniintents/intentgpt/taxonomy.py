from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class IntentDefinition:
    label: str
    description: str
    requires_visual: bool
    requires_audio: bool
    requires_physical: bool
    # Optional keywords for heuristic matching
    keywords: List[str]


class IntentTaxonomy:
    """Canonical intent taxonomy used by IntentGPT.

    The excerpt (and typical OmniIntents framing) includes at least:
    - Request Search
    - Request Collaborator
    - Request Companion
    - Request Physical Help
    - Unknown
    """

    def __init__(self) -> None:
        self._intents: List[IntentDefinition] = [
            IntentDefinition(
                label="Request Search",
                description="User wants to query / search information about something (often a referenced object).",
                requires_visual=True,
                requires_audio=True,
                requires_physical=False,
                keywords=["query", "search", "look up", "information", "info", "查", "查询", "信息"],
            ),
            IntentDefinition(
                label="Request Collaborator",
                description="User wants assistance in the environment (navigation, locating, collaborative manipulation).",
                requires_visual=True,
                requires_audio=True,
                requires_physical=False,
                keywords=["navigate", "go to", "find", "walk", "move", "导航", "带我", "去", "找", "移动"],
            ),
            IntentDefinition(
                label="Request Physical Help",
                description="User wants the system/robot to execute a physical action.",
                requires_visual=True,
                requires_audio=True,
                requires_physical=True,
                keywords=["cut", "slice", "chop", "grab", "pour", "切", "拿", "抓", "倒", "搬"],
            ),
            IntentDefinition(
                label="Request Companion",
                description="User wants social interaction / companionship.",
                requires_visual=False,
                requires_audio=True,
                requires_physical=False,
                keywords=["cheers", "toast", "chat", "talk", "陪我", "聊天", "开心"],
            ),
            IntentDefinition(
                label="Unknown",
                description="System cannot confidently infer the intent; should ask clarification.",
                requires_visual=False,
                requires_audio=False,
                requires_physical=False,
                keywords=[],
            ),
        ]

    @property
    def intents(self) -> List[IntentDefinition]:
        return list(self._intents)

    def to_prompt_list(self) -> str:
        lines = []
        for it in self._intents:
            if it.label == "Unknown":
                continue
            lines.append(f"- {it.label}: {it.description}")
        return "\n".join(lines)

    def get(self, label: str) -> Optional[IntentDefinition]:
        for it in self._intents:
            if it.label == label:
                return it
        return None
