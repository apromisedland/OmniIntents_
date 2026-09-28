"""Deterministic history reference, not the paper's unavailable learned model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..errors import ValidationError


@dataclass
class HistoryEntry:
    sample_id: str
    sequence_index: int
    timestamp: datetime
    location: str | None
    activity: str | None
    specific_label: str


class ContextualMemoryTracker:
    def __init__(self, max_entries: int = 5):
        if type(max_entries) is not int or not 0 <= max_entries <= 5:
            raise ValidationError("max_entries must be in 0..5")
        self.max_entries = max_entries
        self.sessions: dict[str, list[HistoryEntry]] = {}

    def add_entry(self, session_id: str, entry: HistoryEntry) -> None:
        if not self.max_entries:
            return
        entries = self.sessions.setdefault(session_id, [])
        if entries and (
            entry.sequence_index <= entries[-1].sequence_index or entry.timestamp < entries[-1].timestamp
        ):
            raise ValidationError("History updates must be strictly ordered and cannot repeat a sample")
        if any(previous.sample_id == entry.sample_id for previous in entries):
            raise ValidationError("Duplicate history sample")
        entries.append(entry)
        self.sessions[session_id] = entries[-self.max_entries:]

    def previous(self, session_id: str, sequence_index: int, timestamp: datetime, rounds: int) -> list[dict[str, Any]]:
        if type(rounds) is not int or not 0 <= rounds <= self.max_entries:
            raise ValidationError("Invalid history window")
        if rounds == 0:
            return []
        entries = [
            entry for entry in self.sessions.get(session_id, [])
            if entry.sequence_index < sequence_index and entry.timestamp <= timestamp
        ][-rounds:]
        return [
            {"sequence_index": entry.sequence_index, "location": entry.location,
             "activity": entry.activity, "specific_label": entry.specific_label}
            for entry in entries
        ]

    @staticmethod
    def summarize(history: list[dict[str, Any]]) -> str:
        if not history:
            return ""
        parts = [
            f"Round {entry['sequence_index']}: location={entry['location'] or 'unknown'}; "
            f"activity={entry['activity'] or 'unknown'}; intent={entry['specific_label']}."
            for entry in history
        ]
        return "Deterministic history summary. " + " ".join(parts)

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self.sessions.clear()
        else:
            self.sessions.pop(session_id, None)
