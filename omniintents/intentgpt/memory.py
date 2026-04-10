from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class HistoryEntry:
    timestamp: datetime
    location: str
    activity: str
    intent_label: str
    utterance: str = ""


class _TorchSeq2SeqSummarizer(nn.Module):  # type: ignore[misc]
    """Seq2seq scaffolding (LSTM) to mirror the paper (not trained by default)."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, vocab_size: int = 2000):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):  # pragma: no cover
        enc_out, (h, c) = self.encoder(x)
        dec_in = h[-1].unsqueeze(1).repeat(1, 8, 1)
        dec_out, _ = self.decoder(dec_in)
        logits = self.output(dec_out)
        return logits


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # Extract ASCII words/numbers
    tokens = re.findall(r"[a-z0-9]+", text)
    # Add CJK characters as tokens (粗略处理中文)
    tokens += re.findall(r"[\u4e00-\u9fff]", text)
    return tokens


def _tf(tokens: List[str]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for t in tokens:
        d[t] = d.get(t, 0.0) + 1.0
    return d


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class ContextualMemoryTracker:
    """Contextual Memory Tracker (paper 6.1.2).

    - Stores up to `max_entries` recent history entries
    - Produces a summary string for prompts
    - Provides a simple retrieval mechanism for intent inference (top-k similar past tasks)

    Default summarization is heuristic for reliability.
    Torch seq2seq scaffolding is included (optional, not trained by default).
    """

    max_entries: int = 5
    retrieval_top_k: int = 3
    entries: List[HistoryEntry] = field(default_factory=list)

    use_torch_summarizer: bool = False
    torch_model_path: Optional[str] = None

    def __post_init__(self) -> None:
        self._torch_model = None
        if self.use_torch_summarizer and torch is not None and nn is not None:
            self._torch_model = _TorchSeq2SeqSummarizer()
            if self.torch_model_path:
                try:
                    state = torch.load(self.torch_model_path, map_location="cpu")
                    self._torch_model.load_state_dict(state)
                    self._torch_model.eval()
                except Exception:
                    self._torch_model = None

    def add_entry(self, *, timestamp: datetime, location: str, activity: str, intent_label: str, utterance: str = "") -> None:
        self.entries.append(
            HistoryEntry(
                timestamp=timestamp,
                location=location or "unknown",
                activity=activity or "unknown",
                intent_label=intent_label or "unknown",
                utterance=utterance or "",
            )
        )
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def summarize(self) -> str:
        if not self.entries:
            return ""

        # Torch summarizer not meaningful without training; fallback to heuristic.
        return self._heuristic_summary()

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-k similar history entries with scores."""
        if not self.entries:
            return []
        q_tokens = _tokenize(query)
        q_tf = _tf(q_tokens)

        scored: List[Tuple[float, HistoryEntry]] = []
        for e in self.entries:
            text = f"{e.location} {e.activity} {e.intent_label} {e.utterance}"
            tf = _tf(_tokenize(text))
            score = _cosine(q_tf, tf)
            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, e in scored[: self.retrieval_top_k]:
            out.append(
                {
                    "timestamp": e.timestamp.isoformat(),
                    "location": e.location,
                    "activity": e.activity,
                    "intent_label": e.intent_label,
                    "utterance": e.utterance,
                    "similarity": float(score),
                }
            )
        return out

    def _heuristic_summary(self) -> str:
        last = self.entries[-3:] if len(self.entries) >= 3 else self.entries
        parts = []
        for i, e in enumerate(last, start=1):
            act = e.activity or "an unspecified activity"
            parts.append(f"Task {i}: {act}")
        intents = [e.intent_label for e in last if e.intent_label]
        inferred_intent = intents[-1] if intents else "unspecified"
        return (
            f"In the past {len(last)} tasks, the user handled "
            + ", ".join(parts)
            + f", intending to {inferred_intent}."
        )
