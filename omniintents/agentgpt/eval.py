from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..types import AgentType


def confusion_matrix(
    y_true: Sequence[AgentType],
    y_pred: Sequence[AgentType],
    labels: Sequence[AgentType],
) -> List[List[int]]:
    """Compute confusion matrix counts."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        m[idx[t]][idx[p]] += 1
    return m


def pretty_confusion_matrix(matrix: List[List[int]], labels: Sequence[AgentType]) -> str:
    """Render confusion matrix as a fixed-width table string."""
    names = [l.value for l in labels]
    width = max(len(n) for n in names + ["true/pred"]) + 2

    def fmt(s: str) -> str:
        return s.ljust(width)

    header = fmt("true/pred") + "".join(fmt(n) for n in names)
    rows = [header]
    for i, tname in enumerate(names):
        row = fmt(tname) + "".join(fmt(str(matrix[i][j])) for j in range(len(names)))
        rows.append(row)
    return "\n".join(rows)
