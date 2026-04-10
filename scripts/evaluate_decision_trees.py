from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from omniintents.agentgpt.decision_tree import ExplicitDecisionTree, ImplicitDecisionTree
from omniintents.agentgpt.eval import confusion_matrix, pretty_confusion_matrix
from omniintents.types import AgentType, Capability, TaskPlan, TaskStep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate implicit vs explicit decision trees on a dataset.")
    p.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "omniintents" / "data" / "agent_eval_tasks.json"),
        help="Path to JSON dataset with tasks.",
    )
    return p.parse_args()


def task_plan_from_caps(task: str, caps: List[str]) -> TaskPlan:
    required_caps: List[Capability] = []
    for c in caps:
        try:
            required_caps.append(Capability(c))
        except Exception:
            continue
    return TaskPlan(goal=task, steps=[TaskStep(step_id=1, instruction=task, required_capabilities=required_caps)])


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    implicit = ImplicitDecisionTree()
    explicit = ExplicitDecisionTree()

    labels = list(AgentType)

    y_true: List[AgentType] = []
    y_pred_impl: List[AgentType] = []
    y_pred_expl: List[AgentType] = []

    for row in data:
        task = row.get("task", "")
        caps = row.get("required_capabilities", [])
        gt = AgentType(row.get("ground_truth_agent"))
        plan = task_plan_from_caps(task, caps)

        y_true.append(gt)
        y_pred_impl.append(implicit.recommend_candidates(plan)[0])
        y_pred_expl.append(explicit.recommend(plan))

    m_impl = confusion_matrix(y_true, y_pred_impl, labels)
    m_expl = confusion_matrix(y_true, y_pred_expl, labels)

    print("=== Implicit Decision Tree Confusion Matrix ===")
    print(pretty_confusion_matrix(m_impl, labels))
    print("\n=== Explicit Decision Tree Confusion Matrix ===")
    print(pretty_confusion_matrix(m_expl, labels))


if __name__ == "__main__":
    main()
