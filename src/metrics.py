from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    label_names: Sequence[str],
) -> Dict[str, float]:
    """Compute the task's default classification metrics."""
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                pred_labels,
                labels=list(label_names),
                average="macro",
                zero_division=0,
            )
        ),
    }
