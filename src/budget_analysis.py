from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class BudgetEstimate:
    threshold: float
    observed_budget: Optional[int]
    interpolated_budget: Optional[float]
    smoothed_points: List[dict]


def _coerce_budget_rows(budget_rows: Iterable[dict]) -> List[dict]:
    numeric_rows: List[dict] = []
    for row in budget_rows:
        try:
            numeric_rows.append(
                {
                    "budget": int(row["budget"]),
                    "macro_f1": float(row["macro_f1"]),
                }
            )
        except (TypeError, ValueError, KeyError):
            continue
    return sorted(numeric_rows, key=lambda row: row["budget"])


def estimate_first_observed_budget(budget_rows: Iterable[dict], threshold: float) -> Optional[int]:
    """Return the first observed budget that reaches a target score."""
    for row in _coerce_budget_rows(budget_rows):
        if row["macro_f1"] >= threshold:
            return row["budget"]
    return None


def smooth_budget_curve(budget_rows: Iterable[dict]) -> List[dict]:
    """Fit a monotonic learning curve using isotonic regression."""
    rows = _coerce_budget_rows(budget_rows)
    if not rows:
        return []

    xs = np.array([row["budget"] for row in rows], dtype=float)
    ys = np.array([row["macro_f1"] for row in rows], dtype=float)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    smoothed = model.fit_transform(xs, ys)
    return [
        {
            "budget": int(row["budget"]),
            "macro_f1": float(row["macro_f1"]),
            "smoothed_macro_f1": float(value),
        }
        for row, value in zip(rows, smoothed)
    ]


def estimate_interpolated_budget(budget_rows: Iterable[dict], threshold: float) -> Optional[float]:
    """Estimate a budget using a smoothed learning curve and linear interpolation."""
    curve = smooth_budget_curve(budget_rows)
    if not curve:
        return None
    if curve[0]["smoothed_macro_f1"] >= threshold:
        return float(curve[0]["budget"])

    for left, right in zip(curve, curve[1:]):
        left_score = left["smoothed_macro_f1"]
        right_score = right["smoothed_macro_f1"]
        if left_score <= threshold <= right_score:
            if right_score == left_score:
                return float(right["budget"])
            fraction = (threshold - left_score) / (right_score - left_score)
            return float(left["budget"] + fraction * (right["budget"] - left["budget"]))
    return None


def summarize_budget_to_match(budget_rows: Iterable[dict], threshold: float) -> BudgetEstimate:
    """Summarize observed and interpolated budgets for a reference threshold."""
    curve = smooth_budget_curve(budget_rows)
    return BudgetEstimate(
        threshold=float(threshold),
        observed_budget=estimate_first_observed_budget(budget_rows, threshold),
        interpolated_budget=estimate_interpolated_budget(budget_rows, threshold),
        smoothed_points=curve,
    )
