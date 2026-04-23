from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class CostConfig:
    """Simple cost assumptions for comparing adaptation strategies."""
    llm_cost_per_1k_examples: float = 5.0
    annotation_cost_per_example: float = 0.35
    finetune_cost_per_run: float = 18.0
    deployment_examples_per_month: int = 10000
    deployment_months: int = 6


def build_cost_config(raw_config: Optional[dict]) -> CostConfig:
    """Construct a cost config from optional JSON config values."""
    if not raw_config:
        return CostConfig()
    known_fields = {field.name for field in CostConfig.__dataclass_fields__.values()}
    payload = {key: value for key, value in raw_config.items() if key in known_fields}
    return CostConfig(**payload)


def estimate_strategy_costs(
    budget: Optional[float],
    cost_config: CostConfig,
) -> dict:
    """Estimate serving, annotation, and fine-tuning costs for a target budget."""
    deployment_examples = cost_config.deployment_examples_per_month * cost_config.deployment_months
    llm_serving_cost = (deployment_examples / 1000.0) * cost_config.llm_cost_per_1k_examples
    if budget is None:
        budget = 0.0

    annotation_cost = float(budget) * cost_config.annotation_cost_per_example
    finetune_total_cost = cost_config.finetune_cost_per_run if budget > 0 else 0.0
    adapted_model_total = annotation_cost + finetune_total_cost

    return {
        **asdict(cost_config),
        "estimated_budget": None if budget is None else float(budget),
        "llm_serving_cost_over_horizon": float(llm_serving_cost),
        "annotation_cost": float(annotation_cost),
        "finetune_cost": float(finetune_total_cost),
        "adapted_model_total_cost": float(adapted_model_total),
        "llm_minus_adapted_cost_delta": float(llm_serving_cost - adapted_model_total),
    }
