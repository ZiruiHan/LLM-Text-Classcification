#!/usr/bin/env python3
"""
benchmark_runner.py

Starter benchmark runner for the "adaptation under domain shift" project.

Purpose
-------
This file is the core starter benchmark for comparing:
1) zero-shot NLI-style models,
2) fine-tuned encoder models (e.g., BERT, RoBERTa),
3) fine-tuned smaller models (SLMs) using the same sequence-classification interface.

What this script does
---------------------
- Loads local train/test datasets for multiple domains.
- Runs source -> source and source -> target evaluation.
- Computes accuracy, macro-F1, and performance drop under domain shift.
- Builds target-domain learning curves for fine-tuned models by sweeping label budgets.
- Estimates the smallest target budget needed to match a reference score
  (for example, zero-shot LLM / zero-shot NLI performance on the target domain).
- Writes machine-readable CSV outputs for later analysis.

What this script does NOT do yet
--------------------------------
- It does not implement a production LLM API adapter. There is a stub for that.
- It does not yet compute domain-shift features such as PAD/PAD*/MMD.
- It does not yet fit the meta-predictor that forecasts drop and label budget.
- It is intentionally the *benchmark spine* of the repository, not the entire project.

Expected local data format
--------------------------
Each dataset file should be either JSONL or CSV with the columns:
- text
- label

Example config JSON
-------------------
{
  "task_name": "sentiment",
  "label_names": ["negative", "positive"],
  "output_dir": "outputs/sentiment",
  "seed": 42,
  "domains": {
    "reviews": {
      "train": "data/reviews_train.jsonl",
      "test": "data/reviews_test.jsonl"
    },
    "finance": {
      "train": "data/finance_train.jsonl",
      "test": "data/finance_test.jsonl"
    }
  },
  "experiments": [
    {
      "source_domain": "reviews",
      "target_domain": "finance",
      "target_budgets": [16, 32, 64, 128, 256],
      "models": [
        {
          "name": "bart_mnli",
          "strategy": "zero_shot_nli",
          "model_id": "facebook/bart-large-mnli",
          "batch_size": 8
        },
        {
          "name": "bert_base",
          "strategy": "finetune_encoder",
          "model_id": "bert-base-uncased",
          "batch_size": 16,
          "num_train_epochs": 2,
          "learning_rate": 2e-5,
          "max_length": 256
        },
        {
          "name": "distilbert_small",
          "strategy": "finetune_slm",
          "model_id": "distilbert-base-uncased",
          "batch_size": 16,
          "num_train_epochs": 2,
          "learning_rate": 3e-5,
          "max_length": 256
        }
      ]
    }
  ]
}

Usage
-----
python benchmark_runner.py --config configs/sentiment_demo.json

Recommended pip packages
------------------------
pip install torch transformers datasets scikit-learn pandas

Notes
-----
- For a fellowship application, this file is meant to demonstrate credible project structure.
- The most important outputs are:
  (1) cross-domain transfer metrics,
  (2) target learning curves,
  (3) required target label budget to reach a reference score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.budget_analysis import summarize_budget_to_match
from src.cost_model import build_cost_config, estimate_strategy_costs
from src.data_utils import ensure_dir, read_examples, sample_budget, save_rows_to_csv, set_seed, write_json
from src.domain_shift import compute_domain_shift_summary
from src.metrics import compute_classification_metrics
from src.model_backends import ClassifierBackend, build_backend


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class DomainFiles:
    train: str
    test: str


@dataclass
class ModelSpec:
    name: str
    strategy: str
    model_id: str
    batch_size: int = 8
    max_length: int = 256
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    hypothesis_template: str = "This text is about {}."


@dataclass
class ExperimentSpec:
    source_domain: str
    target_domain: str
    models: List[ModelSpec]
    target_budgets: List[int] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    task_name: str
    label_names: List[str]
    output_dir: str
    domains: Dict[str, DomainFiles]
    experiments: List[ExperimentSpec]
    seed: int = 42


def load_config(path: str) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    domains = {k: DomainFiles(**v) for k, v in raw["domains"].items()}
    experiments = []
    for exp in raw["experiments"]:
        model_specs = [ModelSpec(**m) for m in exp["models"]]
        experiments.append(
            ExperimentSpec(
                source_domain=exp["source_domain"],
                target_domain=exp["target_domain"],
                models=model_specs,
                target_budgets=exp.get("target_budgets", []),
            )
        )

    return BenchmarkConfig(
        task_name=raw["task_name"],
        label_names=raw["label_names"],
        output_dir=raw["output_dir"],
        domains=domains,
        experiments=experiments,
        seed=raw.get("seed", 42),
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(true_labels: Sequence[str], pred_labels: Sequence[str], label_names: Sequence[str]) -> Dict[str, float]:
    return compute_classification_metrics(true_labels, pred_labels, label_names)


def estimate_budget_to_reach_threshold(budget_rows: List[dict], threshold: float) -> Optional[int]:
    """
    Return the smallest budget whose macro_f1 meets or exceeds `threshold`.
    """
    if not budget_rows:
        return None
    sorted_rows = sorted(budget_rows, key=lambda row: int(row["budget"]))
    for row in sorted_rows:
        if float(row["macro_f1"]) >= threshold:
            return int(row["budget"])
    return None


def evaluate_backend(
    backend: ClassifierBackend,
    records: List[dict],
    label_names: Sequence[str],
) -> Tuple[Dict[str, float], List[dict]]:
    texts = [r["text"] for r in records]
    true_labels = [r["label"] for r in records]
    pred_labels = backend.predict(texts)
    metrics = compute_metrics(true_labels, pred_labels, label_names)
    preds = []
    for text, gold, pred in zip(texts, true_labels, pred_labels):
        preds.append({
            "text": text,
            "gold_label": gold,
            "pred_label": pred,
        })
    return metrics, preds


def run_single_experiment(
    config: BenchmarkConfig,
    experiment: ExperimentSpec,
    cost_config: object,
) -> Tuple[List[dict], List[dict], dict, List[dict], List[dict]]:
    source_files = config.domains[experiment.source_domain]
    target_files = config.domains[experiment.target_domain]

    source_train = read_examples(source_files.train)
    source_test = read_examples(source_files.test)
    target_train = read_examples(target_files.train)
    target_test = read_examples(target_files.test)

    all_metric_rows: List[dict] = []
    all_prediction_rows: List[dict] = []
    budget_summary_rows: List[dict] = []
    cost_rows: List[dict] = []

    # Store a reference target score from the strongest zero-shot baseline we have observed so far.
    reference_target_scores: Dict[str, float] = {}

    domain_shift_summary = {
        "task_name": config.task_name,
        "source_domain": experiment.source_domain,
        "target_domain": experiment.target_domain,
        **compute_domain_shift_summary(
            [record["text"] for record in source_train],
            [record["text"] for record in target_train],
            config.label_names,
            seed=config.seed,
            source_labels=[record["label"] for record in source_train],
            target_labels=[record["label"] for record in target_train],
        ),
    }

    for model_spec in experiment.models:
        run_name = f"{config.task_name}__{experiment.source_domain}_to_{experiment.target_domain}__{model_spec.name}"
        run_dir = Path(config.output_dir) / run_name
        ensure_dir(run_dir)

        # -------------------------------------------------------------
        # Case 1: zero-shot strategy
        # -------------------------------------------------------------
        if model_spec.strategy in {"zero_shot_nli", "zero_shot_llm"}:
            backend = build_backend(model_spec, config.label_names, run_dir / "zero_shot")
            backend.fit([])

            src_metrics, src_preds = evaluate_backend(backend, source_test, config.label_names)
            tgt_metrics, tgt_preds = evaluate_backend(backend, target_test, config.label_names)

            drop = src_metrics["macro_f1"] - tgt_metrics["macro_f1"]
            metric_rows = [
                {
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "evaluation_split": "source_test",
                    "budget": 0,
                    **src_metrics,
                    "macro_f1_drop_from_source": 0.0,
                },
                {
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "evaluation_split": "target_test",
                    "budget": 0,
                    **tgt_metrics,
                    "macro_f1_drop_from_source": drop,
                },
            ]
            all_metric_rows.extend(metric_rows)

            for split_name, preds in [("source_test", src_preds), ("target_test", tgt_preds)]:
                for p in preds:
                    p.update({
                        "task_name": config.task_name,
                        "source_domain": experiment.source_domain,
                        "target_domain": experiment.target_domain,
                        "model_name": model_spec.name,
                        "model_id": model_spec.model_id,
                        "strategy": model_spec.strategy,
                        "evaluation_split": split_name,
                        "budget": 0,
                    })
                    all_prediction_rows.append(p)

            reference_target_scores[model_spec.name] = tgt_metrics["macro_f1"]
            continue

        # -------------------------------------------------------------
        # Case 2: fine-tuned smaller models
        # -------------------------------------------------------------
        # A. Train on source domain, evaluate on source and target.
        transfer_backend = build_backend(model_spec, config.label_names, run_dir / "source_transfer")
        transfer_backend.fit(source_train, eval_records=source_test)

        src_metrics, src_preds = evaluate_backend(transfer_backend, source_test, config.label_names)
        tgt_metrics, tgt_preds = evaluate_backend(transfer_backend, target_test, config.label_names)

        drop = src_metrics["macro_f1"] - tgt_metrics["macro_f1"]
        all_metric_rows.extend([
            {
                "task_name": config.task_name,
                "source_domain": experiment.source_domain,
                "target_domain": experiment.target_domain,
                "model_name": model_spec.name,
                "model_id": model_spec.model_id,
                "strategy": model_spec.strategy,
                "evaluation_split": "source_test",
                "budget": "source_train_full",
                **src_metrics,
                "macro_f1_drop_from_source": 0.0,
            },
            {
                "task_name": config.task_name,
                "source_domain": experiment.source_domain,
                "target_domain": experiment.target_domain,
                "model_name": model_spec.name,
                "model_id": model_spec.model_id,
                "strategy": model_spec.strategy,
                "evaluation_split": "target_test_transfer",
                "budget": "source_train_full",
                **tgt_metrics,
                "macro_f1_drop_from_source": drop,
            },
        ])

        for split_name, preds in [("source_test", src_preds), ("target_test_transfer", tgt_preds)]:
            for p in preds:
                p.update({
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "evaluation_split": split_name,
                    "budget": "source_train_full",
                })
                all_prediction_rows.append(p)

        # B. Build target-domain learning curve.
        budget_rows_for_this_model: List[dict] = []
        for budget in experiment.target_budgets:
            budget_train = sample_budget(target_train, budget, seed=config.seed + budget)
            budget_backend = build_backend(model_spec, config.label_names, run_dir / f"target_budget_{budget}")
            budget_backend.fit(budget_train, eval_records=target_test)

            budget_metrics, budget_preds = evaluate_backend(budget_backend, target_test, config.label_names)
            row = {
                "task_name": config.task_name,
                "source_domain": experiment.source_domain,
                "target_domain": experiment.target_domain,
                "model_name": model_spec.name,
                "model_id": model_spec.model_id,
                "strategy": model_spec.strategy,
                "evaluation_split": "target_test_budget_curve",
                "budget": budget,
                **budget_metrics,
                "macro_f1_drop_from_source": src_metrics["macro_f1"] - budget_metrics["macro_f1"],
            }
            budget_rows_for_this_model.append(row)
            all_metric_rows.append(row)

            for p in budget_preds:
                p.update({
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "evaluation_split": "target_test_budget_curve",
                    "budget": budget,
                })
                all_prediction_rows.append(p)

        # Optional analysis: estimate first budget that matches a reference zero-shot score.
        # We use the best available zero-shot reference observed so far, if one exists.
        if reference_target_scores:
            reference_name, reference_score = max(reference_target_scores.items(), key=lambda kv: kv[1])
            budget_summary = summarize_budget_to_match(budget_rows_for_this_model, threshold=reference_score)
            n_star = budget_summary.observed_budget
            all_metric_rows.append({
                "task_name": config.task_name,
                "source_domain": experiment.source_domain,
                "target_domain": experiment.target_domain,
                "model_name": model_spec.name,
                "model_id": model_spec.model_id,
                "strategy": model_spec.strategy,
                "evaluation_split": "target_budget_summary",
                "budget": n_star if n_star is not None else "not_reached",
                "accuracy": "",
                "macro_f1": "",
                "macro_f1_drop_from_source": "",
                "reference_model_name": reference_name,
                "reference_target_macro_f1": reference_score,
                "estimated_budget_to_match_reference": n_star,
            })
            budget_summary_rows.append(
                {
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "reference_model_name": reference_name,
                    "reference_target_macro_f1": float(reference_score),
                    "observed_budget_to_match_reference": budget_summary.observed_budget,
                    "interpolated_budget_to_match_reference": budget_summary.interpolated_budget,
                    "smoothed_curve_points_json": json.dumps(budget_summary.smoothed_points),
                }
            )
            cost_rows.append(
                {
                    "task_name": config.task_name,
                    "source_domain": experiment.source_domain,
                    "target_domain": experiment.target_domain,
                    "model_name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "strategy": model_spec.strategy,
                    "reference_model_name": reference_name,
                    **estimate_strategy_costs(budget_summary.interpolated_budget, cost_config),
                }
            )

    return all_metric_rows, all_prediction_rows, domain_shift_summary, budget_summary_rows, cost_rows


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Starter benchmark runner for domain-shift adaptation experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    cost_config = build_cost_config(raw_config.get("cost_model"))

    all_metric_rows: List[dict] = []
    all_prediction_rows: List[dict] = []
    all_domain_shift_rows: List[dict] = []
    all_budget_summary_rows: List[dict] = []
    all_cost_rows: List[dict] = []

    for experiment in config.experiments:
        metric_rows, prediction_rows, domain_shift_summary, budget_summary_rows, cost_rows = run_single_experiment(
            config,
            experiment,
            cost_config,
        )
        all_metric_rows.extend(metric_rows)
        all_prediction_rows.extend(prediction_rows)
        all_domain_shift_rows.append(domain_shift_summary)
        all_budget_summary_rows.extend(budget_summary_rows)
        all_cost_rows.extend(cost_rows)

    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)
    save_rows_to_csv(all_metric_rows, output_dir / "benchmark_metrics.csv")
    save_rows_to_csv(all_prediction_rows, output_dir / "benchmark_predictions.csv")
    save_rows_to_csv(all_domain_shift_rows, output_dir / "domain_shift_features.csv")
    save_rows_to_csv(all_budget_summary_rows, output_dir / "budget_summary.csv")
    save_rows_to_csv(all_cost_rows, output_dir / "cost_analysis.csv")
    write_json(
        {
            "task_name": config.task_name,
            "experiments": all_domain_shift_rows,
        },
        output_dir / "domain_shift_features.json",
    )

    print(f"[done] wrote metrics to: {output_dir / 'benchmark_metrics.csv'}")
    print(f"[done] wrote predictions to: {output_dir / 'benchmark_predictions.csv'}")
    print(f"[done] wrote domain-shift features to: {output_dir / 'domain_shift_features.csv'}")
    print(f"[done] wrote budget summaries to: {output_dir / 'budget_summary.csv'}")
    print(f"[done] wrote cost analysis to: {output_dir / 'cost_analysis.csv'}")


if __name__ == "__main__":
    main()
