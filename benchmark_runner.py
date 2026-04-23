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
import csv
import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class Example:
    text: str
    label: str


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


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_examples(path: str) -> List[Example]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        records: List[Example] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "text" not in obj or "label" not in obj:
                    raise ValueError(f"Missing 'text' or 'label' in {p} at line {line_no}")
                records.append(Example(text=str(obj["text"]), label=str(obj["label"])))
        return records

    if p.suffix.lower() == ".csv":
        records = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "text" not in row or "label" not in row:
                    raise ValueError(f"CSV must contain 'text' and 'label' columns: {p}")
                records.append(Example(text=str(row["text"]), label=str(row["label"])))
        return records

    raise ValueError(f"Unsupported file type for {p}. Use .jsonl or .csv")


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


def save_rows_to_csv(rows: List[dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_budget(records: List[Example], budget: int, seed: int) -> List[Example]:
    if budget >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, budget)


def label_to_id_map(label_names: Sequence[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(label_names)}


def examples_to_hf_dataset(records: List[Example], label_names: Sequence[str]) -> Dataset:
    label2id = label_to_id_map(label_names)
    payload = {
        "text": [r.text for r in records],
        "label": [label2id[r.label] for r in records],
    }
    return Dataset.from_dict(payload)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(true_labels: Sequence[str], pred_labels: Sequence[str], label_names: Sequence[str]) -> Dict[str, float]:
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=list(label_names), average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


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


# ---------------------------------------------------------------------
# Backend interfaces
# ---------------------------------------------------------------------

class ClassifierBackend(ABC):
    def __init__(self, model_spec: ModelSpec, label_names: Sequence[str]) -> None:
        self.model_spec = model_spec
        self.label_names = list(label_names)

    @abstractmethod
    def fit(self, train_records: List[Example], eval_records: Optional[List[Example]] = None) -> None:
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        pass


class ZeroShotNLIBackend(ClassifierBackend):
    """
    A real zero-shot baseline using the Hugging Face zero-shot-classification pipeline.
    This is not a generative LLM, but it is a strong and practical zero-shot benchmark.
    """
    def __init__(self, model_spec: ModelSpec, label_names: Sequence[str]) -> None:
        super().__init__(model_spec, label_names)
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_spec.model_id,
            device=device,
        )

    def fit(self, train_records: List[Example], eval_records: Optional[List[Example]] = None) -> None:
        # True zero-shot: no fitting.
        return

    def predict(self, texts: List[str]) -> List[str]:
        preds: List[str] = []
        bs = max(1, self.model_spec.batch_size)
        for start in range(0, len(texts), bs):
            batch = texts[start:start + bs]
            outputs = self.pipe(
                batch,
                candidate_labels=self.label_names,
                hypothesis_template=self.model_spec.hypothesis_template,
                multi_label=False,
            )
            # HF returns either a dict or a list of dicts depending on input size.
            if isinstance(outputs, dict):
                outputs = [outputs]
            for out in outputs:
                preds.append(out["labels"][0])
        return preds


class SequenceClassifierBackend(ClassifierBackend):
    """
    Fine-tuning backend for both BERT-style encoders and small transformer models.
    The difference between "finetune_encoder" and "finetune_slm" is primarily the model_id.
    """
    def __init__(self, model_spec: ModelSpec, label_names: Sequence[str], output_subdir: Path) -> None:
        super().__init__(model_spec, label_names)
        self.output_subdir = output_subdir
        self.label2id = label_to_id_map(label_names)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_spec.model_id,
            num_labels=len(label_names),
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _tokenize_dataset(self, ds: Dataset) -> Dataset:
        def tokenize_fn(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.model_spec.max_length,
            )
        return ds.map(tokenize_fn, batched=True)

    def fit(self, train_records: List[Example], eval_records: Optional[List[Example]] = None) -> None:
        train_ds = examples_to_hf_dataset(train_records, self.label_names)
        train_ds = self._tokenize_dataset(train_ds)

        eval_ds = None
        if eval_records:
            eval_ds = examples_to_hf_dataset(eval_records, self.label_names)
            eval_ds = self._tokenize_dataset(eval_ds)

        ensure_dir(self.output_subdir)
        training_args = TrainingArguments(
            output_dir=str(self.output_subdir),
            overwrite_output_dir=True,
            per_device_train_batch_size=self.model_spec.batch_size,
            per_device_eval_batch_size=self.model_spec.batch_size,
            num_train_epochs=self.model_spec.num_train_epochs,
            learning_rate=self.model_spec.learning_rate,
            weight_decay=self.model_spec.weight_decay,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="no" if eval_ds is None else "epoch",
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
        )
        trainer.train()

    def predict(self, texts: List[str]) -> List[str]:
        self.model.eval()
        preds: List[str] = []
        bs = max(1, self.model_spec.batch_size)

        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch_texts = texts[start:start + bs]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.model_spec.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
                preds.extend(self.id2label[i] for i in pred_ids)
        return preds


class ChatLLMBackend(ClassifierBackend):
    """
    Intentional stub. A future repo file can implement:
    - OpenAI / Anthropic / Together / vLLM adapters,
    - prompt templating,
    - output parsing,
    - caching and retry logic.

    Keeping this stub here is useful because it defines the interface now.
    """
    def fit(self, train_records: List[Example], eval_records: Optional[List[Example]] = None) -> None:
        return

    def predict(self, texts: List[str]) -> List[str]:
        raise NotImplementedError(
            "ChatLLMBackend is a placeholder. Implement in a later repo file, "
            "but keep the same .predict(List[str]) -> List[str] interface."
        )


# ---------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------

def build_backend(model_spec: ModelSpec, label_names: Sequence[str], run_dir: Path) -> ClassifierBackend:
    if model_spec.strategy == "zero_shot_nli":
        return ZeroShotNLIBackend(model_spec, label_names)
    if model_spec.strategy in {"finetune_encoder", "finetune_slm"}:
        return SequenceClassifierBackend(model_spec, label_names, output_subdir=run_dir)
    if model_spec.strategy == "zero_shot_llm":
        return ChatLLMBackend(model_spec, label_names)
    raise ValueError(f"Unknown strategy: {model_spec.strategy}")


def evaluate_backend(
    backend: ClassifierBackend,
    records: List[Example],
    label_names: Sequence[str],
) -> Tuple[Dict[str, float], List[dict]]:
    texts = [r.text for r in records]
    true_labels = [r.label for r in records]
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


def run_single_experiment(config: BenchmarkConfig, experiment: ExperimentSpec) -> Tuple[List[dict], List[dict]]:
    source_files = config.domains[experiment.source_domain]
    target_files = config.domains[experiment.target_domain]

    source_train = read_examples(source_files.train)
    source_test = read_examples(source_files.test)
    target_train = read_examples(target_files.train)
    target_test = read_examples(target_files.test)

    all_metric_rows: List[dict] = []
    all_prediction_rows: List[dict] = []

    # Store a reference target score from the strongest zero-shot baseline we have observed so far.
    reference_target_scores: Dict[str, float] = {}

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
            n_star = estimate_budget_to_reach_threshold(budget_rows_for_this_model, threshold=reference_score)
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

    return all_metric_rows, all_prediction_rows


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Starter benchmark runner for domain-shift adaptation experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    all_metric_rows: List[dict] = []
    all_prediction_rows: List[dict] = []

    for experiment in config.experiments:
        metric_rows, prediction_rows = run_single_experiment(config, experiment)
        all_metric_rows.extend(metric_rows)
        all_prediction_rows.extend(prediction_rows)

    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)
    save_rows_to_csv(all_metric_rows, output_dir / "benchmark_metrics.csv")
    save_rows_to_csv(all_prediction_rows, output_dir / "benchmark_predictions.csv")

    print(f"[done] wrote metrics to: {output_dir / 'benchmark_metrics.csv'}")
    print(f"[done] wrote predictions to: {output_dir / 'benchmark_predictions.csv'}")


if __name__ == "__main__":
    main()
