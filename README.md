# Adaptation Benchmark Starter

This starter repository contains the core benchmark file for a project on adaptation under domain shift.

## Main file
- `benchmark_runner.py`

## What it already supports
- source -> source evaluation
- source -> target transfer evaluation
- target-domain fine-tuning sweeps over label budgets
- accuracy and macro-F1 reporting
- CSV exports of metrics and predictions

## Intended next steps
- add domain-shift features
- add budget-curve analysis
- add cost utilities
- add an optional generative LLM backend
