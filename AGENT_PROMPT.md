# Agent Prompt for Building the Rest of the Repository

You are helping extend a starter research repository for the project:

**Adaptation-Need Prediction Under Domain Shift**

The most important existing file is `benchmark_runner.py`. Treat it as the repository spine and do **not** break its CLI or core abstractions unless there is a very strong reason.

## Project Goal

The larger project studies whether we can predict, from **unlabeled target-domain data**:

1. how much zero-shot / few-shot LLM performance drops under domain shift,
2. how much labeled target-domain data is needed for a fine-tuned smaller model (e.g., BERT or SLM) to match or surpass that performance,
3. and eventually which adaptation strategy is the best practical choice.

This repository does **not** need to solve the full project yet. It only needs to become a strong, credible starter prototype for a fellowship application.

## Existing Starter File

`benchmark_runner.py` already does the following:
- loads source and target domain datasets from local JSONL / CSV,
- benchmarks multiple model strategies,
- computes cross-domain transfer performance,
- sweeps target-domain label budgets,
- writes metrics and predictions to CSV.

This file is the central benchmark file and should remain easy to run and understand.

## Your Priorities

Please build the rest of the repository around this file in a clean, research-friendly way.

### Priority 1: Make the repository runnable and legible
Add the minimum set of files needed so that a reader can understand and run the project quickly.

Recommended files:
- `README.md`
- `requirements.txt`
- `configs/sentiment_demo.json`
- `configs/nli_demo.json`
- `data/README.md`

### Priority 2: Factor out reusable utilities without overengineering
If you split code, keep interfaces simple and stable.

Good candidates:
- `src/data_utils.py`
- `src/metrics.py`
- `src/model_backends.py`
- `src/cost_model.py`
- `src/budget_analysis.py`

But do **not** create many tiny files with trivial content. Prefer a small number of useful modules.

### Priority 3: Add missing research components
Implement the next most valuable components after the benchmark file:

1. **Domain shift features**
   - lexical overlap / Jaccard
   - embedding-centroid distance
   - document-length and label-frequency shift
   - simple domain-classifier proxy if feasible

2. **Budget analysis**
   - interpolate or smooth target learning curves
   - estimate `n*` more robustly
   - summarize "budget to match reference" across runs

3. **Cost utilities**
   Add a lightweight module that defines:
   - LLM serving cost
   - annotation cost
   - fine-tuning cost
   - total cost over a deployment horizon

4. **Optional LLM adapter**
   Add a separate file for a pluggable chat-LLM backend, but do not hardcode secrets.
   Use environment variables and a clean interface.

## Coding Constraints

- Use Python.
- Write code that is easy to read and credible in a public GitHub repo.
- Add type hints where useful.
- Add concise docstrings.
- Prefer deterministic behavior with explicit random seeds.
- Save outputs in simple formats: CSV and JSON.
- Avoid overcomplicated frameworks.
- Avoid hidden state and notebook-only logic.
- Do not remove the local-file dataset pathway.
- Do not require proprietary APIs unless clearly optional.

## Deliverables You Should Produce

When coding additional files, aim to produce:
1. a repo structure that is easy to understand,
2. at least one demo config and instructions to run it,
3. one or two utility modules that clearly extend the benchmark,
4. a clean README that explains the project motivation and current scope.

## Style Expectations

The repository should look like:
- a serious research prototype,
- easy for a reviewer to inspect,
- not bloated,
- not pretending to solve everything.

The code should be honest about what is implemented and what remains future work.

## Important

This is a starter repo for a fellowship application, not the full paper codebase. Prioritize clarity, realism, and the benchmark backbone over completeness.
