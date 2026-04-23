# Adaptation-Need Prediction Under Domain Shift

 The goal is to study whether we can predict, from unlabeled target-domain data, how much performance drops under domain shift and how much labeled target-domain data is needed for a smaller adapted model to catch up.


**Project Goal**

The larger project asks three practical questions:

1. How much do zero-shot or few-shot LLM-style baselines degrade under domain shift?
2. How much labeled target data is needed for a fine-tuned smaller model to match or beat that target-domain performance?
3. Which adaptation strategy is the better practical choice once performance and cost are considered together?

This starter repo does not attempt to solve the full research agenda yet. It focuses on a credible first prototype:

- run simple cross-domain benchmarks from local files
- compute lightweight domain-shift features
- estimate target label budgets more robustly
- summarize rough cost tradeoffs between LLM serving and target adaptation

**What’s Implemented**

- Source-to-source and source-to-target evaluation from local `jsonl` or `csv` datasets
- Zero-shot NLI baseline through Hugging Face pipelines
- Fine-tuned encoder or smaller-model baselines using the same runner
- Target-domain budget sweeps for learning curves
- Domain-shift features:
  lexical Jaccard overlap, TF-IDF centroid distance, document-length shift, label-frequency shift, and a simple domain-classifier proxy
- Budget analysis:
  isotonic smoothing plus observed and interpolated estimates for the budget needed to match a reference score
- Cost analysis:
  lightweight LLM serving, annotation, and fine-tuning cost estimates
- Optional chat-LLM adapter scaffold via environment variables

**Repository Layout**

- [`benchmark_runner.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/benchmark_runner.py): main benchmark spine and CLI
- [`src/model_backends.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/src/model_backends.py): zero-shot, fine-tuning, and optional LLM backends
- [`src/domain_shift.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/src/domain_shift.py): domain-shift feature extraction
- [`src/budget_analysis.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/src/budget_analysis.py): smoothed learning-curve analysis and `n*` estimation
- [`src/cost_model.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/src/cost_model.py): simple deployment cost utilities
- [`configs/sentiment_demo.json`](/Users/ziruihan/Desktop/Anote-Text-Classcification/configs/sentiment_demo.json): toy sentiment demo
- [`configs/nli_demo.json`](/Users/ziruihan/Desktop/Anote-Text-Classcification/configs/nli_demo.json): toy NLI-style demo
- [`data/README.md`](/Users/ziruihan/Desktop/Anote-Text-Classcification/data/README.md): expected dataset format

**Setup**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run The Benchmark**

Sentiment demo:

```bash
python3 benchmark_runner.py --config configs/sentiment_demo.json
```

NLI demo:

```bash
python3 benchmark_runner.py --config configs/nli_demo.json
```

Each run writes artifacts under the config’s `output_dir`, including:

- `benchmark_metrics.csv`
- `benchmark_predictions.csv`
- `domain_shift_features.csv`
- `domain_shift_features.json`
- `budget_summary.csv`
- `cost_analysis.csv`

**Dataset Format**

Each local dataset file must provide:

- `text`
- `label`

Example JSONL row:

```json
{"text": "The merger improved earnings guidance.", "label": "positive"}
```

The included `data/` files are tiny toy datasets for inspection and smoke testing. Replace them with real source and target datasets for meaningful experiments.

**Optional Chat LLM Backend**

The runner includes an optional `zero_shot_llm` strategy scaffold in [`src/llm_adapter.py`](/Users/ziruihan/Desktop/Anote-Text-Classcification/src/llm_adapter.py). It does not hardcode secrets.

Set:

- `OPENAI_API_KEY`
- optionally `OPENAI_BASE_URL`

This path is intentionally minimal and should be treated as an extension point, not a production-ready integration.

**Current Scope**

This repo is meant to look like a starter prototype, not finished paper code. It does not yet include:

- a meta-predictor that forecasts performance drop or target budget from shift features
- repeated-seed confidence intervals
- large-scale dataset curation
- production LLM prompting, caching, or evaluation infrastructure

Those are reasonable next steps after this benchmark backbone is validated on larger datasets.
