# Data Layout

The benchmark expects local JSONL or CSV files with two fields:

- `text`
- `label`

Each config points to domain-specific `train` and `test` files. This repository includes tiny toy datasets only to make the starter prototype easy to inspect and run.

## JSONL example

```json
{"text": "The battery life is excellent.", "label": "positive"}
{"text": "The interface crashes frequently.", "label": "negative"}
```

## Notes

- Replace the demo files with your own datasets for real experiments.
- Keep label names aligned with `label_names` in the config.
- Domain-shift features are computed from the raw text in the configured source and target splits.
