from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import Dataset


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def read_examples(path: str) -> List[dict]:
    """Load examples from a local JSONL or CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.suffix.lower() == ".jsonl":
        records: List[dict] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "text" not in row or "label" not in row:
                    raise ValueError(f"Missing 'text' or 'label' in {file_path} at line {line_no}")
                records.append({"text": str(row["text"]), "label": str(row["label"])})
        return records

    if file_path.suffix.lower() == ".csv":
        records = []
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "text" not in row or "label" not in row:
                    raise ValueError(f"CSV must contain 'text' and 'label' columns: {file_path}")
                records.append({"text": str(row["text"]), "label": str(row["label"])})
        return records

    raise ValueError(f"Unsupported file type for {file_path}. Use .jsonl or .csv")


def save_rows_to_csv(rows: List[dict], path: Path) -> None:
    """Write row dictionaries to CSV."""
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    """Write a JSON artifact with deterministic formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_budget(records: List[dict], budget: int, seed: int) -> List[dict]:
    """Sample a deterministic target-domain subset for a label budget."""
    if budget >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, budget)


def label_to_id_map(label_names: Sequence[str]) -> Dict[str, int]:
    """Map human-readable labels to classifier ids."""
    return {label: idx for idx, label in enumerate(label_names)}


def records_to_hf_dataset(records: List[dict], label_names: Sequence[str]) -> Dataset:
    """Convert local records into a Hugging Face dataset."""
    label2id = label_to_id_map(label_names)
    payload = {
        "text": [row["text"] for row in records],
        "label": [label2id[row["label"]] for row in records],
    }
    return Dataset.from_dict(payload)
