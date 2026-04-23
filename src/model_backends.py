from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.data_utils import ensure_dir, label_to_id_map, records_to_hf_dataset
from src.llm_adapter import OpenAICompatibleChatClassifier


class ClassifierBackend(ABC):
    def __init__(self, model_spec: object, label_names: Sequence[str]) -> None:
        self.model_spec = model_spec
        self.label_names = list(label_names)

    @abstractmethod
    def fit(self, train_records: List[object], eval_records: Optional[List[object]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        raise NotImplementedError


class ZeroShotNLIBackend(ClassifierBackend):
    """Hugging Face zero-shot NLI baseline."""

    def __init__(self, model_spec: object, label_names: Sequence[str]) -> None:
        super().__init__(model_spec, label_names)
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("zero-shot-classification", model=model_spec.model_id, device=device)

    def fit(self, train_records: List[object], eval_records: Optional[List[object]] = None) -> None:
        return

    def predict(self, texts: List[str]) -> List[str]:
        predictions: List[str] = []
        batch_size = max(1, self.model_spec.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            outputs = self.pipe(
                batch,
                candidate_labels=self.label_names,
                hypothesis_template=self.model_spec.hypothesis_template,
                multi_label=False,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for output in outputs:
                predictions.append(output["labels"][0])
        return predictions


class SequenceClassifierBackend(ClassifierBackend):
    """Fine-tuning backend for encoder and smaller transformer classifiers."""

    def __init__(self, model_spec: object, label_names: Sequence[str], output_subdir: Path) -> None:
        super().__init__(model_spec, label_names)
        self.output_subdir = output_subdir
        self.label2id = label_to_id_map(label_names)
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_spec.model_id,
            num_labels=len(label_names),
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _tokenize_dataset(self, dataset: object) -> object:
        def tokenize_fn(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.model_spec.max_length,
            )

        return dataset.map(tokenize_fn, batched=True)

    def fit(self, train_records: List[object], eval_records: Optional[List[object]] = None) -> None:
        train_dataset = self._tokenize_dataset(records_to_hf_dataset(train_records, self.label_names))
        eval_dataset = None
        if eval_records:
            eval_dataset = self._tokenize_dataset(records_to_hf_dataset(eval_records, self.label_names))

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
            eval_strategy="no" if eval_dataset is None else "epoch",
            report_to=[],
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()

    def predict(self, texts: List[str]) -> List[str]:
        self.model.eval()
        predictions: List[str] = []
        batch_size = max(1, self.model_spec.batch_size)

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.model_spec.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
                predictions.extend(self.id2label[idx] for idx in pred_ids)
        return predictions


class ChatLLMBackend(ClassifierBackend):
    """Optional zero-shot backend implemented via an OpenAI-compatible chat API."""

    def __init__(self, model_spec: object, label_names: Sequence[str]) -> None:
        super().__init__(model_spec, label_names)
        self.client = OpenAICompatibleChatClassifier(model_spec.model_id, label_names)

    def fit(self, train_records: List[object], eval_records: Optional[List[object]] = None) -> None:
        return

    def predict(self, texts: List[str]) -> List[str]:
        return self.client.predict(texts)


def build_backend(model_spec: object, label_names: Sequence[str], run_dir: Path) -> ClassifierBackend:
    """Construct a backend from a benchmark model spec."""
    if model_spec.strategy == "zero_shot_nli":
        return ZeroShotNLIBackend(model_spec, label_names)
    if model_spec.strategy in {"finetune_encoder", "finetune_slm"}:
        return SequenceClassifierBackend(model_spec, label_names, output_subdir=run_dir)
    if model_spec.strategy == "zero_shot_llm":
        return ChatLLMBackend(model_spec, label_names)
    raise ValueError(f"Unknown strategy: {model_spec.strategy}")
