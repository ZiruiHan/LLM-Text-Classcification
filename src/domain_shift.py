from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def lexical_jaccard(source_texts: Sequence[str], target_texts: Sequence[str]) -> float:
    """Compute token-set Jaccard overlap between domains."""
    src_vocab = {token for text in source_texts for token in _tokenize(text)}
    tgt_vocab = {token for text in target_texts for token in _tokenize(text)}
    if not src_vocab and not tgt_vocab:
        return 1.0
    union = src_vocab | tgt_vocab
    if not union:
        return 0.0
    return float(len(src_vocab & tgt_vocab) / len(union))


def embedding_centroid_distance(source_texts: Sequence[str], target_texts: Sequence[str]) -> float:
    """Approximate centroid shift with TF-IDF feature centroids."""
    if not source_texts or not target_texts:
        return 0.0
    vectorizer = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(list(source_texts) + list(target_texts))
    src_matrix = matrix[: len(source_texts)]
    tgt_matrix = matrix[len(source_texts) :]
    src_centroid = np.asarray(src_matrix.mean(axis=0)).ravel()
    tgt_centroid = np.asarray(tgt_matrix.mean(axis=0)).ravel()
    return float(np.linalg.norm(src_centroid - tgt_centroid))


def average_document_length(texts: Sequence[str]) -> float:
    """Return the average whitespace-tokenized document length."""
    if not texts:
        return 0.0
    return float(mean(len(_tokenize(text)) for text in texts))


def document_length_shift(source_texts: Sequence[str], target_texts: Sequence[str]) -> dict:
    """Summarize document-length differences across domains."""
    src_mean = average_document_length(source_texts)
    tgt_mean = average_document_length(target_texts)
    return {
        "source_avg_doc_length": float(src_mean),
        "target_avg_doc_length": float(tgt_mean),
        "avg_doc_length_shift": float(tgt_mean - src_mean),
        "avg_doc_length_ratio": float(tgt_mean / src_mean) if src_mean else None,
    }


def label_frequency_shift(
    source_labels: Optional[Sequence[str]],
    target_labels: Optional[Sequence[str]],
    label_names: Sequence[str],
) -> dict:
    """Compute total variation distance between label distributions when labels are available."""
    if not source_labels or not target_labels:
        return {"label_tv_distance": None}

    src_counts = Counter(source_labels)
    tgt_counts = Counter(target_labels)
    src_total = max(1, sum(src_counts.values()))
    tgt_total = max(1, sum(tgt_counts.values()))

    tv = 0.0
    result = {}
    for label in label_names:
        src_freq = src_counts.get(label, 0) / src_total
        tgt_freq = tgt_counts.get(label, 0) / tgt_total
        result[f"source_label_freq_{label}"] = float(src_freq)
        result[f"target_label_freq_{label}"] = float(tgt_freq)
        tv += abs(src_freq - tgt_freq)
    result["label_tv_distance"] = float(0.5 * tv)
    return result


def domain_classifier_proxy(source_texts: Sequence[str], target_texts: Sequence[str], seed: int) -> float:
    """Train a simple domain classifier and return held-out ROC AUC."""
    if len(source_texts) < 2 or len(target_texts) < 2:
        return 0.5

    texts = list(source_texts) + list(target_texts)
    labels = np.array([0] * len(source_texts) + [1] * len(target_texts))

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    vectorizer = TfidfVectorizer(min_df=1, max_features=10000, ngram_range=(1, 2))
    x_train_matrix = vectorizer.fit_transform(x_train)
    x_test_matrix = vectorizer.transform(x_test)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(x_train_matrix, y_train)
    y_prob = classifier.predict_proba(x_test_matrix)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def compute_domain_shift_summary(
    source_texts: Sequence[str],
    target_texts: Sequence[str],
    label_names: Sequence[str],
    seed: int,
    source_labels: Optional[Sequence[str]] = None,
    target_labels: Optional[Sequence[str]] = None,
) -> dict:
    """Compute simple domain-shift features for one source-target pair."""
    return {
        "lexical_jaccard": lexical_jaccard(source_texts, target_texts),
        "embedding_centroid_distance": embedding_centroid_distance(source_texts, target_texts),
        "domain_classifier_auc": domain_classifier_proxy(source_texts, target_texts, seed=seed),
        **document_length_shift(source_texts, target_texts),
        **label_frequency_shift(source_labels, target_labels, label_names),
    }
