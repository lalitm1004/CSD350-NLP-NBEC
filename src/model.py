from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, cast

import scipy.sparse as sp
import torch

from dataset import Labels, NaiveBayesDataset
from preprocess import FeatureExtractor

logger = logging.getLogger(__name__)


class NaiveBayesModel:
    def __init__(self, dataset: NaiveBayesDataset, alpha: float = 1.0) -> None:
        logger.info("Initializing NaiveBayesModel (alpha=%.2f).", alpha)
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = len(Labels)
        self.vocab_size: int = (
            len(dataset.feature_extractor.vocab)
            if dataset.feature_extractor.vocab
            else 0
        )

        self.log_priors: torch.Tensor = torch.zeros(self.num_classes)
        self.log_likelihoods: torch.Tensor = torch.zeros(
            (self.num_classes, self.vocab_size)
        )
        self.idf: torch.Tensor = torch.zeros(self.vocab_size)

        self.class_priors: torch.Tensor = torch.zeros(self.num_classes)
        self.likelihoods: torch.Tensor = torch.zeros(
            (self.num_classes, self.vocab_size)
        )

    def train(self) -> None:
        logger.info("Starting model training procedure...")
        self._compute_idf()
        self._train()
        logger.info("Model training completed.")

    def predict(self, document_features: torch.Tensor) -> int:
        tfidf = self._compute_tfidf(document_features)
        log_probs = self.log_priors + (self.log_likelihoods * tfidf).sum(dim=1)
        return int(torch.argmax(log_probs).item())

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        row_sums = x.sum(dim=1, keepdim=True).clamp(min=1e-8)
        tf = x / row_sums
        tfidf = tf * self.idf.unsqueeze(0)

        log_probs = tfidf @ self.log_likelihoods.T + self.log_priors
        return torch.argmax(log_probs, dim=1)

    def evaluate(self) -> float:
        return cast(float, self.compute_metrics()["accuracy"])

    def _predict_sparse_chunked(
        self, x_sp: sp.csr_matrix, batch_size: int = 512
    ) -> torch.Tensor:
        n_rows = int(x_sp.shape[0])  # type: ignore
        pred_chunks: List[torch.Tensor] = []

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            chunk = torch.tensor(x_sp[start:end].toarray(), dtype=torch.float32)
            pred_chunks.append(self.predict_batch(chunk))

        return (
            torch.cat(pred_chunks, dim=0)
            if pred_chunks
            else torch.empty(0, dtype=torch.int64)
        )

    def compute_metrics(self, batch_size: int = 512) -> Dict[str, object]:
        logger.debug(
            "Computing metrics over %d samples (batch_size=%d)...",
            len(self.dataset),
            batch_size,
        )

        true_labels = self.dataset.labels
        predicted_labels = self._predict_sparse_chunked(
            self.dataset.features_matrix, batch_size=batch_size
        )

        # Define positive class (SPAM = 1)
        spam_class = 1

        tp = int(
            ((predicted_labels == spam_class) & (true_labels == spam_class))
            .sum()
            .item()
        )

        fp = int(
            ((predicted_labels == spam_class) & (true_labels != spam_class))
            .sum()
            .item()
        )

        fn = int(
            ((predicted_labels != spam_class) & (true_labels == spam_class))
            .sum()
            .item()
        )

        tn = int(
            ((predicted_labels != spam_class) & (true_labels != spam_class))
            .sum()
            .item()
        )

        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

    def _compute_tfidf(self, features: torch.Tensor) -> torch.Tensor:
        tf = features / (features.sum() + 1e-8)
        return tf * self.idf

    def _compute_idf(self) -> None:
        logger.debug("Computing IDF vectors...")
        n_docs = len(self.dataset)
        x: sp.csr_matrix = self.dataset.features_matrix

        doc_freq = torch.as_tensor((x > 0).sum(axis=0)).flatten().to(torch.float32)  # type: ignore
        self.idf = torch.log(
            torch.tensor(float(n_docs), dtype=torch.float32) / (doc_freq + 1.0)
        )

    def _train(self) -> None:
        logger.debug("Computing class counts and likelihoods (vectorised)...")
        x_sp: sp.csr_matrix = self.dataset.features_matrix
        y: torch.Tensor = self.dataset.labels

        word_counts = torch.zeros(
            (self.num_classes, self.vocab_size), dtype=torch.float32
        )

        for class_idx in range(self.num_classes):
            indices = torch.nonzero(y == class_idx, as_tuple=False).flatten().tolist()

            if not indices:
                continue

            summed = (
                torch.as_tensor(x_sp[indices].sum(axis=0)).flatten().to(torch.float32)
            )

            word_counts[class_idx] = summed

        total_tokens = word_counts.sum(dim=1)

        class_counts = torch.bincount(y, minlength=self.num_classes).to(torch.float32)

        self.class_priors = class_counts / class_counts.sum().clamp(min=1e-8)

        self.likelihoods = (word_counts + self.alpha) / (
            total_tokens.unsqueeze(1) + self.alpha * self.vocab_size
        )

        self.log_priors = torch.log(self.class_priors.clamp(min=1e-12))
        self.log_likelihoods = torch.log(self.likelihoods.clamp(min=1e-12))

    def save_weights(
        self,
        filepath: Path,
        vocab: List[str],
        word_to_idx: Dict[str, int],
    ) -> None:
        logger.debug("Saving model state to %s...", filepath)

        state = {
            "class_priors": self.class_priors,
            "likelihoods": self.likelihoods,
            "idf": self.idf,
            "log_priors": self.log_priors,
            "log_likelihoods": self.log_likelihoods,
            "vocab": vocab,
            "word_to_idx": word_to_idx,
            "vocab_size": self.vocab_size,
        }

        torch.save(state, filepath)

    @classmethod
    def load_weights(
        cls, filepath: Path, dataset: NaiveBayesDataset
    ) -> "NaiveBayesModel":
        logger.debug("Loading model state from %s...", filepath)

        state = torch.load(filepath, weights_only=False)

        dataset.feature_extractor.vocab = state["vocab"]
        dataset.feature_extractor.word_to_idx = state["word_to_idx"]

        model = cls(dataset)

        model.class_priors = state["class_priors"]
        model.likelihoods = state["likelihoods"]
        model.idf = state["idf"]
        model.log_priors = state["log_priors"]
        model.log_likelihoods = state["log_likelihoods"]
        model.vocab_size = state["vocab_size"]

        return model

    @classmethod
    def load_for_inference(
        cls, filepath: Path
    ) -> "tuple[NaiveBayesModel, FeatureExtractor]":
        logger.debug("Loading inference state from %s...", filepath)

        state = torch.load(filepath, weights_only=False)

        feature_extractor = FeatureExtractor()
        feature_extractor.vocab = state["vocab"]
        feature_extractor.word_to_idx = state["word_to_idx"]

        class _InferenceStub:
            labels = None
            features_matrix = None

            def __len__(self) -> int:
                return 0

        stub = _InferenceStub()
        stub.feature_extractor = feature_extractor  # type: ignore[attr-defined]

        model = cls.__new__(cls)
        model.dataset = stub  # type: ignore[assignment]
        model.alpha = 1.0
        model.num_classes = len(Labels)
        model.vocab_size = state["vocab_size"]
        model.class_priors = state["class_priors"]
        model.likelihoods = state["likelihoods"]
        model.idf = state["idf"]
        model.log_priors = state["log_priors"]
        model.log_likelihoods = state["log_likelihoods"]

        return model, feature_extractor
