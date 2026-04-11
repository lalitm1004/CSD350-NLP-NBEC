from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from dataset import Labels, NaiveBayesDataset
from preprocess import FeatureExtractor

logger = logging.getLogger(__name__)


class NaiveBayesModel:
    """Multinomial Naive Bayes with TF-IDF weighting.

    Performance notes
    -----------------
    - IDF is computed from the CSR matrix with a single vectorized op.
    - Training aggregates word sums per class via sparse matrix slicing —
      no Python loop over samples.
    - ``predict_batch`` and ``evaluate`` score the entire set in one matmul.
    """

    def __init__(self, dataset: NaiveBayesDataset, alpha: float = 1.0) -> None:
        logger.info("Initializing NaiveBayesModel (alpha=%.2f).", alpha)
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = len(Labels)
        self.vocab_size: int = (
            len(dataset.feature_extractor.vocab) if dataset.feature_extractor.vocab else 0
        )

        # Populated by train()
        self.log_priors: torch.Tensor = torch.zeros(self.num_classes)
        self.log_likelihoods: torch.Tensor = torch.zeros((self.num_classes, self.vocab_size))
        self.idf: torch.Tensor = torch.zeros(self.vocab_size)

        # Kept for save/load round-trip
        self.class_priors: torch.Tensor = torch.zeros(self.num_classes)
        self.likelihoods: torch.Tensor = torch.zeros((self.num_classes, self.vocab_size))

    # ── Public API ───────────────────────────────────────────────────────────

    def train(self) -> None:
        logger.info("Starting model training procedure...")
        self._compute_idf()
        self._train()
        logger.info("Model training completed.")

    def predict(self, features: torch.Tensor) -> int:
        """Predict class for a single dense feature vector."""
        tfidf = self._compute_tfidf(features)
        log_probs = self.log_priors + (self.log_likelihoods * tfidf).sum(dim=1)
        return int(torch.argmax(log_probs).item())

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Vectorised batch prediction.

        X : (N, vocab_size) dense float tensor
        Returns (N,) int tensor of predicted class indices.
        """
        # TF normalisation per row
        row_sums = X.sum(dim=1, keepdim=True).clamp(min=1e-8)
        tf = X / row_sums
        tfidf = tf * self.idf.unsqueeze(0)          # (N, V)

        # log p(c|x)  ∝  log_prior + tfidf @ log_likelihoods.T
        log_probs = tfidf @ self.log_likelihoods.T + self.log_priors  # (N, C)
        return torch.argmax(log_probs, dim=1)

    def evaluate(self) -> float:
        """Convenience wrapper — returns scalar accuracy."""
        return self.compute_metrics()["accuracy"]

    def _predict_sparse_chunked(
        self, X_sp: sp.csr_matrix, batch_size: int = 512
    ) -> np.ndarray:
        """Run prediction over a sparse CSR matrix without ever densifying it all.

        Processes *batch_size* rows at a time so peak memory is:
            batch_size × vocab_size × 4 bytes
        e.g. 512 rows × 274 k tokens × 4 B ≈ 560 MB — always manageable.

        Returns
        -------
        preds : np.ndarray of shape (N,), dtype int64
        """
        N = X_sp.shape[0]
        preds = np.empty(N, dtype=np.int64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = torch.from_numpy(
                X_sp[start:end].toarray().astype(np.float32)
            )                                                  # (B, V) dense
            preds[start:end] = self.predict_batch(chunk).numpy()

        return preds


    def compute_metrics(self, batch_size: int = 512) -> Dict[str, object]:
        """Full evaluation: accuracy, per-class P/R/F1, macro averages.

        Processes the dataset in chunks of *batch_size* rows so memory usage
        is bounded regardless of vocabulary size or dataset size.

        Parameters
        ----------
        batch_size:
            Number of samples densified at once.  Lower values use less RAM;
            higher values are faster.  Default 512 works well up to ~300k vocab.

        Returns
        -------
        dict with keys:
            accuracy        : float
            confusion_matrix: np.ndarray, shape (C, C)  [row=true, col=pred]
            classes         : list[str]  label names
            per_class       : list[dict] per-class precision / recall / f1 / support
            macro_precision : float
            macro_recall    : float
            macro_f1        : float
        """
        logger.debug(
            "Computing metrics over %d samples (batch_size=%d)...",
            len(self.dataset), batch_size,
        )
        y_true = self.dataset.labels                                      # np (N,)
        preds  = self._predict_sparse_chunked(                            # np (N,)
            self.dataset.features_matrix, batch_size=batch_size
        )

        C = self.num_classes
        # Confusion matrix: cm[true, pred]
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1

        accuracy = float(cm.diagonal().sum() / cm.sum())

        class_names = [label.name for label in Labels]
        per_class = []
        precisions, recalls, f1s = [], [], []

        for c in range(C):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            support = int(cm[c, :].sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            per_class.append(
                {
                    "class":     class_names[c],
                    "precision": precision,
                    "recall":    recall,
                    "f1":        f1,
                    "support":   support,
                    "tp": tp, "fp": fp, "fn": fn,
                }
            )

        return {
            "accuracy":        accuracy,
            "confusion_matrix": cm,
            "classes":         class_names,
            "per_class":       per_class,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall":    float(np.mean(recalls)),
            "macro_f1":        float(np.mean(f1s)),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_tfidf(self, features: torch.Tensor) -> torch.Tensor:
        tf = features / (features.sum() + 1e-8)
        return tf * self.idf

    def _compute_idf(self) -> None:
        """IDF = log(N / (df + 1)), computed with one sparse column-sum."""
        logger.debug("Computing IDF vectors...")
        N = len(self.dataset)
        X: sp.csr_matrix = self.dataset.features_matrix

        # Count documents that contain each token (df per column)
        df = np.asarray((X > 0).sum(axis=0)).squeeze()  # (V,)
        idf_np = np.log(N / (df + 1.0))
        self.idf = torch.from_numpy(idf_np.astype(np.float32))

    def _train(self) -> None:
        """Aggregate TF-IDF word sums per class; no Python loop over samples."""
        logger.debug("Computing TF-IDF and aggregating likelihoods (vectorised)...")
        X_sp: sp.csr_matrix = self.dataset.features_matrix   # (N, V) sparse
        y: np.ndarray = self.dataset.labels                   # (N,)

        # --- TF-IDF in sparse-land ----------------------------------------
        # Row-normalise X to get TF, then multiply each row by IDF.
        row_sums = np.asarray(X_sp.sum(axis=1)).squeeze()     # (N,)
        row_sums = np.where(row_sums == 0, 1e-8, row_sums)

        # Scale each row by 1/row_sum (in sparse form)
        inv_sums = sp.diags(1.0 / row_sums)                   # diagonal (N,N)
        tf_sp = inv_sums @ X_sp                                # (N, V) sparse

        idf_np = self.idf.numpy()
        tfidf_sp = tf_sp.multiply(idf_np).tocsr()             # broadcast (N, V) sparse CSR

        # --- Per-class aggregation ----------------------------------------
        class_counts = np.bincount(y, minlength=self.num_classes).astype(np.float32)

        word_sums = np.zeros((self.num_classes, self.vocab_size), dtype=np.float32)
        for c in range(self.num_classes):
            mask = y == c
            word_sums[c] = np.asarray(tfidf_sp[mask].sum(axis=0)).squeeze()

        total_mass = word_sums.sum(axis=1)  # (C,)

        # --- Priors & smoothed likelihoods --------------------------------
        self.class_priors = torch.from_numpy(class_counts / class_counts.sum())

        likelihoods_np = (word_sums + self.alpha) / (
            total_mass[:, None] + self.alpha * self.vocab_size
        )
        self.likelihoods = torch.from_numpy(likelihoods_np)

        self.log_priors = torch.log(self.class_priors)
        self.log_likelihoods = torch.log(self.likelihoods)

    # ── Persistence ──────────────────────────────────────────────────────────

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
    def load_weights(cls, filepath: Path, dataset: NaiveBayesDataset) -> "NaiveBayesModel":
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
        """Load a trained model for inference without needing the dataset or CSV.

        Returns
        -------
        model:
            Ready-to-use ``NaiveBayesModel``.
        feature_extractor:
            Fitted ``FeatureExtractor`` (vocab + word_to_idx restored from the
            checkpoint) for tokenising new input phrases.
        """
        logger.debug("Loading inference state from %s...", filepath)
        state = torch.load(filepath, weights_only=False)

        # Reconstruct the feature extractor from the checkpoint.
        feature_extractor = FeatureExtractor()
        feature_extractor.vocab = state["vocab"]
        feature_extractor.word_to_idx = state["word_to_idx"]

        # Build a minimal stub so NaiveBayesModel.__init__ can read vocab_size.
        class _InferenceStub:
            labels = None
            features_matrix = None
            def __len__(self) -> int:
                return 0

        stub = _InferenceStub()
        stub.feature_extractor = feature_extractor  # type: ignore[attr-defined]

        model = cls.__new__(cls)
        model.dataset = stub          # type: ignore[assignment]
        model.alpha = 1.0
        model.num_classes = len(Labels)
        model.vocab_size = state["vocab_size"]
        model.class_priors    = state["class_priors"]
        model.likelihoods     = state["likelihoods"]
        model.idf             = state["idf"]
        model.log_priors      = state["log_priors"]
        model.log_likelihoods = state["log_likelihoods"]

        return model, feature_extractor

