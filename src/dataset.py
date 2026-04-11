from __future__ import annotations

import logging
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Final, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset

# Import concrete types so Pydantic v2 can resolve the forward references in
# PreprocessingCFG.  preprocess.py does not import dataset.py, so there is no
# circular dependency.
from preprocess import FeatureExtractor, NLP

logger = logging.getLogger(__name__)

DATASET_PATH: Final[Path] = (
    Path(__file__).parent.parent / "data" / "email-spam-classification-dataset.csv"
)


class PreprocessingCFG(BaseModel):
    should_lemmatize: bool
    should_remove_stopwords: bool
    should_lowercase: bool
    stopwords: Set[str]
    feature_extractor: FeatureExtractor
    nlp: NLP

    model_config = {"arbitrary_types_allowed": True}


class Labels(IntEnum):
    HAM = 0
    SPAM = 1


def preprocess_texts(df: pd.DataFrame, cfg: PreprocessingCFG) -> List[List[str]]:
    """Run the full text pipeline and return tokenized docs.

    Having this as a standalone function lets callers run preprocessing once
    and reuse the resulting docs for both ``FeatureExtractor.fit`` and
    ``NaiveBayesDataset`` construction, avoiding the double-processing bug.
    """
    texts: List[str] = df["text"].tolist()

    if cfg.should_lowercase:
        texts = [t.lower() for t in texts]

    docs: List[List[str]] = [cfg.nlp.tokenize(t) for t in texts]

    if cfg.should_lemmatize:
        docs = [cfg.nlp.lemmatize(d) for d in docs]

    if cfg.should_remove_stopwords:
        sw = cfg.stopwords
        docs = [[tok for tok in d if tok not in sw] for d in docs]

    return docs


def _build_csr(docs: List[List[str]], word_to_idx: dict[str, int], vocab_size: int) -> sp.csr_matrix:
    """Convert tokenized docs to a sparse CSR count matrix (N × vocab_size)."""
    rows, cols, data = [], [], []
    for row_idx, doc in enumerate(docs):
        for word, count in Counter(doc).items():
            if word in word_to_idx:
                rows.append(row_idx)
                cols.append(word_to_idx[word])
                data.append(float(count))
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(docs), vocab_size), dtype=np.float32
    )


class NaiveBayesDataset(Dataset):
    """Email dataset backed by a sparse CSR feature matrix.

    Memory model
    ------------
    All features are stored in a single ``scipy.sparse.csr_matrix``
    (N × vocab_size).  Only non-zero counts are kept, so memory scales with
    the corpus sparsity rather than N × vocab_size.

    ``__getitem__`` converts one sparse row to a dense ``torch.Tensor``
    on demand — at most one row is ever dense in memory at a time.

    Parameters
    ----------
    df:
        Raw DataFrame with ``"text"`` and ``"label"`` columns.
    cfg:
        Preprocessing configuration.  ``cfg.feature_extractor`` must already
        be fitted before constructing this dataset.
    docs:
        Optional pre-tokenized documents (output of ``preprocess_texts``).
        If supplied, the text pipeline is skipped entirely, saving significant
        time when the docs have already been computed (e.g. for the training
        split used to fit the vocabulary).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: PreprocessingCFG,
        docs: Optional[List[List[str]]] = None,
    ) -> None:
        logger.debug("Initializing NaiveBayesDataset with %d rows.", len(df))
        self.cfg = cfg
        self.feature_extractor = cfg.feature_extractor

        self.df = df.reset_index(drop=True)

        # ── Text preprocessing ──────────────────────────────────────────────
        if docs is not None:
            logger.debug("Using pre-tokenized docs (skipping text pipeline).")
            self.docs: List[List[str]] = docs
        else:
            logger.debug("Running text preprocessing pipeline...")
            self.docs = preprocess_texts(df, cfg)

        # ── Sparse CSR feature matrix ────────────────────────────────────────
        logger.debug("Building sparse CSR feature matrix...")
        w2i = self.feature_extractor.word_to_idx
        vocab_size = len(self.feature_extractor.vocab)  # type: ignore[arg-type]
        self.features_matrix: sp.csr_matrix = _build_csr(self.docs, w2i, vocab_size)  # type: ignore[arg-type]
        logger.debug(
            "CSR matrix: shape=%s, stored elements=%d",
            self.features_matrix.shape,
            self.features_matrix.nnz,
        )

        self.labels: np.ndarray = self.df["label"].to_numpy(dtype=np.int64)
        logger.debug("Dataset initialization complete.")

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(
            self.features_matrix[index].toarray().squeeze(0)
        )
        label = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return feature, label
