from __future__ import annotations

import logging
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Final, List, Set, Tuple

import pandas as pd
import scipy.sparse as sp
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset

from preprocess import FeatureExtractor, NLP

logger = logging.getLogger(__name__)

DATASET_PATH: Final[Path] = (
    Path(__file__).parent.parent / "data" / "email-spam-classification-dataset.csv"
)


class PreprocessingCFG(BaseModel):
    stopwords: Set[str]
    feature_extractor: FeatureExtractor
    nlp: NLP

    model_config = {"arbitrary_types_allowed": True}


class Labels(IntEnum):
    HAM = 0
    SPAM = 1


def preprocess_texts(df: pd.DataFrame, cfg: PreprocessingCFG) -> List[List[str]]:
    texts: List[str] = df["text"].tolist()
    docs: List[List[str]] = [cfg.nlp.tokenize(t.lower()) for t in texts]
    sw = cfg.stopwords
    return [[tok for tok in d if tok not in sw] for d in docs]


def _build_csr(
    docs: List[List[str]], word_to_idx: dict[str, int], vocab_size: int
) -> sp.csr_matrix:
    """Convert tokenized docs to a sparse CSR count matrix (N x vocab_size)."""
    rows, cols, data = [], [], []
    for row_idx, doc in enumerate(docs):
        for word, count in Counter(doc).items():
            if word in word_to_idx:
                rows.append(row_idx)
                cols.append(word_to_idx[word])
                data.append(float(count))
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(docs), vocab_size), dtype="float32"
    )


class NaiveBayesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        docs: List[List[str]],
    ) -> None:
        logger.debug("Initializing NaiveBayesDataset with %d rows.", len(df))
        if len(df) != len(docs):
            raise ValueError("docs length must match dataframe rows.")

        self.feature_extractor = feature_extractor
        self.df = df.reset_index(drop=True)
        self.docs = docs

        logger.debug("Building sparse CSR feature matrix...")
        w2i = self.feature_extractor.word_to_idx
        vocab_size = len(self.feature_extractor.vocab)  # type: ignore[arg-type]
        self.features_matrix: sp.csr_matrix = _build_csr(self.docs, w2i, vocab_size)  # type: ignore[arg-type]
        logger.debug(
            "CSR matrix: shape=%s, stored elements=%d",
            self.features_matrix.shape,
            self.features_matrix.nnz,
        )

        self.labels = torch.tensor(self.df["label"].tolist(), dtype=torch.int64)
        logger.debug("Dataset initialization complete.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.tensor(
            self.features_matrix[index].toarray(), dtype=torch.float32
        ).squeeze(0)
        label = self.labels[index]
        return feature, label
