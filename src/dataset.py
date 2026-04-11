from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import cast, Final, List, Set, Tuple

import pandas as pd
import torch
import logging
from pydantic import BaseModel
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATASET_PATH: Final[Path] = Path(__file__).parent.parent / "data" / "email-spam-classification-dataset.csv"


class PreprocessingCFG(BaseModel):
    should_lemmatize: bool
    should_remove_stopwords: bool
    should_lowercase: bool
    stopwords: Set[str]
    feature_extractor: 'FeatureExtractor'
    nlp: 'NLP'

    class Config:
        arbitrary_types_allowed = True


class Labels(IntEnum):
    HAM = 0
    SPAM = 1


class NaiveBayesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: PreprocessingCFG):
        logger.debug(f"Initializing NaiveBayesDataset with {len(df)} rows.")
        self.cfg = cfg
        self.feature_extractor = self.cfg.feature_extractor
        self.df = df.copy().reset_index(drop=True)

        if self.cfg.should_lowercase:
            logger.debug("Lowercasing text...")
            self.df["text"] = self.df["text"].str.lower()

        logger.debug("Tokenizing text...")
        self.df["doc"] = self.df["text"].map(
            lambda x: self.cfg.nlp.tokenize(cast(str, x))
        )

        if self.cfg.should_lemmatize:
            logger.debug("Lemmatizing docs...")
            self.df["doc"] = self.df["doc"].map(
                lambda x: cfg.nlp.lemmatize(cast(List[str], x))
            )

        if self.cfg.should_remove_stopwords:
            logger.debug("Removing stopwords...")
            self.df["doc"] = self.df["doc"].map(
                lambda x: cfg.nlp.remove_stopwords(
                    cast(List[str], x), self.cfg.stopwords
                )
            )

        logger.debug("Extracting numerical features...")
        self.df["features"] = self.df["doc"].map(
            lambda x: self.feature_extractor.extract_features(cast(List[str], x))
        )
        logger.debug("Dataset initialization complete.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]

        feature = row["features"]

        label = int(row["label"])
        label = torch.tensor(label, dtype=torch.int64)

        return (feature, label)
