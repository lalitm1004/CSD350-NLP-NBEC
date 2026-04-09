from __future__ import annotations

from abc import ABC
from enum import IntEnum
from pathlib import Path
from typing import cast, List, Set, Type

import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset


class NLP(ABC):
    @staticmethod
    def tokenize(text: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def lemmatize(doc: List[str]) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def remove_stopwords(doc: List[str], stopwords: Set[str]) -> List[str]:
        return [token for token in doc if token not in stopwords]


class FeatureExtractor(ABC):
    @staticmethod
    def extract_features(tokenized_text: List[str], dataset: NBDataset) -> torch.Tensor:
        raise NotImplementedError


class PreprocessingCFG(BaseModel):
    should_lemmatize: bool
    should_remove_stopwords: bool
    stopwords: Set[str]
    nlp: Type[NLP]


class Labels(IntEnum):
    SPAM = 0
    HAM = 1


class NBDataset(Dataset):
    def __init__(self, dataset_path: Path, cfg: PreprocessingCFG, fe: FeatureExtractor):
        self.cfg = cfg
        self.fe = fe
        self.df = pd.read_csv(dataset_path)

        self.df["text"] = self.df["text"].str.lower()

        self.df["doc"] = self.df["text"].map(
            lambda x: self.cfg.nlp.tokenize(cast(str, x))
        )

        if self.cfg.should_lemmatize:
            self.df["doc"] = self.df["doc"].map(
                lambda x: cfg.nlp.lemmatize(cast(List[str], x))
            )

        if self.cfg.should_remove_stopwords:
            self.df["doc"] = self.df["doc"].map(
                lambda x: cfg.nlp.remove_stopwords(
                    cast(List[str], x), self.cfg.stopwords
                )
            )

        self.df["features"] = self.df["tokenized_text"].map(
            lambda x: self.fe.extract_features(cast(List[str], x), self)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pass
