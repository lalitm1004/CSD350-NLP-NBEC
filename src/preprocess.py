from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NLP:
    def __init__(self) -> None:
        logger.debug("Initializing NLP tokenizer.")

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    def remove_stopwords(self, doc: List[str], stopwords: set[str]) -> List[str]:
        return [token for token in doc if token not in stopwords]


class FeatureExtractor:
    def __init__(self, min_df: int = 5) -> None:
        logger.debug("Initializing BagOfWords FeatureExtractor (min_df=%d).", min_df)
        self.min_df = min_df
        self.vocab: Optional[List[str]] = None
        self.word_to_idx: Optional[Dict[str, int]] = None

    def fit(self, docs: List[List[str]]) -> None:
        """Fit vocabulary from a list of pre-tokenized documents."""
        logger.info("Fitting FeatureExtractor vocabulary with %d documents.", len(docs))

        doc_freq: Counter[str] = Counter()
        for doc in docs:
            doc_freq.update(set(doc))

        kept = sorted(token for token, df in doc_freq.items() if df >= self.min_df)

        self.vocab = kept
        self.word_to_idx = {word: i for i, word in enumerate(kept)}
        logger.info(
            "Vocabulary fitted. Unique tokens (min_df=%d): %d.",
            self.min_df,
            len(self.vocab),
        )

    def extract_features(self, tokenized_text: List[str]) -> Dict[int, float]:
        if self.vocab is None or self.word_to_idx is None:
            raise ValueError(
                "FeatureExtractor must be fitted before calling extract_features."
            )

        word_counts = Counter(tokenized_text)
        return {
            self.word_to_idx[word]: float(count)
            for word, count in word_counts.items()
            if word in self.word_to_idx
        }
