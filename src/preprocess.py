from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Dict, List, Optional

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# NLTK downloads (only used if lemmatization is enabled)
nltk.download("wordnet", quiet=False)
nltk.download("stopwords", quiet=False)


class NLP:
    """Lightweight NLP pipeline.

    Tokenization uses a fast regex split rather than NLTK's Punkt-based
    word_tokenize, which runs a full sentence tokenizer and is ~40x slower
    for plain BoW/TF-IDF use-cases.
    """

    def __init__(self) -> None:
        logger.debug("Initializing NLP tokenizer & lemmatizer.")
        self._lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> List[str]:
        """Regex word tokenizer — finds all alphanumeric tokens."""
        return re.findall(r"\b\w+\b", text)

    def lemmatize(self, doc: List[str]) -> List[str]:
        return [self._lemmatizer.lemmatize(token) for token in doc]

    def remove_stopwords(self, doc: List[str], stopwords: set[str]) -> List[str]:
        return [token for token in doc if token not in stopwords]


class FeatureExtractor:
    """Bag-of-Words feature extractor backed by a sparse representation.

    Key design decisions
    --------------------
    - ``fit`` now accepts a column of already-tokenized & filtered docs
      (``List[List[str]]``) so we never repeat the preprocessing pipeline.
    - ``min_df`` prunes tokens that appear in fewer than *min_df* documents,
      dramatically shrinking a 283 k raw-vocab down to a manageable size.
    - ``extract_features`` returns a ``Dict[int, float]`` (sparse) instead of
      a dense ``torch.Tensor``, eliminating the OOM crash.
    """

    def __init__(self, min_df: int = 5) -> None:
        logger.debug("Initializing BagOfWords FeatureExtractor (min_df=%d).", min_df)
        self.min_df = min_df
        self.vocab: Optional[List[str]] = None
        self.word_to_idx: Optional[Dict[str, int]] = None

    def fit(self, docs: List[List[str]]) -> None:
        """Fit vocabulary from a list of pre-tokenized documents.

        Parameters
        ----------
        docs:
            Each element is an already-tokenized (and filtered) document,
            i.e. the ``"doc"`` column produced by ``NaiveBayesDataset``.
        """
        logger.info("Fitting FeatureExtractor vocabulary with %d documents.", len(docs))

        doc_freq: Counter[str] = Counter()
        for doc in docs:
            doc_freq.update(set(doc))  # count each token once per document

        # Apply min_df threshold
        kept = sorted(token for token, df in doc_freq.items() if df >= self.min_df)

        self.vocab = kept
        self.word_to_idx = {word: i for i, word in enumerate(kept)}
        logger.info(
            "Vocabulary fitted. Unique tokens (min_df=%d): %d.", self.min_df, len(self.vocab)
        )

    def extract_features(self, tokenized_text: List[str]) -> Dict[int, float]:
        """Return a sparse feature dict {vocab_idx: count}.

        Using a dict instead of a dense tensor means a document that touches
        k distinct vocabulary tokens uses O(k) memory rather than O(|vocab|).
        """
        if self.vocab is None or self.word_to_idx is None:
            raise ValueError("FeatureExtractor must be fitted before calling extract_features.")

        word_counts = Counter(tokenized_text)
        return {
            self.word_to_idx[word]: float(count)
            for word, count in word_counts.items()
            if word in self.word_to_idx
        }
