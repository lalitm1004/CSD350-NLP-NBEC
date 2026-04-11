from typing import List, Optional, Dict
from collections import Counter
import pandas as pd
import torch
import nltk
import logging

logger = logging.getLogger(__name__)

from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Dataset downloads
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class NLP:
    def __init__(self):
        logger.debug("Initializing SimpleNLP tokenizer & lemmatizer.")
        self._lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def lemmatize(self, doc: List[str]) -> List[str]:
        return [self._lemmatizer.lemmatize(token) for token in doc]


    def remove_stopwords(self, doc: List[str], stopwords: set[str]) -> List[str]:
        return [token for token in doc if token not in stopwords]

class FeatureExtractor:
    def __init__(self):
        logger.debug("Initializing BagOfWords FeatureExtractor.")
        self.vocab: Optional[List[str]] = None
        self.word_to_idx: Optional[Dict[str, int]] = None

    def fit(self, df: pd.DataFrame, nlp_pipeline):
        logger.info(f"Fitting FeatureExtractor vocabulary with {len(df)} records.")
        vocab = set()
        
        # Preprocess to get valid tokens
        for text in df["text"]:
            if nlp_pipeline.should_lowercase:
                text = text.lower()
            tokens = nlp_pipeline.nlp.tokenize(text)
            if nlp_pipeline.should_lemmatize:
                tokens = nlp_pipeline.nlp.lemmatize(tokens)
            if nlp_pipeline.should_remove_stopwords:
                tokens = nlp_pipeline.nlp.remove_stopwords(tokens, nlp_pipeline.stopwords)
            vocab.update(tokens)

        self.vocab = sorted(vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        logger.info(f"Vocabulary fitted. Total unique tokens: {len(self.vocab)}.")

    def extract_features(self, tokenized_text: List[str]) -> torch.Tensor:
        if self.vocab is None or self.word_to_idx is None:
            raise ValueError("Feature extractor must be fitted before extracting features.")

        vocab_size = len(self.vocab)
        feature_vector = torch.zeros(vocab_size, dtype=torch.float32)

        word_counts = Counter(tokenized_text)

        for word, count in word_counts.items():
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                feature_vector[idx] = count

        return feature_vector
