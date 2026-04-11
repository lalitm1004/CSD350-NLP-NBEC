import torch
import logging
from pathlib import Path
from dataset import NaiveBayesDataset, Labels

logger = logging.getLogger(__name__)


class NaiveBayesModel:
    def __init__(self, dataset: NaiveBayesDataset, alpha: float = 1.0):
        logger.info(f"Initializing NaiveBayesModel with alpha={alpha}")
        self.dataset = dataset
        self.alpha = alpha

        self.num_classes = len(Labels)

        self.vocab_size = len(dataset.feature_extractor.vocab) if dataset.feature_extractor.vocab else 0

        # Model parameters
        self.class_priors = torch.zeros(self.num_classes)
        self.likelihoods = torch.zeros((self.num_classes, self.vocab_size))

        # IDF vector
        self.idf = torch.zeros(self.vocab_size)
        self.log_priors = torch.zeros(self.num_classes)
        self.log_likelihoods = torch.zeros((self.num_classes, self.vocab_size))

    def train(self):
        logger.info("Starting model training procedure...")
        self._compute_idf()
        self._train()
        logger.info("Model training completed.")

    def _compute_idf(self):
        """
        Compute IDF: log(N / (df + 1))
        """
        logger.debug("Computing IDF vectors...")
        N = len(self.dataset)
        df = torch.zeros(self.vocab_size)

        for features, _ in self.dataset:
            df += (features > 0).float()

        self.idf = torch.log(N / (df + 1))

    def _compute_tfidf(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute TF-IDF vector
        """
        # Term frequency (normalized)
        tf = features / (features.sum() + 1e-8)

        return tf * self.idf

    def _train(self):
        """
        Train using TF-IDF features
        """
        logger.debug("Computing TF-IDF matrices over training corpus and aggregating likelihoods...")
        class_counts = torch.zeros(self.num_classes)
        word_sums = torch.zeros((self.num_classes, self.vocab_size))
        total_mass = torch.zeros(self.num_classes)

        for features, label in self.dataset:
            label = int(label.item())

            # Convert to TF-IDF
            tfidf = self._compute_tfidf(features)

            class_counts[label] += 1
            word_sums[label] += tfidf
            total_mass[label] += tfidf.sum()

        # Priors
        self.class_priors = class_counts / class_counts.sum()

        # Likelihoods with smoothing
        for c in range(self.num_classes):
            self.likelihoods[c] = (word_sums[c] + self.alpha) / (
                total_mass[c] + self.alpha * self.vocab_size
            )

        # Log space
        self.log_priors = torch.log(self.class_priors)
        self.log_likelihoods = torch.log(self.likelihoods)

    def predict(self, features: torch.Tensor) -> int:
        """
        Predict class using TF-IDF-weighted NB rule
        """
        tfidf = self._compute_tfidf(features)

        log_probs = self.log_priors.clone()

        for c in range(self.num_classes):
            log_probs[c] += (tfidf * self.log_likelihoods[c]).sum()

        return int(torch.argmax(log_probs).item())

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        preds = []

        for features in X:
            preds.append(self.predict(features))

        return torch.tensor(preds)

    def evaluate(self) -> float:
        correct = 0

        for features, label in self.dataset:
            pred = self.predict(features)
            if pred == int(label.item()):
                correct += 1

        return correct / len(self.dataset)

    def save_weights(self, filepath: Path, vocab: list[str], word_to_idx: dict[str, int]):
        logger.debug(f"Saving model state dictionary to {filepath}...")
        state = {
            "class_priors": self.class_priors,
            "likelihoods": self.likelihoods,
            "idf": self.idf,
            "log_priors": self.log_priors,
            "log_likelihoods": self.log_likelihoods,
            "vocab": vocab,
            "word_to_idx": word_to_idx,
            "vocab_size": self.vocab_size
        }
        torch.save(state, filepath)

    @classmethod
    def load_weights(cls, filepath: Path, dataset: NaiveBayesDataset) -> "NaiveBayesModel":
        logger.debug(f"Loading model state dictionary from {filepath}...")
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
