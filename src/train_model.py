from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from nltk.corpus import stopwords

from dataset import DATASET_PATH, NaiveBayesDataset, PreprocessingCFG, preprocess_texts
from model import NaiveBayesModel
from preprocess import FeatureExtractor, NLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _sep(char: str = "-", width: int = 60) -> str:
    return char * width


def print_metrics(metrics: Dict[str, Any], split: str = "Test") -> None:
    print()
    print(_sep("="))
    print(f"  Evaluation Results - {split} Set")
    print(_sep("="))

    tp = metrics["true_positives"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]
    tn = metrics["true_negatives"]

    total = tp + fp + fn + tn

    print("\n  Confusion Breakdown")
    print("  " + _sep())
    print(f"  TP (Spam correctly predicted) : {tp}")
    print(f"  FP (Ham predicted as Spam)    : {fp}")
    print(f"  FN (Spam predicted as Ham)    : {fn}")
    print(f"  TN (Ham correctly predicted)  : {tn}")
    print("  " + _sep())

    print("\n  Metrics")
    print("  " + _sep())
    print(
        f"  {'Accuracy':<20}: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"  {'Precision':<20}: {metrics['precision']:.4f}")
    print(f"  {'Recall':<20}: {metrics['recall']:.4f}")
    print(f"  {'F1 Score':<20}: {metrics['f1_score']:.4f}")
    print("  " + _sep())

    print("\n  Dataset Stats")
    print("  " + _sep())
    print(f"  Total Samples       : {total}")
    print(f"  Actual Spam         : {tp + fn}")
    print(f"  Actual Ham          : {tn + fp}")
    print(f"  Predicted Spam      : {tp + fp}")
    print(f"  Predicted Ham       : {tn + fn}")
    print("  " + _sep())

    print()
    print(_sep("="))
    print()


def main() -> None:
    nlp = NLP()
    feature_extractor = FeatureExtractor(min_df=1)

    cfg = PreprocessingCFG(
        stopwords=set(stopwords.words("english")),
        feature_extractor=feature_extractor,
        nlp=nlp,
    )

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    logger.info("Splitting 80 / 20 (random_state=42)...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    logger.info("Train: %d samples  |  Test: %d samples", len(train_df), len(test_df))

    logger.info("Preprocessing training data...")
    train_docs = preprocess_texts(train_df, cfg)

    logger.info("Preprocessing test data...")
    test_docs = preprocess_texts(test_df, cfg)

    logger.info("Fitting vocabulary (min_df=%d)...", feature_extractor.min_df)
    feature_extractor.fit(train_docs)

    logger.info("Vocabulary size: %d tokens", len(feature_extractor.vocab))  # type: ignore[arg-type]

    logger.info("Building train dataset...")
    train_dataset = NaiveBayesDataset(train_df, feature_extractor, train_docs)

    logger.info("Building test dataset...")
    test_dataset = NaiveBayesDataset(test_df, feature_extractor, test_docs)

    logger.info("Training Naive Bayes model...")
    model = NaiveBayesModel(train_dataset)
    model.train()

    weights_path = Path(__file__).parent.parent / "data" / "model_weights.pt"

    logger.info("Saving weights to %s", weights_path)
    model.save_weights(
        filepath=weights_path,
        vocab=feature_extractor.vocab,  # type: ignore[arg-type]
        word_to_idx=feature_extractor.word_to_idx,  # type: ignore[arg-type]
    )

    logger.info("Evaluating on test set...")
    loaded_model = NaiveBayesModel.load_weights(weights_path, test_dataset)
    metrics = loaded_model.compute_metrics()
    print_metrics(metrics, split="Test")

    logger.info("Evaluating on train set...")
    train_model_eval = NaiveBayesModel.load_weights(weights_path, train_dataset)
    train_metrics = train_model_eval.compute_metrics()
    print_metrics(train_metrics, split="Train")


if __name__ == "__main__":
    main()
