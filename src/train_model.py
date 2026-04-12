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
    class_names = metrics["class_names"]
    per_class_metrics = metrics["per_class_metrics"]
    confusion_matrix = metrics["confusion_matrix"]

    print()
    print(_sep("="))
    print(f"  Evaluation Results - {split} Set")
    print(_sep("="))

    col_width = 10

    # Confusion matrix header
    header = " " * 12 + "".join(
        f"Pred {class_name:<{col_width - 5}}" for class_name in class_names
    )

    print("  Confusion Matrix (rows = true label, cols = predicted label)\n")
    print("  " + header)
    print("  " + _sep())

    # Confusion matrix rows
    for i, true_class in enumerate(class_names):
        row = f"  True {true_class:<6} |"
        for j in range(len(class_names)):
            row += f"  {int(confusion_matrix[i, j]):<{col_width - 2}}"
        print(row)

    print("  " + _sep())

    # Per-class metrics header
    print(
        f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}  {'TP':>6}  {'FP':>6}  {'FN':>6}"
    )
    print("  " + _sep())

    # Per-class metrics rows
    for class_metrics in per_class_metrics:
        print(
            f"  {class_metrics['class_name']:<12} "
            f"{class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} "
            f"{class_metrics['f1_score']:>10.4f} "
            f"{class_metrics['support']:>10d}  "
            f"{class_metrics['true_positives']:>6d}  "
            f"{class_metrics['false_positives']:>6d}  "
            f"{class_metrics['false_negatives']:>6d}"
        )

    print("  " + _sep())

    # Overall metrics
    print(
        f"\n  {'Accuracy':<22}: {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"  {'Macro Precision':<22}: {metrics['macro_precision']:.4f}")
    print(f"  {'Macro Recall':<22}: {metrics['macro_recall']:.4f}")
    print(f"  {'Macro F1':<22}: {metrics['macro_f1_score']:.4f}")

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

    logger.info("Building train dataset (CSR matrix from preprocessed docs)...")
    train_dataset = NaiveBayesDataset(train_df, feature_extractor, train_docs)

    logger.info("Building test dataset (CSR matrix from preprocessed docs)...")
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

    logger.info("Loading weights and evaluating on test set...")
    loaded_model = NaiveBayesModel.load_weights(weights_path, test_dataset)
    metrics = loaded_model.compute_metrics()
    print_metrics(metrics, split="Test")

    logger.info("Loading weights and evaluating on train set...")
    train_model_eval = NaiveBayesModel.load_weights(weights_path, train_dataset)
    train_metrics = train_model_eval.compute_metrics()
    print_metrics(train_metrics, split="Train")


if __name__ == "__main__":
    main()
