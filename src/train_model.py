"""train_model.py — end-to-end training + full evaluation for the NBEC pipeline.

Run with:
    uv run src/train_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Configure logging before any project imports so all loggers pick it up.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords

from dataset import DATASET_PATH, NaiveBayesDataset, PreprocessingCFG, preprocess_texts
from model import NaiveBayesModel
from preprocess import FeatureExtractor, NLP

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 60) -> str:
    return char * width


def print_metrics(metrics: Dict[str, Any], split: str = "Test") -> None:
    """Pretty-print the full evaluation metrics dict returned by compute_metrics()."""
    classes   = metrics["classes"]
    per_class = metrics["per_class"]
    cm        = metrics["confusion_matrix"]

    print()
    print(_sep("═"))
    print(f"  Evaluation Results — {split} Set")
    print(_sep("═"))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    col_width = 10
    header = " " * 12 + "".join(f"Pred {c:<{col_width - 5}}" for c in classes)
    print(f"\n  Confusion Matrix  (rows = true label, cols = predicted label)\n")
    print("  " + header)
    print("  " + _sep())
    for i, true_cls in enumerate(classes):
        row = f"  True {true_cls:<6} |"
        for j in range(len(classes)):
            row += f"  {cm[i, j]:<{col_width - 2}}"
        print(row)
    print("  " + _sep())

    # ── Per-class table ───────────────────────────────────────────────────────
    print(f"\n  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print("  " + _sep())
    for pc in per_class:
        print(
            f"  {pc['class']:<8} "
            f"{pc['precision']:>10.4f} "
            f"{pc['recall']:>10.4f} "
            f"{pc['f1']:>10.4f} "
            f"{pc['support']:>10d}  "
            f"{pc['tp']:>6d}  "
            f"{pc['fp']:>6d}  "
            f"{pc['fn']:>6d}"
        )
    print("  " + _sep())

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print(f"\n  {'Accuracy':<22}: {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'Macro Precision':<22}: {metrics['macro_precision']:.4f}")
    print(f"  {'Macro Recall':<22}: {metrics['macro_recall']:.4f}")
    print(f"  {'Macro F1':<22}: {metrics['macro_f1']:.4f}")
    print()
    print(_sep("═"))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    nlp = NLP()
    feature_extractor = FeatureExtractor(min_df=1)
    cfg = PreprocessingCFG(
        should_lemmatize=False,
        should_remove_stopwords=True,
        should_lowercase=True,
        stopwords=set(stopwords.words("english")),
        feature_extractor=feature_extractor,
        nlp=nlp,
    )

    # ── Load & split ──────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    logger.info("Splitting 80 / 20 (random_state=42)...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df  = df.drop(train_df.index)
    logger.info("Train: %d samples  |  Test: %d samples", len(train_df), len(test_df))

    # ── Preprocess & fit vocabulary ───────────────────────────────────────────
    logger.info("Preprocessing training data...")
    train_docs = preprocess_texts(train_df, cfg)

    logger.info("Fitting vocabulary (min_df=5)...")
    feature_extractor.fit(train_docs)
    logger.info("Vocabulary size: %d tokens", len(feature_extractor.vocab))  # type: ignore[arg-type]

    # ── Build datasets ────────────────────────────────────────────────────────
    logger.info("Building train dataset (CSR matrix)...")
    train_dataset = NaiveBayesDataset(train_df, cfg, docs=train_docs)

    logger.info("Building test dataset (CSR matrix)...")
    test_dataset = NaiveBayesDataset(test_df, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training Naive Bayes model...")
    model = NaiveBayesModel(train_dataset)
    model.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    weights_path = Path(__file__).parent.parent / "data" / "model_weights.pt"
    logger.info("Saving weights to %s", weights_path)
    model.save_weights(
        filepath=weights_path,
        vocab=feature_extractor.vocab,           # type: ignore[arg-type]
        word_to_idx=feature_extractor.word_to_idx,  # type: ignore[arg-type]
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Loading weights and evaluating on test set...")
    loaded_model = NaiveBayesModel.load_weights(weights_path, test_dataset)

    metrics = loaded_model.compute_metrics()
    print_metrics(metrics, split="Test")

    # ── Also report training-set metrics (bias check) ─────────────────────────
    train_model_eval = NaiveBayesModel.load_weights(weights_path, train_dataset)
    train_metrics = train_model_eval.compute_metrics()
    print_metrics(train_metrics, split="Train")

    # ── Custom input demo ─────────────────────────────────────────────────────
    demo_text = "Congratulations! You have won a free lottery ticket"
    tokens = nlp.tokenize(demo_text.lower())
    if cfg.should_remove_stopwords:
        tokens = nlp.remove_stopwords(tokens, cfg.stopwords)

    sparse = feature_extractor.extract_features(tokens)
    vocab_size = len(feature_extractor.vocab)  # type: ignore[arg-type]
    feat_tensor = torch.zeros(vocab_size, dtype=torch.float32)
    for idx, val in sparse.items():
        feat_tensor[idx] = val

    pred = loaded_model.predict(feat_tensor)
    label = "SPAM" if pred == 1 else "HAM"
    print(f'  Demo prediction: "{demo_text}"  →  {label}')
    print()


if __name__ == "__main__":
    main()
