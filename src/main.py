"""main.py — load a trained NBEC model and classify an input phrase.

Usage
-----
Interactive prompt:
    uv run src/main.py

Pass text directly as a command-line argument:
    uv run src/main.py "Congratulations, you have won a free iPhone!"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,          # suppress library noise; only show warnings+
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)

import torch
from nltk.corpus import stopwords

from model import NaiveBayesModel
from preprocess import NLP

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH: Path = Path(__file__).parent.parent / "data" / "model_weights.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

def classify(text: str, model: NaiveBayesModel, nlp: NLP, sw: set[str]) -> dict:
    """Tokenise *text* and return prediction + confidence scores.

    Returns
    -------
    dict with keys:
        label       : "SPAM" or "HAM"
        confidence  : probability of the predicted class (softmax of log-probs)
        scores      : {"HAM": float, "SPAM": float}  — both class probabilities
    """
    tokens = nlp.tokenize(text.lower())
    tokens = nlp.remove_stopwords(tokens, sw)

    sparse = model.dataset.feature_extractor.extract_features(tokens)  # type: ignore[attr-defined]
    vocab_size = model.vocab_size
    feat = torch.zeros(vocab_size, dtype=torch.float32)
    for idx, val in sparse.items():
        feat[idx] = val

    # Raw log-posteriors → softmax probabilities
    tfidf = model._compute_tfidf(feat)
    log_probs = model.log_priors + (model.log_likelihoods * tfidf).sum(dim=1)
    probs = torch.softmax(log_probs, dim=0)

    pred_idx = int(torch.argmax(probs).item())
    label = "SPAM" if pred_idx == 1 else "HAM"

    return {
        "label":      label,
        "confidence": float(probs[pred_idx].item()),
        "scores":     {"HAM": float(probs[0].item()), "SPAM": float(probs[1].item())},
    }


def print_result(text: str, result: dict) -> None:
    label = result["label"]
    conf  = result["confidence"] * 100
    ham_p = result["scores"]["HAM"]  * 100
    spam_p= result["scores"]["SPAM"] * 100
    bar_len = 30
    filled = round(spam_p / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    print()
    print(f'  Input  : "{text}"')
    print(f"  Result : {label}  ({conf:.1f}% confidence)")
    print(f"  HAM    : {ham_p:5.1f}%  |  SPAM : {spam_p:5.1f}%")
    print(f"  Spam   : [{bar}]")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not WEIGHTS_PATH.exists():
        print(
            f"ERROR: No model weights found at {WEIGHTS_PATH}\n"
            "Run `uv run src/train_model.py` first to train and save the model.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {WEIGHTS_PATH}...", end=" ", flush=True)
    model, feature_extractor = NaiveBayesModel.load_for_inference(WEIGHTS_PATH)
    print("done.")
    print(f"Vocabulary: {feature_extractor.vocab and len(feature_extractor.vocab):,} tokens")

    nlp = NLP()
    sw  = set(stopwords.words("english"))

    # ── Input source: CLI arg or interactive loop ──────────────────────────
    if len(sys.argv) > 1:
        # Single phrase passed as a CLI argument
        text = " ".join(sys.argv[1:])
        result = classify(text, model, nlp, sw)
        print_result(text, result)
    else:
        # Interactive loop
        print("\nType a phrase and press Enter to classify it.")
        print("Type 'quit' or press Ctrl-C to exit.\n")
        while True:
            try:
                text = input("  > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                break

            result = classify(text, model, nlp, sw)
            print_result(text, result)


if __name__ == "__main__":
    main()
