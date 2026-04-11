import pandas as pd
import logging
import sys

# Configure tracing / logging to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from dataset import NaiveBayesDataset, PreprocessingCFG, DATASET_PATH
from preprocess import NLP, FeatureExtractor
from model import NaiveBayesModel

from nltk.corpus import stopwords

# =========================
# Config
# =========================
nlp = NLP()
feature_extractor = FeatureExtractor()

cfg = PreprocessingCFG(
    should_lemmatize=False,
    should_remove_stopwords=True,
    should_lowercase=True,
    stopwords=set(stopwords.words("english")),
    feature_extractor=feature_extractor,
    nlp=nlp,
)

# =========================
# Load Dataset & Split
# =========================
logger.info("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Train/Test split of 80/20
logger.info("Splitting dataset 80/20...")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

logger.info(f"Train size: {len(train_df)}")
logger.info(f"Test size: {len(test_df)}")

logger.info("Fitting feature extractor on training data...")
cfg.feature_extractor.fit(train_df, cfg)

logger.info("Preparing Train and Test Datasets...")
train_dataset = NaiveBayesDataset(train_df, cfg)
test_dataset = NaiveBayesDataset(test_df, cfg)

# =========================
# Train Model
# =========================
logger.info("Training model...")
model = NaiveBayesModel(train_dataset)
model.train()
logger.info("Model trained")

# =========================
# Save Weights
# =========================
weights_path = Path("model_weights.pt")
logger.info(f"Saving model weights to {weights_path}...")
model.save_weights(
    filepath=weights_path, 
    vocab=cfg.feature_extractor.vocab, 
    word_to_idx=cfg.feature_extractor.word_to_idx
)

# =========================
# Load Weights & Evaluate
# =========================
logger.info(f"Loading model weights from {weights_path}...")
loaded_model = NaiveBayesModel.load_weights(weights_path, test_dataset)

logger.info("Evaluating on test set...")
accuracy = loaded_model.evaluate()
logger.info(f"Accuracy: {accuracy:.4f}")

# =========================
# Test Custom Input
# =========================
text = "Congratulations! You have won a free lottery ticket"
logger.debug(f"Testing custom input: {text}")

tokens = nlp.tokenize(text.lower())
tokens = nlp.lemmatize(tokens)
tokens = nlp.remove_stopwords(tokens, cfg.stopwords)

features = feature_extractor.extract_features(tokens)

pred = loaded_model.predict(features)
logger.info(f"Prediction for custom input: {'SPAM' if pred == 1 else 'HAM'}")
