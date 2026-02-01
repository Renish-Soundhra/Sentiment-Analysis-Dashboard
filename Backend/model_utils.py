# backend/model_utils.py

import joblib
from pathlib import Path
from typing import List, Union
import logging
# Import the preprocess function from your separate file
from preprocess import preprocess
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths to artifacts
# The corrected path
BASE_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# Global variables for artifacts
model, vectorizer = None, None


def load_artifacts():
    """
    Load ML artifacts: model and vectorizer.
    Raises FileNotFoundError if any artifact is missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")

    global model, vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    logging.info(f"Model and vectorizer loaded successfully from {BASE_DIR}")


def predict_single(text: str) -> str:
    """
    Predict sentiment for a single text input.
    """
    if not model or not vectorizer:
        raise RuntimeError("Model artifacts are not loaded. Call load_artifacts() first.")

    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return pred


def predict_bulk(texts: List[str]) -> List[str]:
    """
    Predict sentiment for a list of text inputs.
    """
    if not model or not vectorizer:
        raise RuntimeError("Model artifacts are not loaded. Call load_artifacts() first.")

    cleaned_texts = [preprocess(t) for t in texts]
    X = vectorizer.transform(cleaned_texts)
    preds = model.predict(X).tolist()
    return preds


def predict(input_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Wrapper function to predict sentiment for single or multiple inputs.
    """
    if isinstance(input_text, str):
        return predict_single(input_text)
    elif isinstance(input_text, list):
        return predict_bulk(input_text)
    else:
        raise ValueError("Input must be a string or list of strings")