# backend/config.py

from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to model artifacts
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# FastAPI settings
APP_TITLE = "Sentiment Analysis API"
APP_DESCRIPTION = "API for predicting sentiment of text using ML model"
APP_VERSION = "1.0.0"

# Other configuration (optional)
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Example: Database config if needed in future
# DB_URL = "mongodb://localhost:27017"
# DB_NAME = "sentiment_db"
