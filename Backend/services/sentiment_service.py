# Backend/services/sentiment_service.py

# Import the prediction functions from your new model_utils file
from model_utils import predict_single, predict_bulk, load_artifacts
from collections import Counter
# You might also need other imports for your summary/trends functions
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# It is good practice to load artifacts at the start of your application
try:
    load_artifacts()
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Handle the error appropriately, perhaps by exiting or logging a critical error.
    # The API endpoints will not work without the models.


class SentimentService:
    def predict_text(self, text: str):
        # Directly call the single prediction function from model_utils
        return predict_single(text)

    def predict_texts(self, texts: list[str]):
        # Directly call the bulk prediction function from model_utils
        return predict_bulk(texts)

    # The rest of your summary/trends functions remain here
    # as they are not model-specific utility functions.

    def get_sentiment_summary(self, predictions: list[str]):
        # The logic here is correct, as it processes the predictions
        counts = Counter(predictions)
        total = len(predictions)
        if total == 0:
            return {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        summary = {
            "Positive": (counts.get("Positive", 0) / total) * 100,
            "Negative": (counts.get("Negative", 0) / total) * 100,
            "Neutral": (counts.get("Neutral", 0) / total) * 100
        }
        return summary
    
    def get_sentiment_trends(self, predictions: list[str]):
        # Example implementation for trends
        counts = Counter(predictions)
        trends = [{"sentiment": s, "count": c} for s, c in counts.items()]
        return trends
        
    def generate_wordcloud_image(self, texts: list[str]):
        if not texts:
            return ""

        combined_text = " ".join(texts)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200
        ).generate(combined_text)

        # Save image to memory
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(img_buffer, format="png")
        plt.close()

        img_buffer.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
        return img_base64