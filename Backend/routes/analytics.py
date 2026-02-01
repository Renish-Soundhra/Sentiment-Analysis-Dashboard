# backend/routes/analytics.py

from fastapi import APIRouter
from services.sentiment_service import SentimentService

router = APIRouter()
service = SentimentService()

@router.get("/predict")
def predict_sentiment(text: str):
    """
    Predict sentiment for a single text.
    """
    sentiment = service.predict_text(text)
    return {"sentiment": sentiment}

@router.post("/predict_bulk") # Removed the /api/ prefix
def predict_bulk_sentiment(texts: list[str]):
    """
    Predict sentiment for multiple texts and return analytics.
    """
    predictions = service.predict_texts(texts)
    summary = service.get_sentiment_summary(predictions) # Changed to use predictions
    trends = service.get_sentiment_trends(predictions)   # Changed to use predictions
    wordcloud_img = service.generate_wordcloud_image(texts)
    
    return {
        "predictions": predictions,
        "summary": summary,
        "trends": trends,
        "wordcloud": wordcloud_img
    }