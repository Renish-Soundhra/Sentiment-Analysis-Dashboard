# backend/routes/analytics.py

from fastapi import APIRouter, HTTPException
from Backend.services.sentiment_service import SentimentService

router = APIRouter()
service = SentimentService()

@router.get("/predict")
def predict_sentiment(text: str):
    """
    Predict sentiment for a single text.
    """
    sentiment = service.predict_text(text)
    return {"sentiment": sentiment}

@router.post("/predict_bulk")  # Removed the /api/ prefix
def predict_bulk_sentiment(texts: list[str]):
    """Predict sentiment for multiple texts and return analytics."""
    cleaned_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned_texts:
        raise HTTPException(status_code=400, detail="Provide at least one non-empty text value.")

    predictions = service.predict_texts(cleaned_texts)
    summary = service.get_sentiment_summary(predictions)
    trends = service.get_sentiment_trends(predictions)
    wordcloud_img = service.generate_wordcloud_image(cleaned_texts)

    return {
        "predictions": predictions,
        "summary": summary,
        "trends": trends,
        "wordcloud": wordcloud_img,
    }
