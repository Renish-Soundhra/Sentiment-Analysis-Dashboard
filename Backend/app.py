# Backend/app.py

from fastapi import FastAPI
from routes import analytics

app = FastAPI(title="Sentiment Analysis API")

# Remove the prefix from this line
app.include_router(analytics.router)

@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API"}