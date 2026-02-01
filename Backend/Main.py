# Main.py

from fastapi import FastAPI
from routes import analytics # Correct import statement

app = FastAPI()

# Your main app doesn't need to load the models directly now.
# The 'sentiment_service' module will handle that for you.

# Include the router from the analytics module to add the API endpoints
app.include_router(analytics.router)