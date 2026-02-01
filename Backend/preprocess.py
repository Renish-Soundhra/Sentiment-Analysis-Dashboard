# backend/preprocess.py
import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer

# ensure stopwords are available
nltk.download('stopwords', quiet=True)

# custom stopwords (same set you used earlier)
customStopwords = {
    "product", "amazon", "charger", "device", "adapter", "ssd", "transferring",
    "tablet", "months", "message", "format", "card", "monday", "gb", "battery",
    "os", "samsung", "gs", "folder", "week", "mistake", "number"
}
baseStopwords = set(nltk_stopwords.words("english")) - {"not", "no", "nor", "don", "didn", "wasn", "isn", "aren"}
baseStopwords.update(customStopwords)

ps = PorterStemmer()

def preprocess(text: str) -> str:
    """
    Clean and stem text the same way as training.
    """
    if text is None:
        return ""
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in baseStopwords]
    return " ".join(words)
