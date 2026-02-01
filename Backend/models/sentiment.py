import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import emoji

POSITIVE_EMOJIS = {
    "😀","😃","😄","😁","😊","😍","🥰","😎","🔥","💯","👍","👏","❤️","😂"
}

NEGATIVE_EMOJIS = {
    "😡","🤬","😠","😢","😭","💔","👎","😞","😤","😩","😫"
}

NEUTRAL_EMOJIS = {
    "😐","😑","🤔","🙄"
}


df = pd.read_csv("amazon_reviews.csv")
df = df[['reviewText', 'overall']].dropna()

def convertRating(score):
    if score <= 2:
        return "Negative"
    elif score in [3, 4]:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["overall"].apply(convertRating)

dfPositive = df[df.sentiment == "Positive"]
dfNeutral = df[df.sentiment == "Neutral"]
dfNegative = df[df.sentiment == "Negative"]

minCount = 3000
dfPositivedown = resample(dfPositive, replace=False, n_samples=minCount, random_state=42)
dfNeutralup = resample(dfNeutral, replace=True, n_samples=minCount, random_state=42)
dfNegativeup = resample(dfNegative, replace=True, n_samples=minCount, random_state=42)

df = pd.concat([dfPositivedown, dfNeutralup, dfNegativeup]).sample(frac=1, random_state=42).reset_index(drop=True)

nltk.download('stopwords')

customStopwords = {
    "product", "amazon", "charger", "device", "adapter", "ssd", "transferring", "tablet", "months", "message", "format",
    "card", "monday", "gb", "battery", "os", "samsung", "gs", "folder", "week", "mistake", "number"
}
baseStopwords = set(nltk_stopwords.words("english")) - {"not", "no", "nor", "don", "didn", "wasn", "isn", "aren"}
baseStopwords.update(customStopwords)

ps = PorterStemmer()

def replace_emojis(text):
    result = []
    for ch in text:
        if ch in POSITIVE_EMOJIS:
            result.append(" EMO_POS ")
        elif ch in NEGATIVE_EMOJIS:
            result.append(" EMO_NEG ")
        elif ch in NEUTRAL_EMOJIS:
            result.append(" EMO_NEU ")
        else:
            result.append(ch)
    return "".join(result)

# def preprocess(text):
#     text = re.sub("[^a-zA-Z]", " ", text)
#     text = text.lower()
#     words = text.split()
#     words = [ps.stem(word) for word in words if word not in baseStopwords]
#     return ' '.join(words)

def preprocess(text):
    text = text.lower()

    # 🔥 ADD THIS LINE
    text = replace_emojis(text)

    text = re.sub("[^a-zA-Z_ ]", " ", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in baseStopwords]
    return ' '.join(words)


corpus = df['reviewText'].apply(preprocess)

cv = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = cv.fit_transform(corpus).toarray()
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

voting_clf = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr), ('rf', rf)],
    voting='soft',
    weights=[1, 2, 2]
)

voting_clf.fit(X_train, y_train)
voting_preds = voting_clf.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, voting_preds))
print(confusion_matrix(y_test, voting_preds))

def predict_sentiment(text):
    cleaned = preprocess(text)
    vec = cv.transform([cleaned])
    return voting_clf.predict(vec)[0]

samples = [
    "I am highly disappointed. The quality is awful, and it came with missing parts. I had to return it immediately. I won’t be purchasing from this brand again.",
    "I had a terrible experience with this product. It stopped working after just a few days, and the customer service was completely unhelpful. I regret buying it.",
    "The product looked promising, but it failed miserably. It didn’t meet any of the expectations and felt cheap and unreliable. Definitely not worth the money.",
    "The product works as expected, but there’s nothing special about it. It gets the job done, but I wouldn’t go out of my way to recommend it to others.",
    "It’s okay. The quality is decent, and the delivery was on time. However, it doesn’t offer anything new compared to similar products in the market.",
    "I didn’t have any major problems with it, but I also didn’t find it particularly impressive. A standard product that serves its purpose.",
    "I’m absolutely thrilled with this product. The quality exceeded my expectations, and it works flawlessly. I’ve already recommended it to my friends!",
    "Fantastic experience! The product arrived early, was well-packaged, and works better than I hoped. Great value for the price. Will buy again!",
    "This product is amazing. It’s well-built, performs reliably, and has made my daily tasks so much easier. I love it and would purchase it again without hesitation.",
]

for sample in samples:
    print(f"\nReview: '{sample}'\n Predicted Sentiment: {predict_sentiment(sample)}")
import joblib
import os

# Create the directory where the models will be saved
# This ensures the directory exists before you try to save files to it
model_dir = os.path.join("Backend", "models")
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and the vectorizer
joblib.dump(voting_clf, os.path.join(model_dir, "sentiment_model.pkl"))
joblib.dump(cv, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("Model and vectorizer saved successfully to the 'Backend/models' directory.")