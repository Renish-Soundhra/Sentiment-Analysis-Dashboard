# 📊 SentimentScope
### Multi-Source Social Media Sentiment Analysis Dashboard

SentimentScope is a **full-stack sentiment analysis platform** that analyzes **real-world social media data** from multiple sources such as **uploaded files, video/reels comment sections, and Reddit threads**, and presents insights through an interactive dashboard.

This project is designed to handle **noisy, informal, short-text data** commonly found on social platforms.

---

## 🚀 Key Features

- 🔹 **Multi-source sentiment analysis**
  - CSV / text file uploads
  - Video / reels comment sections
  - Reddit comment threads
- 🔹 Designed for **real-world social media text**
  - Slang, emojis, abbreviations
  - Short and informal comments
- 🔹 **Machine Learning pipeline**
  - Text preprocessing
  - TF-IDF vectorization
  - Trained sentiment classification model
- 🔹 **Full-stack architecture**
  - Backend services & API routes
  - Interactive Streamlit frontend
- 🔹 Sentiment distribution & analytics visualization
- 🔹 Modular, scalable, and clean project structure

---

## 🧠 Tech Stack

### Backend
- Python
- Modular backend architecture
- Scikit-learn
- Pandas, NumPy
- TF-IDF Vectorizer

### Frontend
- Streamlit
- Python-based UI components
- Data visualization utilities

---

## 📂 Project Structure

```text
├── Backend/
│   ├── models/
│   │   ├── sentiment.py
│   │   ├── model_utils.py
│   │   └── preprocess.py
│   ├── routes/
│   │   └── analytics.py
│   ├── services/
│   │   └── sentiment_service.py
│   ├── app.py
│   ├── Main.py
│   └── config.py
│
├── Frontend/
│   ├── streamlit_app.py
│   ├── fetch_data.py
│   └── display_utils.py
│
├── data/
│   └── uploaded_data/
│
├── requirements.txt
├── README.md
└── .env

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/sentimentscope.git
cd sentimentscope

2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

3️⃣ Install dependencies
```bash
pip install -r requirements.txt

▶️ Running the Application

Start Backend
```bash
python Backend/Main.py

Start Frontend
```bash
streamlit run Frontend/streamlit_app.py

📈 Workflow

User provides input via:

File upload

Uploaded files can use `reviewText`, `text`, `tweet`, `tweet_text`, `content`,
`body`, `post`, `message`, or `comment` columns. Xquik CSV exports work through
the same file-upload flow when one of those text columns is present.

Video / Reels comment sections

Reddit comment threads

Text is cleaned and preprocessed

TF-IDF features are generated

Sentiment classifier predicts:

Positive

Neutral

Negative

Results are visualized on the dashboard

🔐 Model & Data Handling

Trained model (.pkl) files are not committed to GitHub

This ensures:

Clean repository

Reproducibility

Better version control practices

The repository contains the complete preprocessing and inference pipeline, allowing the model to be retrained easily.

🎯 Applications

Social media sentiment tracking

Audience reaction analysis

Brand & product feedback monitoring

Opinion mining for research

Content performance insights

🚧 Future Enhancements

Real-time sentiment streaming

Transformer-based models (BERT / RoBERTa)

Emotion classification

Platform-specific analytics

Cloud deployment (AWS / GCP)

👤 Author

Renish Soundhra S
B.Tech – Artificial Intelligence & Data Science

⭐ If you find this project useful, feel free to star the repository!
