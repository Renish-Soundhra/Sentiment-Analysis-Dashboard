# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from fetch_data import get_comments

# The backend URL should not have the /api prefix, as it is added by the FastAPI app router.
BACKEND_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze sentiments from a file, tweet, Reddit post, or video/reel link.")

with st.sidebar:
    st.header("Select Input Type")
    input_type = st.radio("Input type", ("File", "Reddit", "Video/Reel"))

    source = None
    if input_type == "File":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        source = uploaded_file
    else:
        source = st.text_input(f"Enter {input_type} URL or link")

    analyze_btn = st.button("Analyze")

if analyze_btn:
    comments = []

    try:
        if input_type == "File":
            if uploaded_file is None:
                st.warning("Please upload a file first.")
                st.stop()
            with st.spinner("Reading file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if 'reviewText' not in df.columns:
                    st.error("File must contain a 'reviewText' column.")
                    st.stop()
                comments = df['reviewText'].dropna().tolist()
        else:
            if not source:
                st.warning(f"Please enter a {input_type} link.")
                st.stop()
            with st.spinner(f"Fetching comments from {input_type}..."):
                type_map = {"Tweet": "tweet", "Reddit": "reddit", "Video/Reel": "video"}
                comments = get_comments(source, type_map[input_type])

        if not comments:
            st.warning("No comments found.")
            st.stop()

        st.success(f"Fetched {len(comments)} comments. Sending to API...")

        # Call backend
        try:
            # CORRECTED: Added '/api' to the URL to match the FastAPI router prefix.
            resp = requests.post(
                f"{BACKEND_URL}/predict_bulk", 
                json=comments, 
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            predictions = data.get("predictions", [])
            summary = data.get("summary", {})
            trends = data.get("trends", {})
            wordcloud_img = data.get("wordcloud", "")

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            # Removed st.stop() to allow the page to show the error
        except Exception as e:
            st.error(f"Error calling backend: {e}")
            # Removed st.stop() to allow the page to show the error

        if predictions:
            df_result = pd.DataFrame({"ReviewText": comments, "Sentiment": predictions})

            st.subheader("Sentiment distribution")
            dist = df_result['Sentiment'].value_counts(normalize=True).reset_index()
            dist.columns = ['Sentiment', 'Percentage']
            dist['Percentage'] = (dist['Percentage'] * 100).round(2)
            fig = px.pie(dist, names='Sentiment', values='Percentage', title="Sentiment Distribution (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sentiment counts")
            count_dist = df_result['Sentiment'].value_counts().reset_index()
            count_dist.columns = ['Sentiment', 'Count']
            fig2 = px.bar(count_dist, x='Sentiment', y='Count', text='Count', title="Counts per Sentiment")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Sample data (first 20 rows)")
            st.dataframe(df_result.head(20))

            st.subheader("Top words by sentiment (first 50 words)")
            for sentiment in ["Positive", "Negative", "Neutral"]:
                words = df_result[df_result.Sentiment == sentiment]['ReviewText'].str.cat(sep=' ').split()[:50]
                st.write(f"{sentiment} sample words:", " ".join(words))

            # st.subheader("Wordcloud (placeholder)")
            # st.text(wordcloud_img)

            import base64
            from io import BytesIO
            from PIL import Image

            st.subheader("Wordcloud")

            if wordcloud_img:
                image_bytes = base64.b64decode(wordcloud_img)
                image = Image.open(BytesIO(image_bytes))
                st.image(image, use_container_width=True)
            else:
                st.warning("No wordcloud generated.")


            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, file_name="sentiment_results.csv", mime="text/csv")
        else:
            st.error("No sentiment results received from the backend.")
            # st.stop() # Removed this line

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Select an input type, provide the source, and click Analyze.")