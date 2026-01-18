import streamlit as st
import pandas as pd
import re
from ntscraper import Nitter  # Replaced broken snscrape
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px

# ----------------------------
# 1. Scraper Setup
# ----------------------------
@st.cache_resource
def get_nitter_scraper():
    return Nitter(log_level=1)

scraper = get_nitter_scraper()

# ----------------------------
# 2. Page Config
# ----------------------------
st.set_page_config(page_title="Kelantan Social Sentiment Dashboard", layout="wide")
st.title("📊 Real‑Time Sentiment Analysis Dashboard — Kelantan")

# ----------------------------
# 3. Model Training (Fixes KeyError: 'text')
# ----------------------------
@st.cache_data
def load_and_train():
    try:
        df_train = pd.read_csv("prototaip.csv")
        
        # KEY REPAIR: Automatically detect the correct column
        # This prevents the KeyError 'text' if your CSV uses different names
        possible_text_cols = ['text', 'Tweet', 'content', 'mesej', 'Text']
        text_col = next((c for c in possible_text_cols if c in df_train.columns), None)
        
        possible_label_cols = ['label', 'sentiment', 'Sentiment', 'Label']
        label_col = next((c for c in possible_label_cols if c in df_train.columns), None)

        if not text_col or not label_col:
            st.error(f"Could not find required columns. Found: {list(df_train.columns)}")
            st.stop()

        def clean_text(text):
            if not isinstance(text, str): return ""
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
            text = re.sub(r"[^a-z\s']", "", text)
            return text.strip()

        df_train["clean_text"] = df_train[text_col].apply(clean_text)
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(df_train["clean_text"])
        y_train = df_train[label_col]

        model = MultinomialNB()
        model.fit(X_train, y_train)
        return vectorizer, model, clean_text
    except FileNotFoundError:
        st.error("File 'prototaip.csv' not found. Please upload it to your repository.")
        st.stop()

vectorizer, model, clean_text = load_and_train()

# ----------------------------
# 4. Twitter Scrape (Fixes AttributeError)
# ----------------------------
def scrape_kelantan_tweets(limit):
    query = "Kelantan OR 'orang kelantan'"
    # Nitter scraper works without API keys
    scraped_data = scraper.get_tweets(query, mode='term', number=limit, language='ms')
    
    tweets_list = []
    for t in scraped_data['tweets']:
        tweets_list.append({"date": t['date'], "tweet": t['text']})
    
    df = pd.DataFrame(tweets_list)
    if not df.empty:
        # Clean up the Nitter date format
        df['date'] = pd.to_datetime(df['date'].str.replace('·', '').strip(), errors='coerce')
    return df

# ----------------------------
# 5. Dashboard UI
# ----------------------------
tweet_limit = st.sidebar.slider("Number of Tweets", 10, 100, 30)

if st.button("Refresh Sentiment Now"):
    with st.spinner("Fetching data..."):
        df_tweets = scrape_kelantan_tweets(tweet_limit)

    if not df_tweets.empty:
        # Prediction
        df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
        df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))
        
        # Display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Recent Tweets")
            st.dataframe(df_tweets[["date", "tweet", "sentiment"]], use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment Distribution")
            fig_pie = px.pie(df_tweets, names="sentiment", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Time Trend
        st.subheader("📈 Hourly Trend")
        df_tweets['hour'] = df_tweets['date'].dt.floor('h')
        trend = df_tweets.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        fig_trend = px.line(trend, x='hour', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No tweets found. The scraper might be temporarily rate-limited.")
