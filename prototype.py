import streamlit as st
import pandas as pd
import re
from ntscraper import Nitter  # Replaced broken snscrape
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import plotly.express as px

# ----------------------------
# Dashboard Settings
# ----------------------------
# Note: st.experimental_singleton is deprecated. Using cache_resource.
@st.cache_resource
def get_nitter_scraper():
    return Nitter(log_level=1)

scraper = get_nitter_scraper()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Kelantan Social Sentiment Dashboard", layout="wide")
st.title("📊 Real‑Time Sentiment Analysis Dashboard — Kelantan")
st.write("Live sentiment trends from Malay tweets mentioning Kelantan using Nitter Scraper")

# ----------------------------
# Load & Train Model
# ----------------------------
@st.cache_data
def load_and_train():
    # Load your training data
    try:
        df_train = pd.read_csv("prototaip.csv")
    except FileNotFoundError:
        st.error("Error: 'prototaip.csv' not found. Please ensure the training file is in the same directory.")
        st.stop()

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
        text = re.sub(r"[^a-z\s']", "", text)
        return text.strip()

    df_train["clean_text"] = df_train["text"].apply(clean_text)
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(df_train["clean_text"])
    y_train = df_train["label"]

    model = MultinomialNB()
    model.fit(X_train, y_train)
    return vectorizer, model, clean_text

vectorizer, model, clean_text = load_and_train()

# ----------------------------
# Kelantan Slang Analysis
# ----------------------------
kelantan_slang = ["ganu", "kelate", "kitorg", "dok", "jupo", "kato", "ngan", "sabo", "maok", "pegh", "aih"]

def count_slang(text):
    return sum([1 for w in text.split() if w in kelantan_slang])

# ----------------------------
# Twitter Scrape Function (Fixed)
# ----------------------------
def scrape_kelantan_tweets(limit):
    query = "Kelantan OR 'orang kelantan' OR 'negeri kelantan'"
    # ntscraper returns a dictionary with a 'tweets' list
    scraped_data = scraper.get_tweets(query, mode='term', number=limit, language='ms')
    
    tweets_list = []
    for t in scraped_data['tweets']:
        tweets_list.append({
            "date": t['date'],
            "tweet": t['text']
        })
    
    df = pd.DataFrame(tweets_list)
    if not df.empty:
        # Convert Nitter date format to pandas datetime
        df['date'] = pd.to_datetime(df['date'].str.replace('·', '').strip(), errors='coerce')
    return df

# ----------------------------
# User Interface
# ----------------------------
tweet_limit = st.sidebar.slider("Number of Tweets to Fetch", 10, 100, 30)

if st.button("Refresh Sentiment Now"):
    with st.spinner("Fetching live data from X (via Nitter)..."):
        df_tweets = scrape_kelantan_tweets(tweet_limit)

    if df_tweets.empty:
        st.warning("No tweets found or scraper blocked. Try again in a few minutes.")
    else:
        # Process Data
        df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
        df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))
        df_tweets["slang_count"] = df_tweets["clean_text"].apply(count_slang)

        # Save results
        df_tweets.to_csv("kelantan_live_tweets.csv", index=False)

        # Layout Columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Live Tweets & Sentiment")
            st.dataframe(df_tweets[["date", "tweet", "sentiment", "slang_count"]], height=400)

        with col2:
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df_tweets["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig_pie = px.pie(sentiment_counts, names="sentiment", values="count", 
                             color="sentiment", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Time Trend
        st.subheader("📈 Hourly Sentiment Trend")
        df_tweets['hour'] = df_tweets['date'].dt.floor('h')
        trend = df_tweets.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        fig_trend = px.line(trend, x='hour', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Slang Summary
        slang_total = df_tweets["slang_count"].sum()
        st.info(f"💡 **Insight:** Found {slang_total} Kelantan‑specific slang occurrences in this batch.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Kelantan Sentiment Analysis 2026")
