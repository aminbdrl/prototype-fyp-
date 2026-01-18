import streamlit as st
import pandas as pd
import re
import snscrape.modules.twitter as sntwitter
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import plotly.express as px

# ----------------------------
# Dashboard Auto Refresh
# ----------------------------
REFRESH_SECONDS = 60
st_autorefresh = st.experimental_singleton

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Kelantan Social Sentiment Dashboard", layout="wide")
st.title("📊 Real‑Time Sentiment Analysis Dashboard — Kelantan")
st.write("Live sentiment trends from Malay tweets mentioning Kelantan and related society keywords")

# ----------------------------
# Load Public Dataset for Training
# ----------------------------
@st.cache_data
def load_public_dataset():
    # Load your merged sentiment Csv
    return pd.read_csv("malay_twitter_sentiment.csv")

df_train = load_public_dataset()

# ----------------------------
# Text Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()

df_train["clean_text"] = df_train["text"].apply(clean_text)

# ----------------------------
# Train Model
# ----------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(df_train["clean_text"])
y_train = df_train["label"]

model = MultinomialNB()
model.fit(X_train, y_train)

# ----------------------------
# Kelantan Slang (simple list)
# ----------------------------
kelantan_slang = [
    "ganu", "kelate", "kitorg", "dok", "jupo", "kato", "ngan", "sabo",
    "maok", "pegh", "aih"
]

def count_slang(text):
    return sum([1 for w in text.split() if w in kelantan_slang])

# ----------------------------
# Twitter Scrape
# ----------------------------
def scrape_kelantan_tweets(limit):
    query = (
        "(Kelantan OR 'orang kelantan' OR 'negeri kelantan' OR 'rakyat kelantan') lang:ms"
    )
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append([tweet.date, tweet.content])
    return pd.DataFrame(tweets, columns=["date", "tweet"])

# ----------------------------
# User Input
# ----------------------------
tweet_limit = st.slider("Number of Tweets per Refresh", 20, 200, 50)

# ----------------------------
# Analyze Now
# ----------------------------
if st.button("Refresh Sentiment Now"):

    df_tweets = scrape_kelantan_tweets(tweet_limit)
    df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
    df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))
    
    # slang count
    df_tweets["slang_count"] = df_tweets["clean_text"].apply(count_slang)

    # Save to CSV
    df_tweets.to_csv("kelantan_live_tweets.csv", index=False)

    # Display table
    st.subheader("📋 Live Tweets & Sentiment")
    st.dataframe(df_tweets[["date", "tweet", "sentiment", "slang_count"]])

    # ----------------------------
    # Pie Chart & Percentages
    # ----------------------------
    sentiment_counts = df_tweets["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig_pie = px.pie(
        sentiment_counts, names="sentiment", values="count",
        title="Sentiment Distribution (%)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ----------------------------
    # Time Trend Analysis
    # ----------------------------
    df_tweets['hour'] = df_tweets['date'].dt.floor('H')
    trend = df_tweets.groupby(['hour', 'sentiment']).size().reset_index(name='count')
    fig_trend = px.line(
        trend, x='hour', y='count', color='sentiment',
        title="Sentiment Trend Over Time (Hourly)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ----------------------------
    # Kelantan Slang Summary
    # ----------------------------
    slang_total = df_tweets["slang_count"].sum()
    st.markdown(f"**Total Kelantan‑specific slang occurrences:** {slang_total}")

