import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Kelantan Social Sentiment Dashboard",
    layout="wide"
)

st.title("Sentiment Analysis Dashboard — Kelantan")

# ==================================================
# TEXT PREPROCESSING
# ==================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()

def is_kelantan_related(text):
    if not isinstance(text, str):
        return False
    keywords = [
        "kelantan", "kelate", "kota bharu", "kb",
        "tumpat", "pasir mas", "tanah merah",
        "kuala krai", "gua musang", "bachok",
        "nasi kerabu", "nasi dagang",
        "orang kelate", "oghe kelate"
    ]
    return any(k in text.lower() for k in keywords)

# ==================================================
# LOAD & TRAIN MODEL
# ==================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet", "majority_sent"])
    df["clean_text"] = df["comment/tweet"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["majority_sent"].str.lower().str.strip()

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_and_train()

# ==================================================
# LIVE TWITTER SCRAPER
# ==================================================
@st.cache_data(ttl=1800)
def scrape_kelantan_recent(limit=100):
    nitter = "https://nitter.net/search?f=tweets&q=kelantan"
    tweets = []

    try:
        r = requests.get(nitter, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        items = soup.find_all("div", class_="timeline-item")

        for item in items[:limit]:
            text = item.find("div", class_="tweet-content")
            if text:
                tweets.append({
                    "date": datetime.now(),
                    "tweet": text.text.strip()
                })
    except:
        pass

    return pd.DataFrame(tweets)

# ==================================================
# LOAD CSV DATA
# ==================================================
@st.cache_data
def load_all_kelantan_data():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet"])
    df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq="30min")
    df = df.rename(columns={"comment/tweet": "tweet"})
    return df[df["tweet"].apply(is_kelantan_related)][["date", "tweet"]]

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Try Live Twitter", "Use Complete Kelantan Dataset"],
    index=1
)

if data_source == "Try Live Twitter":
    tweet_limit = st.sidebar.slider("Number of Tweets", 20, 200, 100)

# ==================================================
# RUN ANALYSIS
# ==================================================
if st.sidebar.button("🔄 Refresh Analysis"):

    if data_source == "Try Live Twitter":
        scrape_kelantan_recent.clear()
        df_tweets = scrape_kelantan_recent(tweet_limit)

        if df_tweets.empty:
            st.warning("Live Twitter unavailable — using dataset instead")
            df_tweets = load_all_kelantan_data()
    else:
        df_tweets = load_all_kelantan_data()

    if df_tweets.empty:
        st.error("No data available")
        st.stop()

    df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
    df_tweets["sentiment"] = model.predict(
        vectorizer.transform(df_tweets["clean_text"])
    )

    # ==================================================
    # SENTIMENT DISTRIBUTION (COLOUR FIXED)
    # ==================================================
    st.subheader("📊 Sentiment Distribution")

    fig_pie = px.pie(
        df_tweets,
        names="sentiment",
        hole=0.4,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
        color="sentiment",
        color_discrete_map={
            "positive": "#1f77b4",
            "neutral": "#00cc66",
            "negative": "#ff0000"
        }
    )

    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("👈 Click **Refresh Analysis** to start")
