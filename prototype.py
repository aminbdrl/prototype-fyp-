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

# =====================================
# PAGE CONFIG (MUST BE FIRST)
# =====================================
st.set_page_config(
    page_title="Kelantan Social Sentiment Dashboard",
    layout="wide"
)

st.title("Sentiment Analysis Dashboard — Kelantan")

# =====================================
# 1. TEXT PREPROCESSING
# =====================================
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

    kelantan_keywords = [
        'kelantan', 'kelate', 'kota bharu', 'kb',
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'jeli', 'rantau panjang', 'nasi kerabu', 'nasi dagang',
        'orang kelate', 'oghe kelate'
    ]
    return any(k in text.lower() for k in kelantan_keywords)

def get_kelantan_keywords_found(text):
    if not isinstance(text, str):
        return []
    keywords = [
        'kelantan', 'kelate', 'kota bharu', 'kb',
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'nasi kerabu', 'nasi dagang', 'orang kelate', 'oghe kelate'
    ]
    return [k for k in keywords if k in text.lower()]

# =====================================
# 2. LOAD DATA & TRAIN MODEL
# =====================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet", "majority_sent"])

    df["clean_text"] = df["comment/tweet"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["majority_sent"]

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_and_train()

# =====================================
# 3. TWITTER SCRAPER (NITTER)
# =====================================
@st.cache_data(ttl=1800)
def scrape_kelantan_recent(limit=100, hours=24):
    instances = [
        "https://nitter.net",
        "https://nitter.fdn.fr",
        "https://nitter.poast.org"
    ]

    tweets_data = []

    for site in instances:
        try:
            url = f"{site}/search?f=tweets&q=kelantan"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            tweets = soup.find_all("div", class_="timeline-item")

            for t in tweets[:limit]:
                content = t.find("div", class_="tweet-content")
                if content:
                    tweets_data.append({
                        "date": datetime.now(),
                        "tweet": content.text.strip()
                    })
        except:
            continue

    df = pd.DataFrame(tweets_data).drop_duplicates("tweet")
    return df.head(limit)

# =====================================
# 4. LOAD COMPLETE DATASET
# =====================================
@st.cache_data
def load_all_kelantan_data():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet"])

    df["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df),
        freq="30min"
    )

    df = df.rename(columns={"comment/tweet": "tweet"})
    df = df[df["tweet"].apply(is_kelantan_related)]

    return df[["date", "tweet"]].reset_index(drop=True)

# =====================================
# 5. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Use Complete Kelantan Dataset", "Try Live Twitter"],
    index=0
)

if data_source == "Try Live Twitter":
    tweet_limit = st.sidebar.slider("Number of Tweets", 20, 200, 100)
    hours = st.sidebar.selectbox(
        "Time Window",
        [12, 24, 48, 72, 168],
        index=1
    )

# =====================================
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Processing data..."):

        if data_source == "Try Live Twitter":
            scrape_kelantan_recent.clear()
            df_tweets = scrape_kelantan_recent(tweet_limit, hours)
            df_tweets = df_tweets[df_tweets["tweet"].apply(is_kelantan_related)]
        else:
            df_tweets = load_all_kelantan_data()

    if df_tweets.empty:
        st.error("No Kelantan-related data found.")
        st.stop()

    st.success(f"Analyzing {len(df_tweets)} tweets")

    df_tweets["kelantan_keywords"] = df_tweets["tweet"].apply(get_kelantan_keywords_found)
    df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)

    df_tweets["sentiment"] = model.predict(
        vectorizer.transform(df_tweets["clean_text"])
    )

    # =====================================
    # DISPLAY TABLE
    # =====================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Tweet Analysis")
        df_show = df_tweets.copy()
        df_show["kelantan_keywords"] = df_show["kelantan_keywords"].apply(
            lambda x: ", ".join(x)
        )

        st.dataframe(
            df_show[["date", "tweet", "sentiment", "kelantan_keywords"]],
            use_container_width=True,
            height=400
        )

    # =====================================
    # SENTIMENT DISTRIBUTION (FIXED COLORS)
    # =====================================
    with col2:
        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = df_tweets["sentiment"].value_counts()

        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.4,
            color_discrete_map={
                "negative": "#ff0000",  # RED
                "neutral": "#00cc66",   # GREEN
                "positive": "#1f77b4"   # BLUE
            }
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # TREND ANALYSIS
    # =====================================
    st.subheader("📈 Sentiment Trend Over Time")

    df_tweets["hour"] = df_tweets["date"].dt.floor("h")

    trend = (
        df_tweets
        .groupby(["hour", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig_trend = px.line(
        trend,
        x="hour",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={
            "negative": "#ff0000",
            "neutral": "#00cc66",
            "positive": "#1f77b4"
        }
    )

    st.plotly_chart(fig_trend, use_container_width=True)

else:
    st.info("👈 Click **Refresh Analysis** to start")
