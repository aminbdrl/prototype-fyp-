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
# PAGE CONFIG (MUST BE FIRST)
# ==================================================
st.set_page_config(
    page_title="Kelantan Social Sentiment Dashboard",
    layout="wide"
)

st.title("Sentiment Analysis Dashboard — Kelantan")

# ==================================================
# 1. TEXT PREPROCESSING
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

def get_kelantan_keywords_found(text):
    if not isinstance(text, str):
        return []
    keywords = [
        "kelantan", "kelate", "kota bharu", "kb",
        "tumpat", "pasir mas", "tanah merah",
        "kuala krai", "gua musang", "bachok",
        "nasi kerabu", "nasi dagang",
        "orang kelate", "oghe kelate"
    ]
    return [k for k in keywords if k in text.lower()]

# ==================================================
# 2. LOAD DATASET & TRAIN MODEL
# ==================================================
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
    y = df["majority_sent"].str.lower().str.strip()

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_and_train()

# ==================================================
# 3. LOAD COMPLETE KELANTAN DATASET
# ==================================================
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

# ==================================================
# 4. SIDEBAR
# ==================================================
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Refresh Analysis"):
    df_tweets = load_all_kelantan_data()

    if df_tweets.empty:
        st.error("No Kelantan-related tweets found.")
        st.stop()

    # ==================================================
    # 5. SENTIMENT PREDICTION
    # ==================================================
    df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
    df_tweets["sentiment"] = model.predict(
        vectorizer.transform(df_tweets["clean_text"])
    )

    df_tweets["sentiment"] = df_tweets["sentiment"].str.lower().str.strip()
    df_tweets["kelantan_keywords"] = df_tweets["tweet"].apply(get_kelantan_keywords_found)

    # ==================================================
    # DISPLAY TABLE
    # ==================================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Kelantan Tweet Analysis")

        df_show = df_tweets.copy()
        df_show["kelantan_keywords"] = df_show["kelantan_keywords"].apply(
            lambda x: ", ".join(x)
        )

        st.dataframe(
            df_show[["date", "tweet", "sentiment", "kelantan_keywords"]],
            use_container_width=True,
            height=420
        )

    # ==================================================
    # SENTIMENT DISTRIBUTION (COLOUR FIXED)
    # ==================================================
    with col2:
        st.subheader("📊 Sentiment Distribution")

        fig_pie = px.pie(
            df_tweets,
            names="sentiment",
            hole=0.45,
            category_orders={
                "sentiment": ["positive", "neutral", "negative"]
            },
            color="sentiment",
            color_discrete_map={
                "positive": "#1f77b4",  # BLUE
                "neutral": "#00cc66",   # GREEN
                "negative": "#ff0000"   # RED
            }
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # ==================================================
    # TREND ANALYSIS (SAME COLOURS)
    # ==================================================
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
        category_orders={
            "sentiment": ["positive", "neutral", "negative"]
        },
        color_discrete_map={
            "positive": "#1f77b4",
            "neutral": "#00cc66",
            "negative": "#ff0000"
        }
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # ==================================================
    # SUMMARY METRICS
    # ==================================================
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tweets", len(df_tweets))

    with col2:
        st.metric(
            "Positive (%)",
            f"{(df_tweets.sentiment == 'positive').mean() * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Neutral (%)",
            f"{(df_tweets.sentiment == 'neutral').mean() * 100:.1f}%"
        )

    with col4:
        st.metric(
            "Negative (%)",
            f"{(df_tweets.sentiment == 'negative').mean() * 100:.1f}%"
        )

else:
    st.info("👈 Click **Refresh Analysis** to start")
