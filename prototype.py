import streamlit as st
import pandas as pd
import re
from ntscraper import Nitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from datetime import datetime, timedelta

# =====================================
# PAGE CONFIG (MUST BE FIRST)
# =====================================
st.set_page_config(
    page_title="Kelantan Social Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Near Real-Time Sentiment Analysis Dashboard — Kelantan")

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

# =====================================
# 2. NITTER SCRAPER (STABLE)
# =====================================
@st.cache_resource
def get_nitter_scraper():
    return Nitter(
        log_level=1,
        instances=[
            "https://nitter.net",
            "https://nitter.poast.org",
            "https://nitter.fdn.fr"
        ]
    )

scraper = get_nitter_scraper()

# =====================================
# 3. LOAD DATASET & TRAIN MODEL
# =====================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("prototaip.csv")

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
# 4. SCRAPE KELANTAN TWEETS (NEAR REAL-TIME)
# =====================================
@st.cache_data(ttl=3600)  # cache for 1 hour
def scrape_kelantan_recent(limit=50, hours=24):
    """
    Fetch tweets from the last X hours (near real-time).
    """
    since_time = (datetime.utcnow() - timedelta(hours=hours)).strftime('%Y-%m-%d')
    query = f"Kelantan OR orang kelantan since:{since_time}"

    try:
        result = scraper.get_tweets(
            query,
            mode="term",
            number=limit,
            language="ms"
        )
    except Exception:
        return pd.DataFrame()

    tweets = []
    for t in result.get("tweets", []):
        tweets.append({
            "date": t.get("date"),
            "tweet": t.get("text")
        })

    df = pd.DataFrame(tweets)

    if not df.empty:
        df["date"] = pd.to_datetime(
            df["date"].str.replace("·", "", regex=False),
            errors="coerce"
        )

    return df

# =====================================
# 5. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

tweet_limit = st.sidebar.slider(
    "Number of Tweets (Last 24 Hours)",
    20, 100, 50
)

hours = st.sidebar.selectbox(
    "Time Window",
    [24, 48],
    index=0
)

# =====================================
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Fetching recent Twitter data..."):
        df_tweets = scrape_kelantan_recent(tweet_limit, hours)

    if df_tweets.empty:
        st.warning("No recent tweets available. Showing last cached data.")
        st.stop()

    # Sentiment Prediction
    df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
    df_tweets["sentiment"] = model.predict(
        vectorizer.transform(df_tweets["clean_text"])
    )

    # =====================================
    # DISPLAY RESULTS
    # =====================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Recent Tweet Analysis")
        st.dataframe(
            df_tweets[["date", "tweet", "sentiment"]],
            use_container_width=True
        )

    with col2:
        st.subheader("📊 Sentiment Distribution")
        fig_pie = px.pie(
            df_tweets,
            names="sentiment",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # TREND ANALYSIS
    # =====================================
    st.subheader("📈 Sentiment Trend (Near Real-Time)")

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
        markers=True
    )

    st.plotly_chart(fig_trend, use_container_width=True)

else:
    st.info("Click **Refresh Analysis** to start near real-time analysis.")
