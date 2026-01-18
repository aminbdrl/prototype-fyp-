import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter

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
# 2. LOAD DATASET & TRAIN MODEL
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
# 3. SCRAPE KELANTAN TWEETS (SNSCRAPE)
# =====================================
@st.cache_data(ttl=1800)  # cache for 30 minutes
def scrape_kelantan_recent(limit=50, hours=24):
    """
    Fetch tweets from the last X hours using snscrape.
    """
    since_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
    until_date = datetime.now().strftime('%Y-%m-%d')
    
    # Query for Kelantan-related tweets
    query = f"(Kelantan OR #Kelantan OR orang kelantan) lang:ms since:{since_date} until:{until_date}"
    
    tweets = []
    
    try:
        # Scrape tweets using snscrape
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            
            tweets.append({
                "date": tweet.date,
                "tweet": tweet.rawContent
            })
    
    except Exception as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return pd.DataFrame()
    
    if not tweets:
        # Fallback: try without language filter
        query_broad = f"(Kelantan OR #Kelantan) since:{since_date}"
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query_broad).get_items()):
                if i >= limit:
                    break
                
                tweets.append({
                    "date": tweet.date,
                    "tweet": tweet.rawContent
                })
        except Exception:
            return pd.DataFrame()
    
    df = pd.DataFrame(tweets)
    
    # Remove duplicates
    if not df.empty:
        df = df.drop_duplicates(subset=['tweet'])
        df["date"] = pd.to_datetime(df["date"])
    
    return df.head(limit)

# =====================================
# 4. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

tweet_limit = st.sidebar.slider(
    "Number of Tweets (Last 24 Hours)",
    20, 100, 50
)

hours = st.sidebar.selectbox(
    "Time Window",
    options=[12, 24, 48, 72, 168],
    format_func=lambda x: {
        12: "Last 12 Hours",
        24: "Last 24 Hours (1 Day)",
        48: "Last 48 Hours (2 Days)",
        72: "Last 72 Hours (3 Days)",
        168: "Last 7 Days"
    }[x],
    index=1  # Default to 24 hours
)

# =====================================
# 5. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Fetching recent Twitter data..."):
        # Clear cache to force fresh data
        scrape_kelantan_recent.clear()
        df_tweets = scrape_kelantan_recent(tweet_limit, hours)

    if df_tweets.empty:
        st.error("⚠️ Unable to fetch tweets at this moment.")
        st.info("**Possible reasons:**\n- No tweets found in the selected time window\n- Twitter rate limits\n- Network connection issues\n\nTry selecting a longer time window (e.g., Last 7 Days)")
        st.stop()
    
    st.success(f"✅ Successfully fetched {len(df_tweets)} tweets!")

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
