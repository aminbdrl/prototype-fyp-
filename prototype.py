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
        skip_instance_check=False,
        instances=[
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
            "https://nitter.net",
            "https://nitter.fdn.fr",
            "https://nitter.unixfox.eu",
            "https://nitter.it"
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
@st.cache_data(ttl=1800)  # cache for 30 minutes
def scrape_kelantan_recent(limit=50, hours=24):
    """
    Fetch tweets from the last X hours (near real-time).
    Falls back to broader search if time-filtered search fails.
    """
    queries = [
        "Kelantan",
        "#Kelantan",
        "orang kelantan",
        "Kelantan Malaysia"
    ]
    
    all_tweets = []
    
    # Try each query
    for query in queries:
        try:
            result = scraper.get_tweets(
                query,
                mode="term",
                number=limit // len(queries) + 10,
                language="ms"
            )
            
            if result and result.get("tweets"):
                for t in result.get("tweets", []):
                    all_tweets.append({
                        "date": t.get("date"),
                        "tweet": t.get("text")
                    })
                
                # If we got enough tweets, break
                if len(all_tweets) >= limit:
                    break
        except Exception as e:
            continue
    
    # If still no tweets, try without language filter
    if len(all_tweets) < 10:
        try:
            result = scraper.get_tweets(
                "Kelantan",
                mode="term",
                number=limit
            )
            
            if result and result.get("tweets"):
                for t in result.get("tweets", []):
                    all_tweets.append({
                        "date": t.get("date"),
                        "tweet": t.get("text")
                    })
        except Exception:
            pass
    
    if not all_tweets:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_tweets)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['tweet'])
    
    if not df.empty:
        df["date"] = pd.to_datetime(
            df["date"].str.replace("·", "", regex=False),
            errors="coerce"
        )
        
        # Filter by time window if possible
        if hours and not df["date"].isna().all():
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            df = df[df["date"] >= cutoff]
    
    return df.head(limit)

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
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Fetching recent Twitter data..."):
        # Clear cache to force fresh data
        scrape_kelantan_recent.clear()
        df_tweets = scrape_kelantan_recent(tweet_limit, hours)

    if df_tweets.empty:
        st.error("⚠️ Unable to fetch tweets at this moment.")
        st.info("**Troubleshooting:**\n- Nitter instances may be temporarily down\n- Try again in a few moments\n- Or use demo data below")
        
        # Provide demo/fallback data
        if st.button("📊 Use Demo Data Instead"):
            demo_tweets = [
                {"date": datetime.now() - timedelta(hours=i), 
                 "tweet": f"Demo tweet tentang Kelantan #{i}"}
                for i in range(tweet_limit)
            ]
            df_tweets = pd.DataFrame(demo_tweets)
            df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
            df_tweets["sentiment"] = ["positive", "negative", "neutral"][0:len(df_tweets)] * (len(df_tweets)//3 + 1)
            df_tweets = df_tweets.head(tweet_limit)
        else:
            st.stop()
    else:
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
