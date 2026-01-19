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
# 1. TEXT PREPROCESSING & VALIDATION
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
    
    text_lower = text.lower()
    
    kelantan_keywords = [
        'kelantan', 'kelate', 'kecek kelate', 
        'kota bharu', 'kb', 'pantai cahaya bulan', 'pcb',
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'jeli', 'lojing', 'rantau panjang',
        'nasi kerabu', 'nasi dagang', 'solok lada',
        'tok guru', 'pengkalan chepa', 'wakaf che yeh',
        'pasar siti khadijah', 'pantai irama', 'masjid kampung laut',
        'orang kelate', 'oghe kelate', 'demo kelate'
    ]
    
    return any(keyword in text_lower for keyword in kelantan_keywords)

def get_kelantan_keywords_found(text):
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    
    kelantan_keywords = [
        'kelantan', 'kelate', 'kota bharu', 'kb', 
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'nasi kerabu', 'nasi dagang', 'orang kelate', 'oghe kelate',
        'tok guru', 'jeli', 'lojing', 'rantau panjang'
    ]
    
    found = [kw for kw in kelantan_keywords if kw in text_lower]
    return found

# =====================================
# 2. LOAD DATASET & TRAIN MODEL
# =====================================
@st.cache_data
def load_and_train():
    # Using the customized dataset we generated
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
# 3. TWITTER SCRAPER (Nitter)
# =====================================
@st.cache_data(ttl=1800)
def scrape_kelantan_recent(limit=50, hours=24):
    nitter_instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.net"
    ]
    
    queries = ["Kelantan", "kelate"]
    all_tweets = []
    
    for instance in nitter_instances:
        if len(all_tweets) >= limit: break
        for query in queries:
            try:
                url = f"{instance}/search?f=tweets&q={query}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    tweets = soup.find_all('div', class_='timeline-item')
                    for tweet in tweets[:limit]:
                        tweet_text_elem = tweet.find('div', class_='tweet-content')
                        if tweet_text_elem:
                            all_tweets.append({
                                "date": datetime.now() - timedelta(minutes=random.randint(0, 60)),
                                "tweet": tweet_text_elem.get_text(strip=True)
                            })
                time.sleep(0.5)
            except: continue
    return pd.DataFrame(all_tweets) if all_tweets else pd.DataFrame()

# =====================================
# 4. LOAD DATA FROM CSV
# =====================================
@st.cache_data
def load_all_kelantan_data():
    df = pd.read_csv("final_kelantan_customized_dataset.csv")
    df = df.dropna(subset=["comment/tweet"])
    
    # Generate mock dates for visualization
    df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='10min')
    df = df.rename(columns={"comment/tweet": "tweet"})
    
    df["is_kelantan"] = df["tweet"].apply(is_kelantan_related)
    df_kelantan = df[df["is_kelantan"]].copy()
    
    return df_kelantan[["date", "tweet"]].reset_index(drop=True)

# =====================================
# 5. SIDEBAR & CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Try Live Twitter", "Use Complete Kelantan Dataset"],
    index=1
)

# Color Map Configuration
# Neutral = Light Blue (#ADD8E6)
# Negative = Red (#FF0000)
# Positive = Green (#00CC66)
sentiment_colors = {
    'positive': '#00CC66',
    'negative': '#FF0000',
    'neutral': '#ADD8E6'
}

# =====================================
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Processing Kelantan Data..."):
        if data_source == "Try Live Twitter":
            df_tweets = scrape_kelantan_recent()
            if df_tweets.empty:
                st.warning("Live data restricted. Switching to CSV...")
                df_tweets = load_all_kelantan_data()
        else:
            df_tweets = load_all_kelantan_data()

    if not df_tweets.empty:
        df_tweets["kelantan_keywords"] = df_tweets["tweet"].apply(get_kelantan_keywords_found)
        df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
        df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))

        # Visuals
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Tweet Analysis Table")
            display_df = df_tweets[["date", "tweet", "sentiment", "kelantan_keywords"]].copy()
            display_df["kelantan_keywords"] = display_df["kelantan_keywords"].apply(lambda x: ", ".join(x))
            st.dataframe(display_df, use_container_width=True, height=400)

        with col2:
            st.subheader("📊 Sentiment Distribution")
            counts = df_tweets["sentiment"].value_counts().reset_index()
            counts.columns = ['sentiment', 'count']
            
            fig_pie = px.pie(
                counts, values='count', names='sentiment',
                hole=0.4,
                color='sentiment',
                color_discrete_map=sentiment_colors
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📈 Sentiment Trend Over Time")
        df_tweets["hour"] = df_tweets["date"].dt.floor("h")
        trend = df_tweets.groupby(["hour", "sentiment"]).size().reset_index(name="count")
        
        fig_trend = px.line(
            trend, x="hour", y="count", color="sentiment",
            markers=True,
            color_discrete_map=sentiment_colors
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Statistics Metrics
        st.subheader("📊 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        total = len(df_tweets)
        m1.metric("Total Rows", total)
        m2.metric("Positive", f"{(df_tweets['sentiment']=='positive').sum()/total*100:.1f}%")
        m3.metric("Neutral", f"{(df_tweets['sentiment']=='neutral').sum()/total*100:.1f}%")
        m4.metric("Negative", f"{(df_tweets['sentiment']=='negative').sum()/total*100:.1f}%")
    else:
        st.error("No data found.")

else:
    st.info("👆 Click 'Refresh Analysis' to start the Kelantan Sentiment Dashboard.")
