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
    """
    Check if tweet is actually related to Kelantan
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Kelantan-specific keywords
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
    """
    Return which Kelantan keywords were found in the text
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    
    kelantan_keywords = [
        'kelantan', 'kelate', 'kota bharu', 'kb', 
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'nasi kerabu', 'nasi dagang', 'orang kelate', 'oghe kelate'
    ]
    
    found = [kw for kw in kelantan_keywords if kw in text_lower]
    return found

# =====================================
# 2. LOAD DATASET & TRAIN MODEL
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
# 3. TWITTER SCRAPER (Multiple Methods)
# =====================================
@st.cache_data(ttl=1800)  # cache for 30 minutes
def scrape_kelantan_recent(limit=50, hours=24):
    """
    Fetch tweets using multiple Nitter instances with better error handling.
    """
    nitter_instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.fdn.fr",
        "https://nitter.unixfox.eu",
        "https://nitter.it",
        "https://nitter.1d4.us",
        "https://nitter.kavin.rocks"
    ]
    
    queries = [
        "Kelantan",
        "kelantan",
        "#Kelantan"
    ]
    
    all_tweets = []
    
    for instance in nitter_instances:
        if len(all_tweets) >= limit:
            break
            
        for query in queries:
            if len(all_tweets) >= limit:
                break
                
            try:
                url = f"{instance}/search?f=tweets&q={query}&since=&until=&near="
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find tweet containers
                    tweets = soup.find_all('div', class_='timeline-item')
                    
                    for tweet in tweets[:limit]:
                        try:
                            # Extract tweet text
                            tweet_text_elem = tweet.find('div', class_='tweet-content')
                            if tweet_text_elem:
                                tweet_text = tweet_text_elem.get_text(strip=True)
                                
                                # Extract date
                                date_elem = tweet.find('span', class_='tweet-date')
                                tweet_date = datetime.now() - timedelta(hours=hours/2)  # Approximate
                                
                                if tweet_text and len(tweet_text) > 10:
                                    all_tweets.append({
                                        "date": tweet_date,
                                        "tweet": tweet_text
                                    })
                        except Exception:
                            continue
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception:
                continue
    
    if not all_tweets:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_tweets)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['tweet'])
    
    # Filter to get recent tweets within time window
    if not df.empty:
        cutoff = datetime.now() - timedelta(hours=hours)
        df = df[df["date"] >= cutoff]
    
    return df.head(limit)

# =====================================
# 4. LOAD ALL KELANTAN DATA FROM CSV
# =====================================
@st.cache_data
def load_all_kelantan_data():
    """
    Load complete dataset and filter for Kelantan-related content only
    """
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet"])
    
    # Simulate recent dates across the entire dataset
    df["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df),
        freq='30min'  # More granular time intervals
    )
    
    df = df.rename(columns={"comment/tweet": "tweet"})
    
    # Filter for Kelantan-related only
    df["is_kelantan"] = df["tweet"].apply(is_kelantan_related)
    df_kelantan = df[df["is_kelantan"]].copy()
    
    return df_kelantan[["date", "tweet"]].reset_index(drop=True)

# =====================================
# 5. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Try Live Twitter", "Use Complete Kelantan Dataset"],
    index=1
)

if data_source == "Try Live Twitter":
    tweet_limit = st.sidebar.slider(
        "Number of Tweets",
        20, 200, 100
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
        index=1
    )
else:
    st.sidebar.info("Using complete Kelantan dataset from CSV")

# =====================================
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Loading data..."):
        
        if data_source == "Try Live Twitter":
            scrape_kelantan_recent.clear()
            df_tweets = scrape_kelantan_recent(tweet_limit, hours)
            
            if df_tweets.empty:
                st.warning("⚠️ Live Twitter data unavailable. Loading complete dataset...")
                df_tweets = load_all_kelantan_data()
            else:
                # Filter for Kelantan
                df_tweets["is_kelantan"] = df_tweets["tweet"].apply(is_kelantan_related)
                df_tweets = df_tweets[df_tweets["is_kelantan"]].copy()
        else:
            df_tweets = load_all_kelantan_data()

    if df_tweets.empty:
        st.error("No Kelantan-related data available.")
        st.stop()
    
    st.success(f"✅ Analyzing {len(df_tweets)} Kelantan-related tweets!")

    # Add Kelantan keywords
    df_tweets["kelantan_keywords"] = df_tweets["tweet"].apply(get_kelantan_keywords_found)
    
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
        st.subheader("📋 Kelantan Tweet Analysis")
        
        # Add keyword column for display
        display_df = df_tweets[["date", "tweet", "sentiment", "kelantan_keywords"]].copy()
        display_df["kelantan_keywords"] = display_df["kelantan_keywords"].apply(
            lambda x: ", ".join(x) if x else ""
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "date": "Date/Time",
                "tweet": st.column_config.TextColumn("Tweet", width="large"),
                "sentiment": "Sentiment",
                "kelantan_keywords": "Kelantan Keywords Found"
            }
        )

    with col2:
        st.subheader("📊 Sentiment Distribution")
        
        sentiment_counts = df_tweets["sentiment"].value_counts()
        
fig_pie = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    hole=0.4,
    color_discrete_map={
        'positive': '#1f77b4',   # blue (optional, change if you want)
        'negative': '#ff0000',   # red
        'neutral': '#00cc66'     # green
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
            'positive': '#00cc66',
            'negative': '#ff4444',
            'neutral': '#ffaa00'
        }
    )
    
    fig_trend.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Tweets",
        hovermode='x unified'
    )

    st.plotly_chart(fig_trend, use_container_width=True)
    
    # =====================================
    # STATISTICS
    # =====================================
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(df_tweets))
    
    with col2:
        positive_pct = (df_tweets["sentiment"] == "positive").sum() / len(df_tweets) * 100
        st.metric("Positive", f"{positive_pct:.1f}%")
    
    with col3:
        neutral_pct = (df_tweets["sentiment"] == "neutral").sum() / len(df_tweets) * 100
        st.metric("Neutral", f"{neutral_pct:.1f}%")
    
    with col4:
        negative_pct = (df_tweets["sentiment"] == "negative").sum() / len(df_tweets) * 100
        st.metric("Negative", f"{negative_pct:.1f}%")

else:
    st.info("👆 Click **Refresh Analysis** to start")
    
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Data Source Options:**
        
        1. **Try Live Twitter**: Attempts to fetch real-time tweets (may be unreliable due to rate limits)
        2. **Use Complete Kelantan Dataset**: Analyzes ALL Kelantan-related tweets from your training data
        
        **Features:**
        
        - Automatically filters for Kelantan-specific content only
        - Shows complete dataset when using CSV option
        - Displays which Kelantan keywords were found in each tweet
        - Provides trend analysis and sentiment distribution
        
        **Note:** The complete dataset option gives you the most comprehensive analysis of Kelantan sentiment.
        """)
