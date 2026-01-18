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

st.title("📊 Near Real-Time Sentiment Analysis Dashboard — Kelantan")

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
    Check if tweet is actually related to Kelantan - STRICT filtering
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Kelantan-specific keywords (COMPREHENSIVE LIST)
    kelantan_keywords = [
        # Core
        'kelantan', 'kelate', 'kecek kelate', 'klate',
        
        # Cities & Districts
        'kota bharu', 'kota bahru', 'kb', 'kotabharu',
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'jeli', 'lojing', 'rantau panjang', 'pengkalan chepa',
        'wakaf che yeh', 'wakaf bharu', 'ketereh',
        
        # Famous Places
        'pantai cahaya bulan', 'pcb', 'pantai irama',
        'pasar siti khadijah', 'pasar besar siti khadijah',
        'masjid kampung laut', 'istana jahar',
        'muzium negeri kelantan', 'handicraft village',
        'bukit marak', 'wat photivihan', 'bukit bunga',
        
        # Food (Kelantan Specialty)
        'nasi kerabu', 'nasi dagang', 'nasi tumpang',
        'solok lada', 'ayam percik', 'keropok lekor kelate',
        'budu', 'laksam', 'akok', 'nekbat',
        
        # Culture & Dialect
        'orang kelate', 'oghe kelate', 'demo kelate',
        'gapo', 'mung', 'pah', 'gak', 'kito',
        'tok guru', 'mak cik kelantan', 'pak cik kelantan',
        
        # Events & Politics
        'kerajaan negeri kelantan', 'pas kelantan',
        'pkr kelantan', 'umno kelantan', 'kelantan fc',
        
        # Rivers & Nature
        'sungai kelantan', 'sungai golok', 'gunung stong',
        'gunung ayam', 'jeram pasu', 'air terjun lata rek'
    ]
    
    return any(keyword in text_lower for keyword in kelantan_keywords)

def get_kelantan_keywords_found(text):
    """
    Return which Kelantan keywords were found in the text
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    
    # Most common Kelantan keywords for display
    kelantan_keywords = [
        'kelantan', 'kelate', 'kota bharu', 'kotabharu', 'kb', 
        'tumpat', 'pasir mas', 'tanah merah', 'machang',
        'kuala krai', 'gua musang', 'pasir puteh', 'bachok',
        'nasi kerabu', 'nasi dagang', 'budu', 'laksam',
        'orang kelate', 'oghe kelate', 'demo kelate',
        'pantai cahaya bulan', 'pcb', 'pasar siti khadijah',
        'tok guru', 'pengkalan chepa', 'wakaf che yeh',
        'gapo', 'mung', 'kito', 'solok lada', 'ayam percik'
    ]
    
    found = [kw for kw in kelantan_keywords if kw in text_lower]
    return found[:5]  # Show max 5 keywords

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
# 4. ALTERNATIVE: USE YOUR CSV DATA
# =====================================
@st.cache_data
def load_recent_from_csv(hours=8760):
    """
    Fallback: Use your existing CSV data as 'recent' tweets
    Simulates data from the last X hours (default: 1 year)
    """
    df = pd.read_csv("prototaip.csv")
    df = df.dropna(subset=["comment/tweet"])
    
    # Calculate number of data points based on hours
    # For 1 year, use all data. For shorter periods, sample proportionally
    total_hours_in_year = 8760
    sample_size = min(len(df), int(len(df) * (hours / total_hours_in_year)))
    sample_size = max(sample_size, 50)  # Minimum 50 tweets
    
    # Take the most recent samples
    df_sample = df.tail(sample_size).copy()
    
    # Simulate recent dates based on time window
    df_sample["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df_sample),
        freq=f'{hours//len(df_sample)}H'
    )
    
    df_sample = df_sample.rename(columns={"comment/tweet": "tweet"})
    
    return df_sample[["date", "tweet"]]

# =====================================
# 5. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Try Live Twitter", "Use Training Data (Reliable)"],
    index=1
)

tweet_limit = st.sidebar.slider(
    "Number of Tweets",
    20, 100, 50
)

hours = st.sidebar.selectbox(
    "Time Window",
    options=[12, 24, 48, 72, 168, 720, 8760],
    format_func=lambda x: {
        12: "Last 12 Hours",
        24: "Last 24 Hours (1 Day)",
        48: "Last 48 Hours (2 Days)",
        72: "Last 72 Hours (3 Days)",
        168: "Last 7 Days (1 Week)",
        720: "Last 30 Days (1 Month)",
        8760: "Last 365 Days (1 Year)"
    }[x],
    index=6  # Default to 1 year
)

# =====================================
# 6. RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Fetching data..."):
        
        if data_source == "Try Live Twitter":
            scrape_kelantan_recent.clear()
            df_tweets = scrape_kelantan_recent(tweet_limit, hours)
            
            if df_tweets.empty:
                st.warning("⚠️ Live Twitter data unavailable. Switching to training data...")
                df_tweets = load_recent_from_csv(hours).head(tweet_limit)
        else:
            df_tweets = load_recent_from_csv(hours).head(tweet_limit)

    if df_tweets.empty:
        st.error("No data available.")
        st.stop()
    
    st.success(f"✅ Analyzing {len(df_tweets)} tweets!")

    # Filter for Kelantan-related tweets only
    df_tweets["is_kelantan"] = df_tweets["tweet"].apply(is_kelantan_related)
    df_tweets["kelantan_keywords"] = df_tweets["tweet"].apply(get_kelantan_keywords_found)
    
    # Show filtering stats
    total_tweets = len(df_tweets)
    kelantan_tweets = df_tweets["is_kelantan"].sum()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Tweets Fetched", total_tweets)
    with col_stat2:
        st.metric("✅ Kelantan-Related", kelantan_tweets)
    with col_stat3:
        relevance_pct = (kelantan_tweets / total_tweets * 100) if total_tweets > 0 else 0
        st.metric("Relevance Rate", f"{relevance_pct:.1f}%")
    
    # Filter to only Kelantan tweets
    df_tweets = df_tweets[df_tweets["is_kelantan"]].copy()
    
    if df_tweets.empty:
        st.warning("⚠️ No Kelantan-related tweets found. Try a different time window.")
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
        st.subheader("📋 Tweet Analysis")
        
        # Add keyword column for display
        display_df = df_tweets[["date", "tweet", "sentiment", "kelantan_keywords"]].copy()
        display_df["kelantan_keywords"] = display_df["kelantan_keywords"].apply(
            lambda x: ", ".join(x) if x else ""
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
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
                'positive': '#00cc66',
                'negative': '#ff4444',
                'neutral': '#ffaa00'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # TREND ANALYSIS
    # =====================================
    st.subheader("📈 Sentiment Trend Over Time")

    # Dynamic time grouping based on time window
    if hours <= 48:  # 2 days or less - group by hour
        df_tweets["time_group"] = df_tweets["date"].dt.floor("h")
        time_label = "Hour"
    elif hours <= 168:  # 1 week or less - group by day
        df_tweets["time_group"] = df_tweets["date"].dt.floor("D")
        time_label = "Day"
    elif hours <= 720:  # 1 month or less - group by day
        df_tweets["time_group"] = df_tweets["date"].dt.floor("D")
        time_label = "Day"
    else:  # More than 1 month - group by week
        df_tweets["time_group"] = df_tweets["date"].dt.to_period("W").dt.to_timestamp()
        time_label = "Week"
    
    trend = (
        df_tweets
        .groupby(["time_group", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig_trend = px.line(
        trend,
        x="time_group",
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
        xaxis_title=f"Time ({time_label})",
        yaxis_title="Number of Tweets",
        hovermode='x unified'
    )

    st.plotly_chart(fig_trend, use_container_width=True)
    
    # =====================================
    # STATISTICS
    # =====================================
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_pct = (df_tweets["sentiment"] == "positive").sum() / len(df_tweets) * 100
        st.metric("Positive", f"{positive_pct:.1f}%")
    
    with col2:
        neutral_pct = (df_tweets["sentiment"] == "neutral").sum() / len(df_tweets) * 100
        st.metric("Neutral", f"{neutral_pct:.1f}%")
    
    with col3:
        negative_pct = (df_tweets["sentiment"] == "negative").sum() / len(df_tweets) * 100
        st.metric("Negative", f"{negative_pct:.1f}%")

else:
    st.info("👆 Click **Refresh Analysis** to start")
    
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Data Source Options:**
        
        1. **Try Live Twitter**: Attempts to fetch real-time tweets (may be unreliable due to rate limits)
        2. **Use Training Data (Reliable)**: Uses your labeled dataset as recent data (recommended for demos)
        
        **Kelantan Validation:**
        - System automatically filters tweets to ensure they're Kelantan-related
        - Checks for keywords like: Kelantan, Kelate, Kota Bharu, Tumpat, Nasi Kerabu, etc.
        - Shows "Kelantan Keywords Found" column to verify relevance
        
        **Note:** This system uses a near real-time approach, analyzing tweets from the selected time window.
        """)
