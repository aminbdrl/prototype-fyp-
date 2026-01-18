import streamlit as st
import pandas as pd
import re
from ntscraper import Nitter # Replacement for broken snscrape
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px

# ----------------------------
# 1. Scraper Initialization
# ----------------------------
@st.cache_resource
def get_nitter_scraper():
    # Nitter allows scraping without an official Twitter API key
    return Nitter(log_level=1)

scraper = get_nitter_scraper()

# ----------------------------
# 2. Page Config
# ----------------------------
st.set_page_config(page_title="Kelantan Social Sentiment Dashboard", layout="wide")
st.title("📊 Real‑Time Sentiment Analysis Dashboard — Kelantan")
st.write("Live sentiment trends using the Nitter scraper and your custom training dataset.")

# ----------------------------
# 3. Model Training (Updated for prototaip.csv)
# ----------------------------
@st.cache_data
def load_and_train():
    try:
        # Load the uploaded dataset
        df_train = pd.read_csv("prototaip.csv")
        
        # DEFINED COLUMNS FROM YOUR DATASET:
        # text_col = "comment/tweet" 
        # label_col = "majority_sent"
        
        def clean_text(text):
            if not isinstance(text, str): return ""
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
            text = re.sub(r"[^a-z\s']", "", text)
            return text.strip()

        # Preprocess using the correct columns from your CSV
        df_train["clean_text"] = df_train["comment/tweet"].apply(clean_text)
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(df_train["clean_text"])
        y_train = df_train["majority_sent"]

        model = MultinomialNB()
        model.fit(X_train, y_train)
        return vectorizer, model, clean_text
    except FileNotFoundError:
        st.error("Error: 'prototaip.csv' not found in the directory.")
        st.stop()
    except KeyError as e:
        st.error(f"Mapping Error: Could not find column {e} in your CSV.")
        st.stop()

vectorizer, model, clean_text = load_and_train()

# ----------------------------
# 4. Twitter Scrape Function (Nitter)
# ----------------------------
def scrape_kelantan_tweets(limit):
    # Using 'kelantan' as the search term
    query = "Kelantan OR 'orang kelantan'"
    scraped_data = scraper.get_tweets(query, mode='term', number=limit, language='ms')
    
    tweets_list = []
    for t in scraped_data['tweets']:
        tweets_list.append({
            "date": t['date'],
            "tweet": t['text']
        })
    
    df = pd.DataFrame(tweets_list)
    if not df.empty:
        # Normalize Nitter's date format for plotting
        df['date'] = pd.to_datetime(df['date'].str.replace('·', '').strip(), errors='coerce')
    return df

# ----------------------------
# 5. User Interface & Analysis
# ----------------------------
tweet_limit = st.sidebar.slider("Number of Live Tweets to Fetch", 10, 100, 30)

if st.button("Refresh Sentiment Now"):
    with st.spinner("Fetching live data from X/Twitter..."):
        df_tweets = scrape_kelantan_tweets(tweet_limit)

    if not df_tweets.empty:
        # Analyze Scraped Data
        df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
        df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))
        
        # Layout Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Scraped Tweets & Predictions")
            st.dataframe(df_tweets[["date", "tweet", "sentiment"]], use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment Breakdown")
            fig_pie = px.pie(df_tweets, names="sentiment", hole=0.4, color="sentiment",
                             color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'gray'})
            st.plotly_chart(fig_pie, use_container_width=True)

        # Time Trend Analysis
        st.subheader("📈 Sentiment Over Time")
        df_tweets['hour'] = df_tweets['date'].dt.floor('h')
        trend = df_tweets.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        fig_trend = px.line(trend, x='hour', y='count', color='sentiment', markers=True, title="Hourly Sentiment Volume")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.warning("The scraper returned no results. This can happen if Nitter instances are busy. Please try again in a moment.")
