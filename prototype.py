import streamlit as st
import pandas as pd
import re
from ntscraper import Nitter 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px

# ----------------------------
# 1. Text Preprocessing (Moved outside to fix Pickle Error)
# ----------------------------
def clean_text(text):
    """Clean the text for sentiment analysis."""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    # Remove URLs, handles, and hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    # Keep only letters and apostrophes
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()

# ----------------------------
# 2. Scraper Setup
# ----------------------------
@st.cache_resource
def get_nitter_scraper():
    return Nitter(log_level=1)

scraper = get_nitter_scraper()

# ----------------------------
# 3. Model Training (Caches only the vectorizer and model)
# ----------------------------
@st.cache_data
def load_and_train():
    try:
        # Load dataset
        df_train = pd.read_csv("prototaip.csv")
        
        # Remove empty rows to prevent "NaN" error
        df_train = df_train.dropna(subset=['comment/tweet', 'majority_sent'])
        
        # Apply cleaning using the top-level function
        df_train["clean_text"] = df_train["comment/tweet"].apply(clean_text)
        
        # Feature Extraction
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(df_train["clean_text"])
        y_train = df_train["majority_sent"]

        # Train Model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Return only pickleable objects
        return vectorizer, model
    except FileNotFoundError:
        st.error("File 'prototaip.csv' not found. Please ensure it is in your project folder.")
        st.stop()

# Load the model and vectorizer
vectorizer, model = load_and_train()

# ----------------------------
# 4. Page Config & UI
# ----------------------------
st.set_page_config(page_title="Kelantan Social Sentiment Dashboard", layout="wide")
st.title("📊 Real‑Time Sentiment Analysis Dashboard — Kelantan")

def scrape_kelantan_tweets(limit):
    query = "Kelantan OR 'orang kelantan'"
    scraped_data = scraper.get_tweets(query, mode='term', number=limit, language='ms')
    
    tweets_list = []
    for t in scraped_data['tweets']:
        tweets_list.append({"date": t['date'], "tweet": t['text']})
    
    df = pd.DataFrame(tweets_list)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'].str.replace('·', '').strip(), errors='coerce')
    return df

# ----------------------------
# 5. Dashboard Analysis
# ----------------------------
tweet_limit = st.sidebar.slider("Number of Live Tweets", 10, 100, 30)

if st.button("Refresh Sentiment Now"):
    with st.spinner("Fetching live data..."):
        df_tweets = scrape_kelantan_tweets(tweet_limit)

    if not df_tweets.empty:
        # Prediction
        df_tweets["clean_text"] = df_tweets["tweet"].apply(clean_text)
        df_tweets["sentiment"] = model.predict(vectorizer.transform(df_tweets["clean_text"]))
        
        # Display Data
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Live Analysis Results")
            st.dataframe(df_tweets[["date", "tweet", "sentiment"]], use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment Distribution")
            fig_pie = px.pie(df_tweets, names="sentiment", hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Timeline
        st.subheader("📈 Trend Over Time")
        df_tweets['hour'] = df_tweets['date'].dt.floor('h')
        trend = df_tweets.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        fig_trend = px.line(trend, x='hour', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No live tweets found. Try increasing the limit or checking your internet connection.")
