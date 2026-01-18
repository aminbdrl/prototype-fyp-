import streamlit as st
import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from textblob import TextBlob

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Kelantan Social Unity Sentiment Analysis",
    layout="wide"
)

st.title("📊 Kelantan Social Unity Sentiment Analysis")
st.caption("Dataset: Malaysian Twitter by Topics (Hugging Face)")

# =====================================
# TEXT CLEANING
# =====================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# =====================================
# KELANTAN KEYWORDS
# =====================================
KELANTAN_KEYWORDS = [
    "kelantan", "kelate", "kota bharu", "kotabharu", "kb",
    "tumpat", "pasir mas", "tanah merah", "machang",
    "kuala krai", "gua musang", "pasir puteh", "bachok",
    "jeli", "wakaf che yeh", "pcb",
    "nasi kerabu", "nasi dagang", "laksam", "budu",
    "orang kelate", "oghe kelate", "demo kelate"
]

def is_kelantan(text):
    text = text.lower()
    return any(k in text for k in KELANTAN_KEYWORDS)

# =====================================
# LOAD DATASET
# =====================================
@st.cache_data
def load_data():
    dataset = load_dataset("malaysia-ai/malaysian-twitter-by-topics")
    df = dataset["train"].to_pandas()

    # Inspect columns
    st.write("Dataset columns:", df.columns.tolist())

    # Rename for consistency
    if "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"text": "tweet", "label": "topic"})
    else:
        st.error("Expected 'text' and 'label' columns not found in dataset")
        st.stop()

    df["clean_tweet"] = df["tweet"].apply(clean_text)
    return df

# =====================================
# TRAIN TOPIC MODEL
# =====================================
@st.cache_data
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_tweet"])
    y = df["topic"]

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

# =====================================
# SIMPLE SENTIMENT FUNCTION (TextBlob)
# =====================================
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

with st.spinner("Loading dataset & training model..."):
    df_all = load_data()
    vectorizer, topic_model = train_model(df_all)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Controls")
max_tweets = st.sidebar.slider("Number of Tweets", 50, 300, 150)

# =====================================
# ANALYSIS
# =====================================
if st.sidebar.button("Run Analysis"):

    df_kelantan = df_all[df_all["tweet"].apply(is_kelantan)]

    if df_kelantan.empty:
        st.warning("No Kelantan-related tweets found.")
        st.stop()

    df_kelantan = df_kelantan.head(max_tweets)

    # Topic prediction
    df_kelantan["topic_prediction"] = topic_model.predict(
        vectorizer.transform(df_kelantan["clean_tweet"])
    )

    # Sentiment scoring
    df_kelantan["sentiment_label"] = df_kelantan["clean_tweet"].apply(get_sentiment)

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tweets", len(df_kelantan))
    c2.metric("Positive (%)", f"{(df_kelantan['sentiment_label']=='Positive').mean()*100:.1f}")
    c3.metric("Negative (%)", f"{(df_kelantan['sentiment_label']=='Negative').mean()*100:.1f}")

    # TABLE
    st.subheader("📋 Kelantan Tweets")
    st.dataframe(
        df_kelantan[["tweet", "topic_prediction", "sentiment_label"]],
        use_container_width=True
    )

    # PIE CHART
    st.subheader("📊 Sentiment Distribution")
    fig = px.pie(df_kelantan, names="sentiment_label", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Click **Run Analysis** to start")

    with st.expander("ℹ️ About"):
        st.markdown("""
**Kelantan Social Unity Sentiment Analysis**

• Dataset: Malaysian Twitter by Topics  
• Models: TF-IDF + Naive Bayes (topics) + TextBlob (sentiment)  
• Filtering: Keyword-based Kelantan detection  
• Purpose: Analyse both **topics** and **sentiment** in Kelantan-related tweets
""")
