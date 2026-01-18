import streamlit as st
import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Kelantan Social Unity Sentiment Analysis",
    layout="wide"
)

st.title("📊 Kelantan Social Unity Sentiment Analysis")
st.caption("Dataset: Malaysian Twitter Sentiment (Hugging Face)")

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
# LOAD DATASET (SCRIPT-FREE)
# =====================================
@st.cache_data
def load_data():
    dataset = load_dataset("malaysian-twitter-sentiment")
    df = dataset["train"].to_pandas()

    df = df.rename(columns={"text": "tweet", "label": "sentiment"})
    df["clean_tweet"] = df["tweet"].apply(clean_text)
    return df

# =====================================
# TRAIN MODEL
# =====================================
@st.cache_data
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_tweet"])
    y = df["sentiment"]

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

with st.spinner("Loading dataset & training model..."):
    df_all = load_data()
    vectorizer, model = train_model(df_all)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

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

    df_kelantan["prediction"] = model.predict(
        vectorizer.transform(df_kelantan["clean_tweet"])
    )
    df_kelantan["sentiment_label"] = df_kelantan["prediction"].map(LABEL_MAP)

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tweets", len(df_kelantan))
    c2.metric("Positive (%)", f"{(df_kelantan['sentiment_label']=='Positive').mean()*100:.1f}")
    c3.metric("Negative (%)", f"{(df_kelantan['sentiment_label']=='Negative').mean()*100:.1f}")

    # TABLE
    st.subheader("📋 Kelantan Tweets")
    st.dataframe(
        df_kelantan[["tweet", "sentiment_label"]],
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

• Dataset: Malaysian Twitter Sentiment  
• Model: TF-IDF + Naive Bayes  
• Filtering: Keyword-based Kelantan detection  
• Purpose: Analyse public sentiment related to social unity in Kelantan

This dataset was selected due to its stability and script-free access.
""")
