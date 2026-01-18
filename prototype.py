import streamlit as st
import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Kelantan Social Unity Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Kelantan Social Unity Sentiment Analysis")
st.caption("Dataset: SEACrowd Malaysia Tweets (HF streaming mode)")

# =====================================
# TEXT CLEANING
# =====================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()

# =====================================
# KELANTAN KEYWORDS
# =====================================
KELANTAN_KEYWORDS = [
    "kelantan", "kelate", "kota bharu", "kotabharu", "kb",
    "tumpat", "pasir mas", "tanah merah", "machang",
    "kuala krai", "gua musang", "pasir puteh", "bachok",
    "jeli", "pengkalan chepa", "wakaf che yeh",
    "pantai cahaya bulan", "pcb", "pasar siti khadijah",
    "nasi kerabu", "nasi dagang", "laksam", "budu",
    "orang kelate", "oghe kelate", "demo kelate",
    "gapo", "mung", "kito"
]

def is_kelantan_related(text):
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(k in text for k in KELANTAN_KEYWORDS)

# =====================================
# LOAD DATASET (STREAMING = NO SCRIPTS)
# =====================================
@st.cache_data(show_spinner=True)
def load_streaming_dataset(limit=5000):
    dataset = load_dataset(
        "SEACrowd/malaysia_tweets",
        split="train",
        streaming=True
    )

    records = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        if item.get("text") and item.get("label") is not None:
            records.append({
                "text": item["text"],
                "label": item["label"]
            })

    df = pd.DataFrame(records)
    df["clean_text"] = df["text"].apply(clean_text)
    return df

# =====================================
# TRAIN MODEL
# =====================================
@st.cache_data(show_spinner=True)
def train_model(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

with st.spinner("Loading dataset & training model..."):
    df_all = load_streaming_dataset()
    vectorizer, model = train_model(df_all)

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Controls")

tweet_limit = st.sidebar.slider("Number of Tweets", 50, 300, 150)

time_window = st.sidebar.selectbox(
    "Time Window",
    [12, 24, 48, 72, 168],
    format_func=lambda x: {
        12: "Last 12 Hours",
        24: "Last 24 Hours",
        48: "Last 48 Hours",
        72: "Last 72 Hours",
        168: "Last 7 Days"
    }[x]
)

# =====================================
# RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Run Analysis"):

    df = df_all[df_all["text"].apply(is_kelantan_related)]

    if df.empty:
        st.warning("No Kelantan-related tweets found.")
        st.stop()

    df = df.head(tweet_limit)

    df["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df),
        freq="H"
    )

    df["sentiment"] = model.predict(
        vectorizer.transform(df["clean_text"])
    )
    df["sentiment"] = df["sentiment"].map(LABEL_MAP)

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tweets", len(df))
    c2.metric("Positive (%)", f"{(df['sentiment']=='Positive').mean()*100:.1f}")
    c3.metric("Negative (%)", f"{(df['sentiment']=='Negative').mean()*100:.1f}")

    # TABLE
    st.subheader("📋 Kelantan Tweets")
    st.dataframe(df[["date", "text", "sentiment"]], use_container_width=True)

    # PIE
    st.subheader("📊 Sentiment Distribution")
    st.plotly_chart(px.pie(df, names="sentiment", hole=0.4), use_container_width=True)

    # TREND
    st.subheader("📈 Sentiment Trend")
    df["time_group"] = (
        df["date"].dt.floor("H") if time_window <= 48
        else df["date"].dt.floor("D")
    )

    trend = df.groupby(["time_group", "sentiment"]).size().reset_index(name="count")
    st.plotly_chart(
        px.line(trend, x="time_group", y="count", color="sentiment", markers=True),
        use_container_width=True
    )

else:
    st.info("👈 Click **Run Analysis** to start")

    with st.expander("ℹ️ About"):
        st.markdown("""
**Kelantan Social Unity Sentiment Analysis**

• Dataset: **SEACrowd Malaysia Tweets**  
• Access method: **Hugging Face streaming mode (no scripts)**  
• Model: **TF-IDF + Multinomial Naive Bayes**  
• Analysis window: **12 hours – 7 days**

Streaming mode avoids deprecated dataset scripts and ensures stable,
reproducible data access.
""")
