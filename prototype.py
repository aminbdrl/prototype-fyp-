import streamlit as st
import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
from datetime import datetime, timedelta

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Kelantan Social Unity Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Kelantan Social Unity Sentiment Analysis")
st.caption("Dataset: SEACrowd Malaysia Tweets (Hugging Face)")

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
# KELANTAN KEYWORD FILTER
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
# LOAD DATASET & TRAIN MODEL
# =====================================
@st.cache_data(show_spinner=True)
def load_and_train():
    dataset = load_dataset(
        "SEACrowd/malaysia_tweets",
        trust_remote_code=True
    )

    df = dataset["train"].to_pandas()
    df = df.dropna(subset=["text", "label"])

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return df, vectorizer, model

with st.spinner("Loading SEACrowd dataset & training model..."):
    df_all, vectorizer, model = load_and_train()

# Map labels
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =====================================
# SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Analysis Controls")

tweet_limit = st.sidebar.slider(
    "Number of Tweets",
    50, 500, 200
)

time_window = st.sidebar.selectbox(
    "Time Window",
    options=[12, 24, 48, 72, 168],
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

    df = df_all.copy()

    # Filter Kelantan tweets
    df = df[df["text"].apply(is_kelantan_related)]

    if df.empty:
        st.warning("No Kelantan-related tweets found.")
        st.stop()

    # Limit data
    df = df.head(tweet_limit)

    # Simulate time (short-term only)
    df["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df),
        freq="H"
    )

    # Predict sentiment
    df["predicted_label"] = model.predict(
        vectorizer.transform(df["clean_text"])
    )
    df["sentiment"] = df["predicted_label"].map(LABEL_MAP)

    # =====================================
    # METRICS
    # =====================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tweets", len(df))
    with col2:
        st.metric("Positive",
                  f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
    with col3:
        st.metric("Negative",
                  f"{(df['sentiment']=='Negative').mean()*100:.1f}%")

    # =====================================
    # TABLE
    # =====================================
    st.subheader("📋 Kelantan Tweet Sentiment")
    st.dataframe(
        df[["date", "text", "sentiment"]],
        use_container_width=True
    )

    # =====================================
    # PIE CHART
    # =====================================
    st.subheader("📊 Sentiment Distribution")

    fig_pie = px.pie(
        df,
        names="sentiment",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # TREND
    # =====================================
    st.subheader("📈 Sentiment Trend")

    if time_window <= 48:
        df["time_group"] = df["date"].dt.floor("H")
    else:
        df["time_group"] = df["date"].dt.floor("D")

    trend = (
        df.groupby(["time_group", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig_trend = px.line(
        trend,
        x="time_group",
        y="count",
        color="sentiment",
        markers=True
    )

    st.plotly_chart(fig_trend, use_container_width=True)

else:
    st.info("👈 Click **Run Analysis** to start")

    with st.expander("ℹ️ About This System"):
        st.markdown("""
**Kelantan Social Unity Sentiment Analysis Dashboard**

• Dataset: **SEACrowd Malaysia Tweets (Hugging Face)**  
• Approach: **Supervised Machine Learning**  
• Model: **TF-IDF + Multinomial Naive Bayes**  
• Focus: **Kelantan-related social sentiment**  
• Time Window: **Short-term (12 hours – 7 days)**  

This system uses a pre-labelled Malaysian Twitter dataset and applies
rule-based keyword filtering to isolate Kelantan-related discussions.
""")
