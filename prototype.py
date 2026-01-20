import streamlit as st
import pandas as pd
import re
import time
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Kelantan Social Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Kelantan Social Sentiment Dashboard")

# =====================================
# TEXT PREPROCESSING
# =====================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()

def is_kelantan_related(text):
    keywords = [
        "kelantan", "kelate", "kota bharu", "kb", "tumpat",
        "pasir mas", "tanah merah", "machang", "kuala krai",
        "gua musang", "pasir puteh", "bachok",
        "nasi kerabu", "nasi dagang", "orang kelate"
    ]
    return any(k in text.lower() for k in keywords)

# =====================================
# LOAD & TRAIN MODEL
# =====================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet", "majority_sent"])

    df["clean_text"] = df["comment/tweet"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["majority_sent"]

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_and_train()

# =====================================
# LOAD DATASET
# =====================================
@st.cache_data
def load_kelantan_data():
    df = pd.read_csv("kelantan_extended.csv")
    df = df.dropna(subset=["comment/tweet"])

    df["date"] = pd.date_range(
        end=datetime.now(),
        periods=len(df),
        freq="1H"
    )

    df = df.rename(columns={"comment/tweet": "tweet"})
    df = df[df["tweet"].apply(is_kelantan_related)]

    return df[["date", "tweet"]].reset_index(drop=True)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Controls")
forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

# =====================================
# RUN ANALYSIS
# =====================================
if st.sidebar.button("🔄 Run Analysis"):

    df = load_kelantan_data()

    df["clean_text"] = df["tweet"].apply(clean_text)
    df["sentiment"] = model.predict(
        vectorizer.transform(df["clean_text"])
    )

    # =====================================
    # SENTIMENT DISTRIBUTION
    # =====================================
    st.subheader("📊 Sentiment Distribution")

    fig_pie = px.pie(
        df,
        names="sentiment",
        hole=0.4,
        color="sentiment",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral": "#f1c40f",
            "negative": "#e74c3c"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # TREND (ACTUAL)
    # =====================================
    st.subheader("📈 Sentiment Trend (Actual)")

    df["date_only"] = df["date"].dt.date

    daily = (
        df.groupby(["date_only", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig_trend = px.line(
        daily,
        x="date_only",
        y="count",
        color="sentiment",
        markers=True
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # =====================================
    # 🔮 SENTIMENT TREND PREDICTION
    # =====================================
    st.subheader("🔮 Sentiment Trend Prediction")

    future_dates = pd.date_range(
        start=daily["date_only"].max() + timedelta(days=1),
        periods=forecast_days
    )

    fig_pred = go.Figure()

    for sentiment in daily["sentiment"].unique():

        df_s = daily[daily["sentiment"] == sentiment].copy()
        df_s["t"] = range(len(df_s))

        X = df_s[["t"]]
        y = df_s["count"]

        model_lr = LinearRegression()
        model_lr.fit(X, y)

        future_t = range(len(df_s), len(df_s) + forecast_days)
        y_pred = model_lr.predict(pd.DataFrame({"t": future_t}))

        fig_pred.add_trace(go.Scatter(
            x=df_s["date_only"],
            y=df_s["count"],
            mode="lines+markers",
            name=f"{sentiment} (Actual)"
        ))

        fig_pred.add_trace(go.Scatter(
            x=future_dates,
            y=y_pred,
            mode="lines+markers",
            line=dict(dash="dash"),
            name=f"{sentiment} (Predicted)"
        ))

    fig_pred.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        hovermode="x unified"
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # =====================================
    # ⏳ SENTIMENT TIMELINE (ACTUAL + FUTURE)
    # =====================================
    st.subheader("⏳ Sentiment Timeline Prediction")

    timeline_fig = go.Figure()

    for sentiment in daily["sentiment"].unique():

        df_s = daily[daily["sentiment"] == sentiment].copy()
        df_s["t"] = range(len(df_s))

        X = df_s[["t"]]
        y = df_s["count"]

        lr = LinearRegression()
        lr.fit(X, y)

        future_t = list(range(len(df_s), len(df_s) + forecast_days))
        future_y = lr.predict(pd.DataFrame({"t": future_t}))

        combined_dates = list(df_s["date_only"]) + list(future_dates)
        combined_counts = list(df_s["count"]) + list(future_y)

        timeline_fig.add_trace(go.Scatter(
            x=combined_dates,
            y=combined_counts,
            mode="lines+markers",
            name=f"{sentiment} Timeline"
        ))

    timeline_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        hovermode="x unified"
    )

    st.plotly_chart(timeline_fig, use_container_width=True)

    # =====================================
    # METRICS
    # =====================================
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tweets", len(df))
    col2.metric("Positive %", f"{(df.sentiment=='positive').mean()*100:.1f}%")
    col3.metric("Negative %", f"{(df.sentiment=='negative').mean()*100:.1f}%")

else:
    st.info("👈 Click **Run Analysis** to start sentiment forecasting")
