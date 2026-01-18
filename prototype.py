import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from wordcloud import WordCloud
import plotly.express as px

# =====================================
# 1. PAGE CONFIG & STYLING
# =====================================
st.set_page_config(page_title="Kelantan Social Unity Hub", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# =====================================
# 2. KEYWORDS & MAPPINGS
# =====================================
KELANTAN_KEYWORDS = [
    "kelantan", "kelate", "kota bharu", "kb", "tumpat", "pasir mas", 
    "tanah merah", "machang", "kuala krai", "gua musang", "pasir puteh", 
    "bachok", "jeli", "nasi kerabu", "nasi dagang", "laksam", "budu",
    "oghe kelate", "demo kelate", "sek kito", "mace mano"
]

TOPIC_NAMES = {
    0: "Politics",
    1: "Social/Unity",
    2: "Economy",
    3: "Entertainment",
    4: "Sports",
    5: "Technology"
}

MALAY_STOPWORDS = set([
    "yang", "dan", "di", "itu", "ini", "untuk", "dengan", "adalah", "ada", 
    "ke", "dari", "sebagai", "akan", "dalam", "bisa", "saya", "dia", "mereka",
    "buat", "tahu", "pun", "tak", "nak", "je", "la", "kalo", "u", "dah", "kita"
])

# =====================================
# 3. CORE FUNCTIONS
# =====================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def is_kelantan(text):
    text = text.lower()
    return any(k in text for k in KELANTAN_KEYWORDS)

@st.cache_resource
def load_models():
    # Load Malay Sentiment BERT
    # LABEL_0: Neg, LABEL_1: Neu, LABEL_2: Pos
    sentiment_pipe = pipeline("sentiment-analysis", model="rmtariq/ft-Malay-bert")
    return sentiment_pipe

@st.cache_data
def prepare_data():
    dataset = load_dataset("malaysia-ai/malaysian-twitter-by-topics")
    df = dataset["train"].to_pandas()
    if "text" in df.columns:
        df = df.rename(columns={"text": "tweet", "label": "topic"})
    df["clean_tweet"] = df["tweet"].apply(clean_text)
    return df

@st.cache_data
def train_topic_model(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_tweet"])
    y = df["topic"]
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

# =====================================
# 4. INITIALIZATION
# =====================================
st.title("📊 Kelantan Social Unity Analysis")
st.info("Uses Malay BERT for sentiment and MultinomialNB for topic classification.")

with st.spinner("Initializing AI Models... (First run may take a minute)"):
    sentiment_pipeline = load_models()
    df_all = prepare_data()
    vectorizer, topic_model = train_topic_model(df_all)

# =====================================
# 5. SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Settings")
max_tweets = st.sidebar.slider("Number of Kelantan Tweets", 20, 500, 100)
selected_topic = st.sidebar.multiselect("Filter by Topic", options=list(TOPIC_NAMES.values()), default=list(TOPIC_NAMES.values()))

# =====================================
# 6. EXECUTION LOGIC
# =====================================
if st.sidebar.button("Run Analysis"):
    # Filter for Kelantan
    df_kelantan = df_all[df_all["tweet"].apply(is_kelantan)].copy()
    
    if df_kelantan.empty:
        st.warning("No Kelantan-related tweets found in this sample.")
    else:
        # Limit data
        df_kelantan = df_kelantan.head(max_tweets)

        # Predict Topics
        topic_preds = topic_model.predict(vectorizer.transform(df_kelantan["clean_tweet"]))
        df_kelantan["topic_name"] = [TOPIC_NAMES.get(p, "General") for p in topic_preds]

        # Predict Sentiment (Malay BERT)
        with st.spinner("BERT is analyzing Malay sentiment..."):
            def get_bert_sentiment(t):
                res = sentiment_pipeline(t[:512])[0]
                m = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
                return m.get(res['label'], "Neutral")
            
            df_kelantan["sentiment"] = df_kelantan["tweet"].apply(get_bert_sentiment)

        # Apply Topic Filter from Sidebar
        df_final = df_kelantan[df_kelantan["topic_name"].isin(selected_topic)]

        # --- METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", len(df_final))
        c2.metric("Positive %", f"{(df_final['sentiment']=='Positive').mean()*100:.1f}%")
        c3.metric("Neutral %", f"{(df_final['sentiment']=='Neutral').mean()*100:.1f}%")
        c4.metric("Negative %", f"{(df_final['sentiment']=='Negative').mean()*100:.1f}%")

        # --- VISUALIZATIONS ---
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Sentiment Share")
            fig_pie = px.pie(df_final, names="sentiment", color="sentiment",
                             color_discrete_map={"Positive":"#2ecc71", "Neutral":"#95a5a6", "Negative":"#e74c3c"})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Topic Distribution")
            fig_bar = px.bar(df_final.groupby(["topic_name", "sentiment"]).size().reset_index(name='count'), 
                             x="topic_name", y="count", color="sentiment", barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- WORD CLOUDS ---
        st.divider()
        st.subheader("☁️ Word Clouds (Positive vs Negative)")
        wc_pos_col, wc_neg_col = st.columns(2)
        
        for sent, col, color in [("Positive", wc_pos_col, "Greens"), ("Negative", wc_neg_col, "Reds")]:
            text = " ".join(df_final[df_final["sentiment"] == sent]["clean_tweet"])
            if text:
                wc = WordCloud(background_color="white", colormap=color, stopwords=MALAY_STOPWORDS, width=600, height=300).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                col.pyplot(fig)
            else:
                col.info(f"Not enough {sent} data.")

        # --- DATA TABLE ---
        st.subheader("📋 Detailed Tweet Log")
        st.dataframe(df_final[["tweet", "topic_name", "sentiment"]], use_container_width=True)

else:
    st.markdown("### 👈 Configure settings and click 'Run Analysis' to start.")
    st.image("https://img.icons8.com/clouds/200/000000/analysis.png", width=100)
