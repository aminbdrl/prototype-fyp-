from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
import math
import pandas as pd
import joblib
import re
import os

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    input_text = db.Column(db.Text)

    prediction = db.Column(db.String(50))

    confidence = db.Column(db.Integer)
app.secret_key = 'fyp2_secret_key'

def load_data():

    data = SentimentData.query.all()

    rows = []

    for item in data:
        rows.append({
            "post/keyword": item.post_keyword,
            "comment/tweet": item.comment_text,
            "username": item.username,
            "like count": item.like_count,
            "reply count": item.reply_count,
            "time created": item.time_created,
            "majority_sent": item.sentiment_label,
            "majority_sarc": item.sarcasm_label,
            "lang_id": item.language_id
        })

    if len(rows) == 0:
        return pd.read_csv("data/kelantan_extended.csv")

    return pd.DataFrame(rows)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text):
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)[0]
        confidence = round(max(probabilities) * 100)
    else:
        confidence = 0

    return prediction, confidence
def save_prediction_log(text, result, confidence):
    new_log = PredictionLog(
        input_text=text,
        prediction=result,
        confidence=confidence
    )

    db.session.add(new_log)
    db.session.commit()


@app.route("/")
def overview(last_updated = datetime.now().strftime("%d %B %Y, %I:%M %p")):
    
    df = load_data()

    sent_col = "majority_sent"

    df[sent_col] = df[sent_col].astype(str).str.lower().str.strip()

    range_filter = request.args.get("range", "30")

    if range_filter == "7":
        df = df.head(1000)

    elif range_filter == "30":
        df = df.head(5000)

    elif range_filter == "90":
        df = df.head(10000)

    total = len(df)

    positive = len(df[df[sent_col] == "positive"])
    neutral = len(df[df[sent_col] == "neutral"])
    negative = len(df[df[sent_col] == "negative"])

    positive_percent = round((positive / total) * 100)
    neutral_percent = round((neutral / total) * 100)
    negative_percent = round((negative / total) * 100)

    unity_score = round(positive_percent + (neutral_percent * 0.5))
    
    positive_trend = [positive_percent - 5, positive_percent - 2, positive_percent, positive_percent + 3]
    neutral_trend = [neutral_percent + 3, neutral_percent + 1, neutral_percent, neutral_percent - 2]
    negative_trend = [negative_percent + 2, negative_percent + 1, negative_percent, negative_percent - 1]

    topic_col = "post/keyword"

    trending_topics = (
        df[topic_col]
        .astype(str)
        .value_counts()
        .head(8)
        .reset_index()
    )

    trending_topics.columns = ["topic", "mentions"]

    local_issues = trending_topics.to_dict(orient="records")
    if positive_percent > negative_percent:
        ai_summary = (
        "Public sentiment in Kelantan is generally positive. "
        "Most discussions show stable community engagement and social harmony."
    )
    else:
        ai_summary = (
        "Negative sentiment is increasing. "
        "Certain local issues may require attention from authorities."
    )
    return render_template(
        "overview.html",
        ai_summary=ai_summary,
        positive=positive_percent,
        neutral=neutral_percent,
        negative=negative_percent,
        unity_score=unity_score,
        local_issues=local_issues,
        range_filter=range_filter,
        last_updated=last_updated,
        positive_trend=positive_trend,
        neutral_trend=neutral_trend,
        negative_trend=negative_trend,)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    
    result = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("user_text")
        result, confidence = predict_sentiment(user_text)
        save_prediction_log(user_text, result, confidence)

    logs = PredictionLog.query.order_by(PredictionLog.id.desc()).limit(10).all()

    return render_template(
        "prediction.html",
        result=result,
        confidence=confidence,
        user_text=user_text,
        logs=logs
    
    )
class AdminUser(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(255))
class SentimentData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_keyword = db.Column(db.String(255))
    comment_text = db.Column(db.Text)
    username = db.Column(db.String(255))
    like_count = db.Column(db.Integer)
    reply_count = db.Column(db.Integer)
    time_created = db.Column(db.String(100))
    sentiment_label = db.Column(db.String(50))
    sarcasm_label = db.Column(db.String(50))
    language_id = db.Column(db.String(50))
@app.route("/analysis")
def analysis():
    df = load_data()

    sent_col = "majority_sent"
    topic_col = "post/keyword"

    df[sent_col] = df[sent_col].astype(str).str.lower().str.strip()

    total = len(df)
    positive = len(df[df[sent_col] == "positive"])
    neutral = len(df[df[sent_col] == "neutral"])
    negative = len(df[df[sent_col] == "negative"])

    community_engagement = min(100, round((total / 1000) * 100))
    social_harmony = round(((positive + neutral) / total) * 100)
    discourse_quality = round((neutral / total) * 100 + (positive / total) * 50)

    top_topics = (
        df[topic_col]
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_topics.columns = ["topic", "mentions"]
    topics = top_topics.to_dict(orient="records")

    return render_template(
        "analysis.html",
        community_engagement=community_engagement,
        social_harmony=social_harmony,
        discourse_quality=discourse_quality,
        positive=positive,
        neutral=neutral,
        negative=negative,
        topics=topics
    )

@app.route("/districts")
def districts():
    districts_data = [
        {"district": "Kota Bharu", "score": 76},
        {"district": "Pasir Mas", "score": 71},
        {"district": "Tumpat", "score": 79},
        {"district": "Bachok", "score": 68},
        {"district": "Machang", "score": 70},
        {"district": "Tanah Merah", "score": 74},
        {"district": "Gua Musang", "score": 72},
        {"district": "Pasir Puteh", "score": 69}
    ]

    return render_template(
        "districts.html",
        districts_data=districts_data
    )
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        admin = AdminUser.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password, password):
            session["admin"] = True
        return redirect(url_for("admin"))
    else:
        error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))

@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect(url_for("login"))

    df = load_data()

    search = request.args.get("search", "").lower().strip()

    if search:
        df = df[
            df["post/keyword"].astype(str).str.lower().str.contains(search, na=False) |
            df["comment/tweet"].astype(str).str.lower().str.contains(search, na=False) |
            df["username"].astype(str).str.lower().str.contains(search, na=False) |
            df["majority_sent"].astype(str).str.lower().str.contains(search, na=False) |
            df["lang_id"].astype(str).str.lower().str.contains(search, na=False)
        ]

    total_records = len(df)
    total_users = df["username"].nunique()
    total_topics = df["post/keyword"].nunique()
    total_languages = df["lang_id"].nunique()

    latest_data = df.tail(20).to_dict(orient="records")
    db_records = SentimentData.query.count()

    prediction_logs_count = PredictionLog.query.count()

    admin_count = AdminUser.query.count()
    return render_template(
        "admin.html",
        db_records=db_records,
        prediction_logs_count=prediction_logs_count,
        admin_count=admin_count,
        total_records=total_records,
        total_users=total_users,
        total_topics=total_topics,
        total_languages=total_languages,
        latest_data=latest_data,
        search=search
    )

@app.route("/clear-logs")
def clear_logs():
    if not session.get("admin"):
        return redirect(url_for("login"))

    PredictionLog.query.delete()
    db.session.commit()

    return redirect(url_for("admin"))

@app.route("/export")
def export_report():
    if not session.get("admin"):
        return redirect(url_for("login"))
    df = load_data()

    report_file = "data/export_report.csv"

    summary = df[["post/keyword", "comment/tweet", "username", "majority_sent", "lang_id"]]

    summary.to_csv(report_file, index=False)

    return send_file(report_file, as_attachment=True)
def safe_int(value):

    try:

        if pd.isna(value):
            return 0

        value = str(value).strip()

        if value in ["", "---", "-", "nan", "None"]:
            return 0

        return int(float(value))

    except:
        return 0
@app.route("/upload", methods=["POST"])
def upload():

    if not session.get("admin"):
        return redirect(url_for("login"))

    file = request.files.get("csv_file")

    if file and file.filename.endswith(".csv"):

        file.save("data/kelantan_extended.csv")

        df = pd.read_csv("data/kelantan_extended.csv")

        SentimentData.query.delete()

        for _, row in df.iterrows():

            data = SentimentData(

                post_keyword=str(row.get("post/keyword", "")),

                comment_text=str(row.get("comment/tweet", "")),

                username=str(row.get("username", "")),

                like_count=safe_int(row.get("like count", 0)),

                reply_count=safe_int(row.get("reply count", 0)),

                time_created=str(row.get("time created", "")),

                sentiment_label=str(row.get("majority_sent", "")),

                sarcasm_label=str(row.get("majority_sarc", "")),

                language_id=str(row.get("lang_id", ""))
            )

            db.session.add(data)

        db.session.commit()

    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True, port=5001)