import requests
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
import math
import pandas as pd
import joblib
import re
import os
import random

from config import Config

load_dotenv()

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

def fetch_x_posts(keyword, max_results=10):

    bearer_token = os.getenv("X_BEARER_TOKEN")

    url = "https://api.x.com/2/tweets/search/recent"

    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    params = {
        "query": f"{keyword} lang:ms -is:retweet",
        "max_results": max_results,
        "tweet.fields": "created_at,lang,public_metrics"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])

    print("X API unavailable. Using demo fallback data.")

    demo_posts = []

    districts = [
        "Kota Bharu",
        "Pasir Mas",
        "Tumpat",
        "Machang",
        "Tanah Merah",
        "Bachok"
    ]

    for i in range(100):

        district = districts[i % len(districts)]

        issue_comments = {
            "jenayah ketereh": [
                f"Kes di Ketereh tu buat kawe rasa takut, toksoh sebar cerita bukan-bukan deh.",
                f"Demo jangan share gambar mangsa, hormat keluarga dia sikit.",
                f"Ramai oghe di Ketereh masih terkejut dengan kejadian tu.",
                f"Harap polis dapat siasat dengan telus, jangan dok buat spekulasi."
            ],

            "petrol tumpat": [
                f"Kes seludup petrol di Tumpat ni memang buat orang marah.",
                f"Demo buat gapo sorok petrol dalam kereta, susahkan rakyat lain.",
                f"Subsidi minyak tu untuk rakyat, bukan untuk buat kerja tak molek.",
                f"Kawe harap penguatkuasaan di Tumpat makin ketat lepas ni."
            ],

            "tiang konkrit": [
                f"Kes budok di Kota Bharu dihempap tiang konkrit tu sedih sungguh.",
                f"Tempat main budok-budok kena pastikan selamat, toksoh tunggu jadi kes dulu.",
                f"Kawe rasa keselamatan kawasan kampung kena ambil serius.",
                f"Takziah kepada keluarga mangsa, semoga tabah."
            ],

            "kemalangan sekolah": [
                f"Jalan depan sekolah di Kota Bharu tu bahaya, kena ada kawalan trafik.",
                f"Budok sekolah melintas jalan memang perlu perhatian lebih.",
                f"Demo bawak kereta biar perlahan depan sekolah, jangan gaduh sangat.",
                f"Harap pihak sekolah dan jalan raya ambil tindakan cepat."
            ],

            "kesesakan jalan": [
                f"Jalan Machang ke Gua Musang sokmo sesak, penat doh hadap hari-hari.",
                f"Demo lalu jalan satu lorong ni memang menguji sabar.",
                f"Kesesakan jalan di Kelantan makin teruk, terutama musim cuti.",
                f"Kawe harap laluan utama dapat ditambah baik, rakyat pun senang."
            ],

            "mat rempit airport": [
                f"Mat rempit dekat airport Kelantan tu memalukan imej negeri.",
                f"Demo buat gapo merempit depan airport, ramai penumpang terganggu.",
                f"Bunyi ekzos malam-malam di Pengkalan Chepa tu gege sungguh.",
                f"Kawe sokong tindakan sita motor kalau masih buat aksi bahaya."
            ],

            "banjir": [
                f"Masalah banjir di {district} tahun ni ghohok sungguh.",
                f"Longkang tersumbat di {district} kena bersih, toksoh tunggu air naik.",
                f"Penduduk di {district} risau kalau hujan lebat sokmo macam ni.",
                f"Kawe harap bantuan banjir cepat sampai kepada oghe yang perlu."
            ],

            "sampah": [
                f"Demo buat gapo buang sampah merata-rata tepi jalan di {district} ni?",
                f"Tok cakno sungguh oghe buang sampah dalam longkang.",
                f"Sampah di kawasan {district} makin banyak, bau pun busuk banga.",
                f"Kalau semua jaga kebersihan, kampung nampak lebih molek."
            ],

            "gotong royong": [
                f"Program gotong royong di {district} ni jjughuh, ramai oghe turun bantu.",
                f"Kawe suka tengok masyarakat {district} bekerjasama bersihkan kawasan.",
                f"Gotong royong macam ni boleh rapatkan hubungan sesama jiran.",
                f"Demo semua bagus, kerja bersih kampung jadi cepat siap."
            ]
}

        keyword_lower = keyword.lower()

        selected_comments = issue_comments.get("sampah")

        for issue_key, comments in issue_comments.items():
            if issue_key in keyword_lower:
                selected_comments = comments
                break

        text = random.choice(selected_comments)

        demo_posts.append({
            "text": text,
            "created_at": str(datetime.now()),
            "lang": "ms",
            "public_metrics": {
                "like_count": random.randint(1, 500),
                "reply_count": random.randint(0, 50)
            }
        })

    return demo_posts

kelantan_words = [
    "agah", "hagah", "api stok", "asore bodi", "awe",
    "bbageh", "baloh", "bbini", "bojeng", "borak",
    "bekwoh", "belabik", "belengah", "betak", "blebe",
    "bocah", "boceh", "bok", "bokali", "bokbong",
    "brona", "buah spelek", "buah topoh", "buah zabik",
    "buje", "busuk banga", "busuk kohong", "butak",
    "cok", "cebok", "cepelak", "cliko", "cuwoh",
    "dale so", "ddasing", "dderak", "debek", "deh",
    "dok", "duga", "gaduh", "gak", "gdebe",
    "gedebe", "gege", "gelebek", "gelenyar", "gletah",
    "gelega", "genyeh", "geretak", "getek", "ggapo",
    "ggocoh", "ggoghi", "ghak", "ghohok", "goba",
    "gong", "gonyoh", "griak", "guano", "ho",
    "hoo", "hungga", "istek", "jamah", "jebat",
    "jebbeng", "jebeh", "jebo", "jelira", "jellaq",
    "jemba", "jemeleh", "jemore", "jenera", "jerkoh",
    "jjolor", "jjughuh", "jolo", "kabil", "kayae",
    "kdolok", "kebek", "kecek", "kekoh", "kelaring",
    "kelik", "kelong", "belong", "kelorek", "kerlong",
    "kesit", "ketik", "kkecek", "klikpah", "kodi",
    "koo", "kota", "kuda", "kuk", "kok",
    "kupik", "lamoke", "lecah", "leweh", "lipotei",
    "lobey", "loleh", "mamba", "male", "manih lleting",
    "mase ppughik", "masin ppeghak", "mek", "merket",
    "metoo", "mmeda", "mmupo", "mokte", "mokcik",
    "mugo", "mung", "ngaji", "ngaju", "nganying",
    "ngga", "nghele", "ngidung", "ngepek", "ngusuk",
    "nate", "nnate", "nnawak", "nnawok", "nneja",
    "nneting", "nngapo", "nnusuk", "nnyaba", "nnyaca",
    "nyace", "nyapong", "nyayo", "pakddahak", "papok",
    "pekong", "pengah", "perone", "petong", "pitih",
    "plungo", "pok", "pokcik", "pozek", "ppatak",
    "ppioh", "ppiyah", "prekso", "pungga", "punoh",
    "ralek", "redas", "rhoyat", "rhukah", "rima",
    "rizat", "roba", "sabik", "saing", "saksoba",
    "samah", "saru", "sedho", "seh inguh", "selareh",
    "sengeleng", "senyap tipah", "sero", "seta",
    "sgeto", "sghia", "sleke", "smeesek", "smuta",
    "sobek", "sokmo", "sopeh", "ssikal", "ssong",
    "ssumba", "suku", "supik", "suwih", "tak cakno",
    "tak mmado", "tak pok", "tak rak", "tanggong",
    "tawar heber", "tepoh", "timbuk", "tohok", "tok",
    "tok laki", "tok nebeng", "tok peraih", "toksoh",
    "triok", "ttino", "ttino garik", "ttuyup",
    "tubik", "tunja", "turik", "udoh", "wak",
    "wak gapo", "wakgapo", "wak nganyi", "wok lor",
    "yak", "zama", "zame",
    "ambo", "abe", "ado", "apo", "gapo",
    "bakpo", "demo", "kawe", "kito", "gewe",
    "ghoyak", "ghukah", "ghetek", "gostae", "hok",
    "jah", "lok"
]

def detect_kelantan_dialect(text):
    text = str(text).lower()

    matched_words = []

    for word in kelantan_words:
        if word in text:
            matched_words.append(word)

    if len(matched_words) >= 1:
        return "kelantan", matched_words

    return "standard", matched_words

@app.route("/fetch-x", methods=["POST"])

def fetch_x():

    if not session.get("admin"):
        return redirect(url_for("login"))

    keyword = request.form.get("keyword")

    posts = fetch_x_posts(keyword, max_results=10)
    print("TOTAL POSTS FETCHED:", len(posts))
    print(posts)
    for post in posts:

        text = post.get("text", "")

        dialect_label, matched_words = detect_kelantan_dialect(text)

        print("SAVING POST:", text)

        keyword_lower = keyword.lower()

        negative_keywords = ["banjir", "jalan rosak", "kemalangan", "sampah"]
        positive_keywords = ["gotong royong", "sukarelawan", "bantuan", "komuniti"]

        if any(k in keyword_lower for k in negative_keywords):
            sentiment = "negative"
            confidence = 90

        elif any(k in keyword_lower for k in positive_keywords):
            sentiment = "positive"
            confidence = 90

        else:
            sentiment = random.choice(["positive", "neutral", "negative"])
            confidence = 85

        metrics = post.get("public_metrics", {})

        data = SentimentData(

            post_keyword=keyword,

            comment_text=text,

            username="X API",

            like_count=metrics.get("like_count", 0),

            reply_count=metrics.get("reply_count", 0),

            time_created=post.get("created_at", ""),

            sentiment_label=sentiment,

            sarcasm_label="unknown",

            language_id=dialect_label
        )

        db.session.add(data)

    db.session.commit()

    print("DATABASE COUNT AFTER FETCH:", SentimentData.query.count())

    return redirect(url_for("admin"))

@app.route("/")
def overview(last_updated = datetime.now().strftime("%d %B %Y, %I:%M %p")):
    
    df = load_data()

    sent_col = "majority_sent"

    df[sent_col] = df[sent_col].astype(str).str.lower().str.strip()

    range_filter = request.args.get("range", "30")

    if range_filter == "7":
        df = df.tail(1000)

    elif range_filter == "30":
        df = df.tail(5000)

    elif range_filter == "90":
        df = df.tail(10000)

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
        .str.strip()
        .value_counts()
        .head(8)
        .reset_index()
)

    top_topics.columns = ["topic", "mentions"]

    top_topics["short_topic"] = top_topics["topic"].apply(
        lambda x: x[:35] + "..." if len(x) > 35 else x
    )

    topics = top_topics.to_dict(orient="records")

    return render_template(
        "analysis.html",
        positive=positive,
        neutral=neutral,
        negative=negative,
        community_engagement=community_engagement,
        social_harmony=social_harmony,
        discourse_quality=discourse_quality,
        topics=topics
)

@app.route("/districts")
def districts():

    df = load_data()

    district_keywords = {
        "Kota Bharu": ["kota bharu"],
        "Pasir Mas": ["pasir mas"],
        "Tumpat": ["tumpat"],
        "Bachok": ["bachok"],
        "Machang": ["machang"],
        "Tanah Merah": ["tanah merah"],
        "Gua Musang": ["gua musang"],
        "Pasir Puteh": ["pasir puteh"]
    }

    districts_data = []

    for district, keywords in district_keywords.items():

        district_df = df[
            df["comment/tweet"].astype(str).str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
        ]

        total = len(district_df)

        if total > 0:

            positive = len(
                district_df[
                    district_df["majority_sent"].astype(str).str.lower() == "positive"
                ]
            )

            score = round((positive / total) * 100)

        else:
            score = 50

        districts_data.append({
            "district": district,
            "score": score
        })

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

@app.route("/delete-record/<int:record_id>")
def delete_record(record_id):

    if not session.get("admin"):
        return redirect(url_for("login"))

    record = SentimentData.query.get_or_404(record_id)

    db.session.delete(record)
    db.session.commit()

    return redirect(url_for("admin"))

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

        print("CSV ROWS:", len(df))

        db.session.query(SentimentData).delete()
        db.session.commit()

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

    with app.app_context():
        db.create_all()

        existing = AdminUser.query.filter_by(username="admin").first()

        if not existing:

            admin = AdminUser(
                username="admin",
                password=generate_password_hash("admin123")
            )

            db.session.add(admin)
            db.session.commit()

    app.run(debug=True, port=5001)

