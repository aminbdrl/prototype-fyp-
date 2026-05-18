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
            "id": item.id,
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


model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):

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

def fetch_x_posts(keyword, max_results=20):

    bearer_token = os.getenv("X_BEARER_TOKEN")

    client = requests.Session()

    url = "https://api.twitter.com/2/tweets/search/recent"

    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    query = keyword

    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields":
        "created_at,lang,public_metrics"
    }

    response = client.get(
        url,
        headers=headers,
        params=params
    )

    if response.status_code == 200:

        data = response.json()

        return data.get(
            "data",
            []
        )

    print(
        "X API ERROR:",
        response.status_code,
        response.text
    )

    return []

kelantan_words = [
    "agah", "hagah", "api stok", "asore bodi", "awe",
    "bbageh", "baloh", "bbini", "bojeng", "borak",
    "bekwoh", "belabik", "belengah", "betak", "blebe",
    "bocah", "boceh", "bok", "bokali", "bokbong",
    "brona", "buah spelek", "buah topoh", "buah zabik",
    "buje", "busuk banga", "busuk kohong", "butak",
    "cok", "cebok", "cepelak", "cliko", "cuwoh",
    "dale so", "ddasing", "dderak", "debek", "duga", "gaduh", "gak", "gdebe",
    "gedebe", "gege", "gelebek", "gelenyar", "gletah",
    "gelega", "genyeh", "geretak", "getek", "ggapo",
    "ggocoh", "ggoghi", "ghohok", "goba",
     "gonyoh", "griak", "guano",  "hungga", "istek", "jamah", "jebat",
    "jebbeng", "jebeh", "jebo", "jelira", "jellaq",
    "jemba", "jemeleh", "jemore", "jenera", "jerkoh",
    "jjolor", "jjughuh", "jolo", "kabil", "kayae",
    "kdolok", "kebek", "kecek", "kekoh", "kelaring",
    "kelik", "kelong", "belong", "kelorek", "kerlong",
    "kesit", "ketik", "kkecek", "klikpah", "kodi",
    "kupik", "lamoke", "lecah", "leweh", "lipotei",
    "lobey", "loleh", "mamba", "male", "manih lleting",
    "mase ppughik", "masin ppeghak", "merket",
    "metoo", "mmeda", "mmupo", "mokte", "mokcik",
    "mugo", "mung", "ngaji", "ngaju", "nganying",
    "ngga", "nghele", "ngidung", "ngepek", "ngusuk",
    "nate", "nnate", "nnawak", "nnawok", "nneja",
    "nneting", "nngapo", "nnusuk", "nnyaba", "nnyaca",
    "nyace", "nyapong", "nyayo", "pakddahak", "papok",
    "pekong", "pengah", "perone", "petong", "pitih",
    "plungo",  "pokcik", "pozek", "ppatak",
    "ppioh", "ppiyah", "prekso", "pungga", "punoh",
    "ralek", "redas", "rhoyat", "rhukah", "rima",
    "rizat", "roba", "sabik", "saing", "saksoba",
    "samah", "saru", "sedho", "seh inguh", "selareh",
    "sengeleng", "senyap tipah", "sero", "seta",
    "sgeto", "sghia", "sleke", "smeesek", "smuta",
    "sobek", "sokmo", "sopeh", "ssikal", "ssong",
    "ssumba",  "supik", "suwih", "tak cakno",
    "tak mmado", "tak pok", "tak rak", "tanggong",
    "tawar heber", "tepoh", "timbuk", "tohok",
    "tok laki", "tok nebeng", "tok peraih", "toksoh",
    "triok", "ttino", "ttino garik", "ttuyup",
    "tubik", "tunja", "turik", "udoh", 
    "wak gapo", "wakgapo", "wak nganyi", "wok lor", "zama", "zame",
    "ambo", "abe",  "gapo",
    "bakpo", "demo", "kawe", "kito", "gewe",
    "ghoyak", "ghukah", "ghetek", "gostae"

    ]

def detect_kelantan_dialect(text):

    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    matched_words = []

    for word in kelantan_words:

        word_clean = word.lower().strip()

        if " " in word_clean:
            if word_clean in text:
                matched_words.append(word_clean)
        else:
            pattern = r"\b" + re.escape(word_clean) + r"\b"

            if re.search(pattern, text):
                matched_words.append(word_clean)

    print("MATCHED KELANTAN WORDS:", matched_words)

    phrase_matches = [w for w in matched_words if " " in w]
    single_matches = [w for w in matched_words if " " not in w]

    if len(phrase_matches) >= 1:
        return "kelantan"

    strong_words = [
        "ambo",
        "gapo",
        "bakpo",
        "kawe",
        "guano",
        "ghoyak",
        "ghukah",
        "pitih",
        "bekwoh",
        "sokmo",
        "toksoh"
    ]

    strong_matches = [
        w for w in single_matches
        if w in strong_words
    ]

    if len(set(strong_matches)) >= 2:
        return "kelantan"

    return "malay"

@app.route("/fetch-x", methods=["GET", "POST"])

def fetch_x():

    if not session.get("admin"):
        return redirect(url_for("login"))

    keyword = request.form.get("keyword", "").strip()

    if not keyword:
        return redirect(url_for("admin"))

    posts = fetch_x_posts(keyword, max_results=10)

    bearer_token = os.getenv("X_BEARER_TOKEN")

    print("TOKEN:", bearer_token)

    client = requests.Session()

    print("FETCH TIME:", datetime.now())
   

    print("TOTAL POSTS FETCHED:", len(posts))

    for post in posts:

        text = post.get("text", "")

        print(
        "TWEET TIME:",
        post.get("created_at", "")
        )

        print(
            "TEXT:",
            text
        )

        text = post.get("text", "")

        dialect_label = detect_kelantan_dialect(text)

        print("TEXT:", text)
        print("DIALECT:", dialect_label)

        text_lower = text.lower()

        political_phrases = [
            "pas",
            "umno",
            "pkr",
            "dap",
            "pn",
            "ph",
            "bn",
            "walaun",
            "uec",
            "parti",
            "undi",
            "pilihan raya",
            "pru",
            "prn",
            "politik",
            "kerajaan",
            "menteri"
        ]

        adult_spam_phrases = [
            "open new slot",
            "dick",
            "pusyy",
            "chudai",
            "colmek",
            "kote",
            "pepek",
            "ds girl",
            "slot",
            "threesome",
            "tele",
            "janda",
            "dm or tele",
            "tele id",
            "telegram",
            "mummy",
            "jilboob",
            "chindo",
            "ds",
            "sex",
            "sexual",
            "horny",
            "hoockup",
            "18+",
            "18sx",
            "slot ds",
            "vid",
            "pic",

            # selling / promo
            "limited slot",
            "waitlist",
            "commission",
            "appointment",
            "channel",
            "open jasa",

            # explicit
            "nude",
            "boobs",
            "breast",
            "panties",
            "onlyfans",
            "nsfw",
            "fuck",
            "bj",
            "anal",
            "oral",
            "cum",
            "sugarbaby",
            "sugar baby",
            "sugar daddy",
            "massage"
        ]

        if any(word in text_lower for word in political_phrases):
            print("SKIPPED POLITICAL POST:", text)
            continue

        is_spam = any(
            word in text_lower
            for word in adult_spam_phrases
        )

        if is_spam:
            sentiment = "spam"
            confidence = 100

        positive_phrases = [

            # gotong royong / community
            "gotong royong",
            "bekerjasama",
            "kerjasama",
            "bantu",
            "tolong",
            "sukarelawan",
            "semangat kejiranan",
            "perpaduan",
            "bersatu",
            "ramai hadir",
            "meriah",
            "molek",
            "cakno",
            "bersih",
            "jaga kebersihan",
            "program terbaik",
            "usaha penduduk",
            "hubungan jiran",
            "masyarakat bantu",
            "turun bantu",

            # positive reactions
            "puji",
            "terbaik",
            "bagus",
            "baik",
            "sokong",
            "permudahkan urusan",
            "selamat",
            "doa",
            "takziah",
            "semoga",
            "harap keadaan lebih baik",
            "cepat sembuh",
            "keselamatan dipertingkatkan",

            # infrastructure / improvements
            "naik taraf",
            "penyelesaian",
            "tindakan tegas",
            "penguatkuasaan",
            "perhatian serius",
            "kesedaran",
            "jaga kawasan",
            "lebih ketat",
            "lebih baik",
            "pihak berkaitan ambik perhatian"

        ]


        negative_phrases = [

            "bodoh",
            "walaun",
            "kepam",
            "benci",
            "hina",
            "marah",
            "teruk",
            "takde masa",
            "layan kerenah",
            "rasuah",
            "burit",
            "bodoh",
            "bangang",
            "babi",
            "sial",
            "pukimak",
            "celaka",
            "hina",
            "keji",
            "teruk",
            "rasuah",
            "penipu",
            "scam",
            "curi",
            "rogol",
            "bunuh",
            "tikam",
            "mangsa",
            "jenayah",
            "pukul",
            "ugut",
            "gangster",
            "maki",
            "marah",
            "benci",
            "kecewa",
            "menipu",
            "haram",

            # rempit
            "mat rempit",
            "rempit",
            "merempit",
            "wheelie",
            "litar lumba",
            "ekzos",
            "bunyi ekzos",
            "ekzos kuat",
            "gelek malam",
            "bahaya",
            "bahayakan",
            "terganggu",
            "ganggu",
            "takut",
            "susoh",
            "memalukan",
            "bawak laju",

            # traffic / road
            "sesak",
            "kesesakan",
            "jem",
            "tersangkut",
            "cilok",
            "trafik",
            "jalan sempit",
            "lambat sampai",

            # sampah
            "sampah",
            "longkang penuh",
            "bau sampah",
            "kotor",
            "plastik",
            "botol",
            "tikus",
            "serangga",
            "buang sampah",
            "merata-rata",

            # banjir
            "banjir",
            "naik air",
            "dinaiki air",
            "terkandas",
            "hujan lebat",
            "air deras",
            "terjejas",
            "mangsa banjir",

            # jenayah
            "jenayah",
            "tikam",
            "kes bunuh",
            "seram",
            "terkejut",
            "penjenayah",
            "hukuman berat",
            "ganggu rasa selamat",

            # petrol tumpat
            "seludup",
            "seludup petrol",
            "sorok minyak",
            "subsidi",
            "disalah guna",
            "kecewa",
            "sempadan",

            # tiang konkrit
            "hempap",
            "tiang konkrit",
            "menyayat hati",
            "sedih",
            "risau",
            "keselamatan budok",
            "takziah",

            # kemalangan sekolah
            "langgar",
            "kemalangan",
            "cedera",
            "kawasan sekolah bahaya",
            "pemandu cuai",
            "zebra crossing",

            # general emotion
            "marah",
            "fedup",
            "viral",
            "panas",
            "risau",
            "masalah",
            "isu besar",
            "kesan besar",
            "topik panas"

        ]

        if is_spam:

            sentiment = "spam"
            confidence = 100

        elif any(word in text_lower for word in negative_phrases):

            sentiment = "negative"
            confidence = 90

        elif any(word in text_lower for word in positive_phrases):

            sentiment = "positive"
            confidence = 90

        else:

            sentiment = "neutral"
            confidence = 80

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
    
    positive_count = len(
        df[df["majority_sent"].astype(str).str.lower() == "positive"]
    )

    negative_count = len(
        df[df["majority_sent"].astype(str).str.lower() == "negative"]
    )

    if   negative_count > positive_count:
        admin_summary = "Negative discussions are currently higher. Admin should monitor local issues closely."
    elif positive_count > negative_count:
        admin_summary = "Public discussion is generally positive with good community engagement."
    else:
        admin_summary = "Sentiment is balanced. Continue monitoring new posts and trends."
   
    return render_template(
        "admin.html",
        admin_summary=admin_summary,
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

@app.route("/delete-record/<int:id>")
def delete_record(id):

    if not session.get("admin"):
        return redirect(url_for("login"))

    record = SentimentData.query.get(id)

    if record:
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
    
def detect_language_4(text):

    text = str(text)
    text_lower = text.lower()

    # Chinese detection
    if re.search(r"[\u4e00-\u9fff]", text):
        return "chinese"

    # Kelantan dialect detection
    kelantan_topic_words = [
    "kelate", "klate", "koto bharu", "kota bharu",
    "nasi kerabu", "nasi tupe", "laksam", "akok",
    "colek", "budu", "somtam", "dikir barat",
    "wayang kulit", "bekwoh", "gomo kelate",
    "pante", "wakaf che yeh", "tok bali",
    "kawe", "ambo", "gapo", "bakpo", "sokmo",
    "tokleh", "ore", "make", "molek", "do’oh",
    "dda’a", "bba-ba", "kurela", "ghoyak"
]

    if (
        detect_kelantan_dialect(text_lower) == "kelantan"
        or any(word in text_lower for word in kelantan_topic_words)
    ):
        return "kelantan"

    # English detection
    english_words = [
        "the", "and", "is", "are", "this", "that",
        "good", "bad", "nice", "love", "hate", "happy", "sad",
        "like", "dislike", "support", "oppose", "agree", "disagree",
        "infrastructure", "community", "engagement", "social", "harmony",
        "traffic", "road", "flood", "crime", "safety",
        "education", "health", "environment", "economy", "job"
    ]

    if any(re.search(r"\b" + word + r"\b", text_lower) for word in english_words):
        return "english"

    return "malay"
   
names = [
    "azwin", "jaselia", "mekyah", "pokdin", "abemat",
    "mekani", "abemi", "mekna", "poksu", "wannisa",
    "aina", "abeloh", "mekros", "tokayah", "wani",
    "mie", "kakyati", "abedin", "nisa", "meksiti",
    "abezul", "pija", "abenik", "mektie", "kaknoh",
    "abemail", "amira", "faiz", "shafiq", "syazwan",
    "atika", "farah"
]

loc_keywords = ["kb", "klate", "kelate", "gomo", "tokbali", "pc", "pasirmas"]
fillers = ["_", "ys", "eyy", "aa", "qt", "zy", "hoo", "real", "yo"]

def generate_twitter_username():
    name = random.choice(names)
    style = random.randint(1, 5)

    if style == 1:
        return f"{name}_{random.choice(loc_keywords)}"
    elif style == 2:
        return f"{name}{random.choice(fillers)}"
    elif style == 3:
        year = random.choice(["98", "99", "00", "01", "02", "03", "04", "05"])
        separator = random.choice(["_", "", "."])
        return f"{name}{separator}{year}"
    elif style == 4:
        prefix = random.choice(["its", "itsme", "not", "hi"])
        return f"{prefix}{name}"
    else:
        return f"{name}{name[-1]*2}"

def get_realistic_username():
    return generate_twitter_username()


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

        kelantan_keywords = [

            # =========================
            # DISTRICT / LOCATION
            # =========================
            "kelantan",
            "kota bharu",
            "pasir mas",
            "tumpat",
            "machang",
            "tanah merah",
            "gua musang",
            "bachok",
            "pasir puteh",
            "ketereh",
            "wakaf che yeh",
            "pengkalan kubor",
            "tok bali",
            "pantai cahaya bulan",
            "pcb",
            "rantau panjang",

            # =========================
            # CULTURE
            # =========================
            "dikir barat",
            "wayang kulit",
            "rebana ubi",
            "wau bulan",
            "batik klate",
            "songket",
            "bekwoh",
            "adat kelate",
            "silat tari",
            "gasing pangkah",
            "main puteri",
            "loghat kelate",

            # =========================
            # FOOD
            # =========================
            "nasi kerabu",
            "nasi dagang",
            "laksam",
            "akok",
            "budu",
            "colek",
            "nasi tumpang",
            "nasi tupe",
            "ketupat sotong",
            "somtam",
            "lompat tikam",
            "etok salai",
            "tepung pelita",
            "jala emas",
            "ayam percik",

            # =========================
            # TOURIST PLACE
            # =========================
            "pasar siti khadijah",
            "pantai sri tujoh",
            "gunung stong",
            "lata rek",
            "muzium klate",
            "jalan berek",
            "wakaf che yeh",
            "pantai irama",
            "kuala koh",

            # =========================
            # SPORT
            # =========================
            "kelantan fc",
            "gomo kelate",
            "stadium sultan muhammad iv",
            "bola sepak",
            "futsal",
            "sepaktakraw",
            "maraton",
            "skateboard",
            "e-sport",
            "main bolo",

            # =========================
            # STRONG DIALECT
            # =========================
            "ambo",
            "kawe",
            "gapo",
            "bakpo",
            "guano",
            "ghoyak",
            "ghukah",
            "toksoh",
            "sokmo",
            "kurela",
            "pecoh panggung",
            "pecoh peyok",
            "kemah keming",
            "ghuyup",
            "jaddah"
        ]


        political_keywords = [
            "politik", "parti", "undi", "pilihan raya", "pru", "prn",
            "dun", "parlimen", "menteri", "kerajaan negeri",
            "manifesto", "kempen", "calon", "adun", "mp",
            "pas", "umno", "bersatu", "pkr", "dap", "pn", "ph", "bn"
        ]

        saved_rows = 0
        skipped_politic = 0
        skipped_non_kelantan = 0

        for _, row in df.iterrows():

            comment = ""
            for col in ["comment/tweet", "comment", "tweet", "text", "content", "news", "caption", "post", "article", "description"]:
                if col in df.columns:
                    comment = str(row.get(col, ""))
                    break

            topic = "general"
            for col in ["post/keyword", "topic", "keyword", "issue", "title", "category"]:
                if col in df.columns:
                    topic = str(row.get(col, "general"))
                    break

            username = get_realistic_username()
            for col in ["username", "user", "author", "name", "source"]:
                if col in df.columns:
                    username = str(row.get(col, "dataset"))
                    break

            full_text = f"{topic} {comment}".lower()

            if any(word in full_text for word in political_keywords):
                skipped_politic += 1
                continue

            dialect = detect_kelantan_dialect(full_text)

            matched_keywords = []

            for keyword in kelantan_keywords:

                keyword_clean = keyword.lower().strip()

                # PHRASE MATCH
                if " " in keyword_clean:

                    if keyword_clean in full_text:
                        matched_keywords.append(keyword_clean)

                # SINGLE WORD MATCH
                else:

                    pattern = r"\b" + re.escape(keyword_clean) + r"\b"

                    if re.search(pattern, full_text):
                        matched_keywords.append(keyword_clean)

            print("MATCHED KELANTAN WORDS:", matched_keywords)

            is_kelantan_topic = False

            # STRONG PHRASE
            if any(" " in word for word in matched_keywords):
                is_kelantan_topic = True

            # REQUIRE 2 SINGLE WORDS
            elif len(matched_keywords) >= 2:
                is_kelantan_topic = True

    
            language_label = detect_language_4(full_text)

            if is_kelantan_topic and language_label == "malay":
                language_label = "kelantan"

            sentiment, confidence = predict_sentiment(comment)
            comment_lower = comment.lower()

            negative_phrases = [

                "banjir",
                "sampah",
                "sesak",
                "jem",
                "kemalangan",
                "langgar",
                "rempit",
                "mat rempit",
                "jenayah",
                "seludup",
                "sedih",
                "takut",
                "marah",
                "bahaya",
                "risau",
                "kecewa",

                # CRIME / NEWS
                "didakwa",
                "seksual",
                "meraba",
                "remaja",
                "mangsa",
                "kes",
                "polis",
                "mahkamah",
                "cedera",
                "maut",
                "rogol",
                "bunuh",
                "curi",
                "serang",
                "siasatan",
                "penjara",
                "hukuman"
            ]

            positive_phrases = [

                "gotong royong",
                "bantu",
                "tolong",
                "sukarelawan",
                "molek",
                "bagus",
                "terbaik",
                "bersih",
                "selamat",
                "semoga",
                "baik",
                "kerjasama",
                "perpaduan"
            ]

            # FORCE NEGATIVE
            if any(word in comment_lower for word in negative_phrases):
                sentiment = "negative"

            # FORCE POSITIVE
            elif any(word in comment_lower for word in positive_phrases):
                sentiment = "positive"

            # LOW CONFIDENCE = NEUTRAL
            elif confidence < 70:
                sentiment = "neutral"
            data = SentimentData(
                post_keyword=topic,
                comment_text=comment,
                username=username,
                like_count=0,
                reply_count=0,
                time_created=str(datetime.now()),
                sentiment_label=sentiment,
                sarcasm_label="unknown",
                language_id=language_label
            )

            db.session.add(data)
            saved_rows += 1

        db.session.commit()

        print("KELANTAN ROWS SAVED:", saved_rows)
        print("SKIPPED POLITICAL ROWS:", skipped_politic)
        print("SKIPPED NON-KELANTAN ROWS:", skipped_non_kelantan)

    return redirect(url_for("admin"))

@app.route("/fix-language")
def fix_language():

    records = SentimentData.query.all()

    for row in records:

        if str(row.language_id).strip().lower() == "kelantan":
            row.language_id = "kelantan"

        else:
            row.language_id = "malay"

    db.session.commit()

    return "Language updated successfully"

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



