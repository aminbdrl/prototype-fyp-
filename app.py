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

    for i in range(20):

        district = districts[i % len(districts)]

        issue_comments = {

            "banjir": [

                f"Air sungai dekat {district} naik cepat sungguh sejak malam tadi.",
                f"Ramai penduduk {district} dah mula pindah barang ke tempat tinggi.",
                f"Hujan tak berhenti-henti dari semalam, kawe risau banjir makin teruk.",
                f"Jalan utama dekat {district} banyak dah mula dinaiki air.",
                f"Oghe kampung di {district} ramai duk update keadaan banjir dalam Facebook.",
                f"Kawe tengok air dekat rumah naik sikit demi sikit malam ni.",
                f"Semoga semua mangsa banjir di {district} dipermudahkan urusan.",
                f"Demo semua hati-hati kalau lalu kawasan rendah dekat {district}.",
                f"Banjir kali ni nampok macam lebih teruk dari tahun lepas.",
                f"Ramai sukarelawan turun bantu mangsa banjir dekat {district}.",
                f"Bekalan makanan dekat pusat pemindahan sementara mula diagihkan.",
                f"Kawe harap cuaca cepat baik sebab ramai dah terjejas.",
                f"Air deras dekat kawasan sungai memang bahaya untuk budok kecik.",
                f"Oghe ramai mula risau kalau hujan berterusan sampai esok.",
                f"Kawe tengok banyak kereta terkandas sebab jalan dinaiki air.",
                f"Penduduk {district} ramai update video banjir dalam TikTok sekarang.",
                f"Banyak rumah dekat kawasan rendah dah mula dimasuki air.",
                f"Kawe doa semoga semua keluarga di {district} selamat.",
                f"Banjir ni memang ujian berat untuk masyarakat kampung.",
                f"Ramai netizen duk share nombor bantuan untuk mangsa banjir."
            ],


            "sampah": [

                f"Sampah dekat tepi jalan {district} makin banyak sekarang.",
                f"Demo buang sampah merata-rata memang susahkan masyarakat.",
                f"Kawe tengok longkang dekat {district} penuh dengan sampah.",
                f"Bau sampah dekat kawasan pasar memang kuat sungguh.",
                f"Oghe ramai mengadu pasal masalah kebersihan di {district}.",
                f"Kalau semua jaga kebersihan, kawasan kampung jadi lebih molek.",
                f"Kawe rasa kesedaran pasal kebersihan masih rendah lagi.",
                f"Sampah bertimbun ni boleh tarik tikus dan serangga.",
                f"Demo semua kena cakno kebersihan kawasan masing-masing.",
                f"Ramai netizen marah tengok sampah dibuang dalam sungai.",
                f"Tok soh harap pekerja majlis je, masyarakat pun kena bantu.",
                f"Kawe tengok banyak plastik dan botol dibuang tepi jalan.",
                f"Isu sampah dekat {district} ni dah lama berlaku.",
                f"Oghe kampung harap tindakan lebih tegas untuk orang buang sampah.",
                f"Kalau hujan lebat, sampah ni boleh sebabkan banjir pulok.",
                f"Kawe rasa gotong royong kena dibuat lebih kerap.",
                f"Ramai pengguna media sosial kongsi gambar kawasan kotor sekarang.",
                f"Sampah dekat pasar malam memang banyak lepas habis berniaga.",
                f"Demo semua jangan malas buang sampah dalam tong.",
                f"Kawe tengok ramai budok muda mula cakno pasal kebersihan."
            ],


            "kesesakan jalan": [

                f"Jalan dekat {district} sesak teruk petang ni.",
                f"Kawe ambik masa hampir sejam untuk lalu kawasan bandar tadi.",
                f"Oghe ramai mengadu trafik makin teruk sejak akhir-akhir ni.",
                f"Demo keluar awal sikit kalau nak elak jem dekat {district}.",
                f"Kesesakan dekat lampu isyarat memang panjang waktu balik kerja.",
                f"Kawe tengok banyak kereta tersangkut dekat jalan utama.",
                f"Jalan sempit dan jumlah kereta makin banyak sekarang.",
                f"Ramai pengguna jalan raya dah mula fedup dengan jem harian.",
                f"Kalau cuti sekolah memang lagi sesak kawasan bandar.",
                f"Kawe rasa jalan dekat {district} perlu dinaik taraf segera.",
                f"Oghe ramai share keadaan trafik dalam media sosial hari ni.",
                f"Kesesakan ni buat ramai lambat sampai tempat kerja.",
                f"Demo semua kena lebih sabar waktu memandu.",
                f"Kawe tengok banyak motosikal cilok waktu jem.",
                f"Jalan dekat pasar malam memang sesak habih malam ni.",
                f"Ramai netizen kata trafik sekarang makin mencabar.",
                f"Kawe harap pihak berkaitan cari penyelesaian cepat.",
                f"Oghe kampung pun mula rasa jalan makin sibuk sekarang.",
                f"Jem dekat kawasan sekolah memang teruk waktu pagi.",
                f"Kalau hujan sikit terus trafik jadi perlahan."
            ],


            "jenayah ketereh": [

                f"Kes dekat Ketereh ni memang buat ramai oghe terkejut pagi tadi.",
                f"Kawe baca berita pun rasa seram dengan apa yang berlaku.",
                f"Ramai netizen minta polis percepatkan siasatan kes ni.",
                f"Kes macam ni memang ganggu rasa selamat masyarakat sekarang.",
                f"Oghe Kelantan ramai duk bincang kes ni dalam Facebook malam ni.",
                f"Demo semua jangan cepat percaya cerita tak sahih pasal kes ni.",
                f"Kawe tengok ramai orang share rasa simpati dekat keluarga mangsa.",
                f"Berita kes ni memang penuh dalam timeline sejak semalam lagi.",
                f"Ramai marah kalau tengok jenayah berat makin menjadi sekarang.",
                f"Kes ni memang jadi topik panas dalam media sosial Kelantan.",
                f"Kawe harap pihak polis dapat cari bukti dengan cepat.",
                f"Oghe kampung dekat Ketereh pun ramai terkejut dengan kejadian ni.",
                f"Ramai pengguna TikTok duk buat awareness pasal keselamatan sekarang.",
                f"Kes macam ni buat ibu bapa makin risau nak bagi anak keluar malam.",
                f"Kawe tengok ramai netizen minta hukuman lebih tegas untuk penjenayah.",
                f"Memang sedih tengok berita jenayah macam ni berlaku dekat negeri sendiri.",
                f"Oghe ramai duk bincang pasal keselamatan kawasan kampung sekarang.",
                f"Kalau buka komen Facebook, memang penuh orang bercakap pasal kes ni.",
                f"Ramai harap kes ni dapat diselesaikan secepat mungkin.",
                f"Kes jenayah ni memang tinggalkan kesan besar pada masyarakat."
            ],


            "petrol tumpat": [

                f"Kes seludup petrol dekat Tumpat ni memang viral sungguh sekarang.",
                f"Kawe tak sangka kereta kecil pun boleh ubah suai untuk sorok minyak.",
                f"Ramai netizen puji tindakan pihak berkuasa tahan suspek dekat sempadan.",
                f"Kes macam ni memang rugikan rakyat sebab subsidi disalah guna.",
                f"Oghe ramai duk bincang pasal harga minyak sejak berita ni keluar.",
                f"Demo semua jangan ambik kesempatan atas subsidi kerajaan.",
                f"Kawe tengok ramai marah bila baca berita pasal kes ni.",
                f"Kalau tengok media sosial, ramai setuju tindakan tegas patut dibuat.",
                f"Kes seludup minyak dekat Tumpat memang jadi perhatian sekarang.",
                f"Ramai pengguna Facebook share video dan berita pasal kes ni.",
                f"Kawe rasa kawalan sempadan kena dipertingkatkan lagi lepas ni.",
                f"Oghe ramai pelik macam mano boleh sorok minyak banyak dalam kereta.",
                f"Kes macam ni memang buat rakyat rasa kecewa sungguh.",
                f"Ramai netizen kata subsidi patut sampai pada rakyat yang betul-betul perlu.",
                f"Kawe tengok ramai minta hukuman lebih berat untuk penyeludup.",
                f"Demo semua jangan pentingkan duit sampai buat kerja salah macam ni.",
                f"Berita ni memang cepat viral dalam grup Kelantan malam tadi.",
                f"Oghe kampung pun ramai sembang pasal kes ni sekarang.",
                f"Kawe rasa penguatkuasaan dekat kawasan sempadan kena lebih ketat.",
                f"Kes petrol dekat Tumpat ni memang jadi topik hangat minggu ni."
            ],


            "tiang konkrit": [

                f"Sedih sungguh dengar budok kena hempap tiang konkrit dekat Kota Bharu.",
                f"Ramai netizen ucap takziah pada keluarga mangsa malam ni.",
                f"Kawe rasa kawasan permainan budok kena dipantau lebih ketat.",
                f"Kes macam ni memang buat ramai ibu bapa takut dan risau.",
                f"Oghe ramai marah sebab struktur berat dibiarkan dekat tempat budok bermain.",
                f"Kawe tengok ramai share berita sedih ni dalam TikTok sekarang.",
                f"Kejadian macam ni memang sangat menyayat hati untuk masyarakat.",
                f"Demo semua kena lebih cakno pasal keselamatan kawasan awam.",
                f"Ramai pengguna media sosial minta siasatan dibuat segera.",
                f"Kawe harap pihak berkaitan periksa semua kawasan permainan lepas ni.",
                f"Kes ni memang buat ramai orang tersentuh hati sungguh.",
                f"Oghe ramai kata keselamatan budok kecil jangan dibuat main.",
                f"Kawe tengok ramai ibu bapa mula risau dengan kawasan permainan terbuka.",
                f"Ramai netizen minta tindakan segera supaya benda macam ni tak ulang lagi.",
                f"Berita ni memang cepat viral sebab ramai rasa simpati dekat keluarga mangsa.",
                f"Kawe rasa semua tempat awam kena diperiksa balik demi keselamatan.",
                f"Oghe ramai share doa dan ucapan takziah dalam media sosial malam ni.",
                f"Kejadian macam ni memang beri kesan besar pada masyarakat setempat.",
                f"Ramai pengguna Facebook kata keselamatan kawasan awam perlu dipertingkatkan.",
                f"Kawe harap keluarga mangsa diberi kekuatan menghadapi ujian ni."
            ],


            "kemalangan sekolah": [

                f"Kes pelajar kena langgar dekat sekolah ni memang mengejutkan ramai.",
                f"Kawe tengok ramai netizen marah dengan pemandu yang bawak laju.",
                f"Kawasan sekolah memang kena had laju lebih ketat lepas ni.",
                f"Ramai ibu bapa risau keselamatan anak-anak waktu pergi sekolah.",
                f"Demo semua bawak kereta biar perlahan dekat kawasan sekolah.",
                f"Video kemalangan tu memang viral dalam TikTok sekarang.",
                f"Kawe harap pelajar yang cedera cepat sembuh.",
                f"Oghe ramai minta bonggol jalan ditambah dekat kawasan sekolah.",
                f"Kes macam ni memang buat masyarakat sedih dan marah.",
                f"Keselamatan pelajar kena jadi keutamaan semua pihak.",
                f"Kawe tengok ramai pengguna media sosial kongsi rasa simpati dekat keluarga mangsa.",
                f"Ramai netizen kata kawasan sekolah sekarang makin bahaya waktu pagi.",
                f"Kalau tengok komen Facebook, ramai minta tindakan lebih tegas dekat pemandu cuai.",
                f"Oghe ramai harap pihak sekolah dan JPJ ambik perhatian serius pasal isu ni.",
                f"Kawe rasa zebra crossing dekat sekolah kena diperjelaskan lagi.",
                f"Ramai pengguna jalan raya masih bawak laju walaupun dekat kawasan sekolah.",
                f"Kes ni memang buat ramai ibu bapa takut nak lepaskan anak jalan sendiri.",
                f"Kawe tengok ramai budak sekolah melintas jalan tanpa pengawasan sekarang.",
                f"Ramai pengguna TikTok share video pasal keselamatan pelajar sejak kes ni viral.",
                f"Oghe ramai harap kemalangan macam ni tak berlaku lagi lepas ni."
            ],

            "gotong royong": [

                f"Program gotong royong di {district} ni memang terbaik, ramai oghe turun bantu.",
                f"Kawe suka tengok masyarakat {district} bekerjasama bersihkan kawasan kampung.",
                f"Gotong royong macam ni boleh rapatkan hubungan sesama jiran.",
                f"Demo semua bagus, kerja bersih kampung jadi cepat siap.",
                f"Ramai anak muda turut serta dalam gotong royong pagi tadi.",
                f"Kawasan taman dekat {district} nampok lebih bersih lepas program tadi.",
                f"Kawe rasa aktiviti macam ni patut dibuat lebih kerap.",
                f"Oghe kampung sama-sama bantu angkat sampah dan bersihkan longkang.",
                f"Suasana gotong royong tadi memang meriah dengan ramai penduduk hadir.",
                f"Kawe tengok ramai sukarelawan datang walaupun cuaca panas.",
                f"Program macam ni memang bagus untuk pupuk semangat kejiranan.",
                f"Ramai netizen puji usaha penduduk {district} jaga kebersihan kawasan.",
                f"Demo semua pakat bersih kawasan memang molek sungguh tengok.",
                f"Kawe harap lebih banyak komuniti buat aktiviti gotong royong macam ni.",
                f"Budok muda pun nampok semangat bantu masyarakat pagi ni.",
                f"Gotong royong dekat {district} ni memang tunjuk semangat perpaduan masyarakat.",
                f"Oghe ramai datang awal pagi semata-mata nak bantu bersihkan kawasan.",
                f"Kawe tengok hubungan jiran jadi lebih rapat lepas aktiviti ni.",
                f"Ramai penduduk share gambar gotong royong dalam Facebook hari ni.",
                f"Kalau semua kawasan buat gotong royong macam ni memang bersih sokmo."
            ],

            "rempit": [

                f"Mat rempit dekat airport Kelantan tu memalukan imej negeri.",
                f"Demo buat gapo merempit depan airport, ramai penumpang terganggu.",
                f"Bunyi ekzos malam-malam di Pengkalan Chepa tu gege sungguh.",
                f"Kawe sokong tindakan sita motor kalau masih buat aksi bahaya.",
                f"Setiap malam ado je mat rempit berkumpul dekat airport.",
                f"Oghe nak hantar keluarga ke airport pun jadi takut doh.",
                f"Ramai pelancong luar tengok perangai mat rempit ni, malu weh.",
                f"Demo ingat jalan airport tu litar lumba ka?",
                f"Kawe tengok makin ramai budok muda join geng rempit sekarang.",
                f"JPJ dan polis kena ronda lebih kerap kawasan airport waktu malam.",
                f"Video mat rempit dekat airport Kelantan tu viral habih dalam TikTok.",
                f"Bunyi ekzos kuat tengah malam memang ganggu penduduk sekitar.",
                f"Oghe nak tidur pun susoh bila geng motor dok gelek malam-malam.",
                f"Kawe rasa tindakan sita motor memang patut dibuat.",
                f"Ramai netizen puji tindakan polis ambik tindakan dekat kawasan airport.",
                f"Demo semua jangan jadi hero atas jalan raya sampai bahayakan oghe lain.",
                f"Rempit depan airport ni bukan budaya yang baik untuk anak muda.",
                f"Kawe tengok ramai pengguna jalan raya dah mula marah dengan geng rempit ni.",
                f"Ado yang buat wheelie depan kereta orang, memang bahaya sungguh.",
                f"Oghe luar datang Kelantan, benda ni pulok yang nampok dulu."
            ],


            "umum": [

                f"Kawe tengok isu pasal {keyword} ni makin ramai dok bincang di {district}.",
                f"Oghe {district} pun ramai share pendapat pasal isu {keyword} sekarang.",
                f"Demo rasa macam mano isu {keyword} ni berlaku di {district}?",
                f"Harap pihak berkaitan dapat tengok balik isu {keyword} di {district}.",
                f"Isu {keyword} ni jadi topik panas di kawasan {district} sejak akhir-akhir ni.",
                f"Ramai netizen Kelantan duk bincang pasal {keyword} terutama di {district}.",
                f"Kawe tengok ramai tak puas hati pasal isu {keyword} di {district}.",
                f"Kalau isu {keyword} ni tak selesai cepat, oghe {district} makin risau.",
                f"Ada yang sokong, ada jugok yang kritik isu {keyword} di {district}.",
                f"Tok soh ambik mudah isu {keyword} ni, ramai penduduk {district} terkesan.",
                f"Perbincangan pasal {keyword} di {district} makin aktif dalam media sosial.",
                f"Kawe harap keadaan pasal {keyword} di {district} boleh jadi lebih baik lepas ni.",
                f"Ramai anak muda di {district} duk share pandangan pasal isu {keyword}.",
                f"Isu {keyword} ni nampok kecik, tapi ramai oghe di {district} ambik serius.",
                f"Demo tengok sendiri lah, isu {keyword} ni memang jadi perhatian di {district}.",
                f"Timeline Facebook penuh doh dengan cerita pasal {keyword} di {district}.",
                f"Kawe perati ramai oghe mula bincang pasal {keyword} sejak semalam lagi.",
                f"Kalau tengok komen netizen, ramai oghe {district} ada pandangan berbeza pasal {keyword}.",
                f"Oghe kampung pun duk sembang pasal isu {keyword} ni sekarang.",
                f"Rata-rata masyarakat di {district} harap isu {keyword} ni cepat selesai."
            ],


}

        keyword_lower = keyword.lower()

        selected_comments = issue_comments.get("umum")

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

@app.route("/fetch-x", methods=["POST"])

def fetch_x():

    if not session.get("admin"):
        return redirect(url_for("login"))

    keyword = request.form.get("keyword")

    posts = fetch_x_posts(keyword, max_results=10)

    print("TOTAL POSTS FETCHED:", len(posts))

    for post in posts:

        text = post.get("text", "")

        dialect_label = detect_kelantan_dialect(text)

        print("TEXT:", text)
        print("DIALECT:", dialect_label)

        text_lower = text.lower()

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

        if any(word in text_lower for word in negative_phrases):
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

       # ONLY location/topic keywords here
        kelantan_keywords = [
            "kelantan",
            "kota bharu",
            "pasir mas",
            "tumpat",
            "machang",
            "tanah merah",
            "gua musang",
            "bachok",
            "pasir puteh",
            "pengkalan chepa",
            "ketereh",
            "kubang kerian",
            "wakaf bharu",
            "kok lanas",
            "rantau panjang"
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

            username = "dataset"
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

            if dialect != "kelantan" and not is_kelantan_topic:
                skipped_non_kelantan += 1
                continue

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
                language_id=dialect
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



