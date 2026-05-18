import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Model used:", type(model))
print("Vectorizer used:", type(vectorizer))