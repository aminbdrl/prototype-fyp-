import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/kelantan_extended.csv")

df = df.dropna(subset=["comment/tweet", "majority_sent"])

X = df["comment/tweet"].astype(str)
y = df["majority_sent"].astype(str).str.lower()

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")

X_vectorized = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("MODEL USED:", type(model))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1-score:", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))