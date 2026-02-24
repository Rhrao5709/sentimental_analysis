import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ======================
# LOAD DATA
# ======================
DATA_PATH = "data/IMDB Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ======================
# PREPARE DATA
# ======================
TEXT_COL = "review"
LABEL_COL = "sentiment"

X = df[TEXT_COL].astype(str)
y = df[LABEL_COL].astype(str)

# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# PIPELINE
# ======================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("clf", LinearSVC())
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

print("\nEvaluating model...")
preds = pipeline.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}\n")
print(classification_report(y_test, preds))

# ======================
# SAVE
# ======================
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl")

print("\nModel saved to model/model.pkl")