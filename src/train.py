import joblib
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import clean_series

MODEL_DIR = Path("model")


def train(data_path: str = "data/imdb_reviews.csv"):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reviews  |  positive: {(df['label']==1).sum()}  negative: {(df['label']==0).sum()}")

    df["text"] = clean_series(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF captures unigrams and bigrams — "not good" is very different from "good"
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.pkl")

    metrics = {"roc_auc": round(auc, 4)}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train()
