import joblib
from pathlib import Path

from src.preprocess import clean

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(Path("model") / "pipeline.pkl")


def predict(text: str) -> dict:
    _load()
    cleaned = clean(text)
    label = _pipeline.predict([cleaned])[0]
    prob = _pipeline.predict_proba([cleaned])[0]
    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": round(float(prob[label]), 4),
        "positive_prob": round(float(prob[1]), 4),
        "negative_prob": round(float(prob[0]), 4),
    }
