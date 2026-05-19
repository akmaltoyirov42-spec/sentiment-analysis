# Sentiment Analysis — Movie Reviews

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

Classifies movie reviews as positive or negative. TF-IDF + Logistic Regression trained on 50k IMDB reviews. Simple Streamlit UI to test it live.

Dataset: [IMDB 50k Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.982 |
| Accuracy | 0.897 |
| F1 (positive) | 0.90 |
| F1 (negative) | 0.90 |

Used bigrams (`ngram_range=(1,2)`) — "not good" and "not bad" are treated as separate features instead of just "good" and "bad". Makes a noticeable difference on negation-heavy reviews.

---

## Run it

```bash
git clone https://github.com/akmaltoyirov42-spec/sentiment-analysis.git
cd sentiment-analysis

pip install -r requirements.txt

# Option A: use the real Kaggle dataset (50k reviews)
# download CSV from kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# rename it to imdb_reviews.csv and place in data/

# Option B: generate a small sample to test the pipeline
python data/generate_sample.py

python -m src.train
streamlit run app.py
```

---

## Files

```
├── src/
│   ├── preprocess.py   text cleaning (HTML, URLs, punctuation)
│   ├── train.py        TF-IDF + LogReg pipeline
│   └── predict.py      inference
├── app.py              Streamlit UI
└── data/
    └── generate_sample.py
```

---

## Stack

scikit-learn, pandas, Streamlit
