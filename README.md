# sentiment analysis — movie reviews

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

classifies IMDB reviews as positive or negative using TF-IDF + logistic regression. trained on the 50k review dataset. small streamlit UI to test it.

dataset: [IMDB 50k Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## results

| metric | score |
|---|---|
| ROC-AUC | 0.982 |
| accuracy | 0.897 |
| F1 (positive) | 0.90 |
| F1 (negative) | 0.90 |

using bigrams (`ngram_range=(1,2)`) gave a noticeable bump. "not good" and "not bad" become separate features instead of getting flattened to "good" / "bad". makes negation work properly.

---

## run it

```bash
git clone https://github.com/akmaltoyirov42-spec/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt

# option A: real dataset from kaggle (link above)
# download CSV, rename to imdb_reviews.csv, drop in data/

# option B: generate a small sample to test the pipeline
python data/generate_sample.py

python -m src.train
streamlit run app.py
```

## notes

- `sublinear_tf=True` in TfidfVectorizer helps — repeated words don't dominate
- bigrams matter more than expected
- logistic regression with TF-IDF beats most fancy stuff for this kind of task
- HTML tags hidden in reviews break things if you don't strip them

---

## what's next

want to fine-tune DistilBERT on the same dataset and compare. classical ML hits 89% — curious how much a transformer adds. probably 2-3% but at much higher cost. good benchmark.

---

scikit-learn, pandas, Streamlit
