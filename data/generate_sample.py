"""
Generates a small sample dataset for testing without downloading IMDB.
Real dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

import pandas as pd

positive = [
    "This movie was absolutely fantastic, I loved every minute of it",
    "One of the best films I have seen in years, great acting and story",
    "Brilliant performances from the entire cast, highly recommend",
    "A masterpiece of storytelling, kept me engaged from start to finish",
    "The cinematography was stunning and the plot was gripping",
    "Exceeded all my expectations, will definitely watch again",
    "Wonderful film with great characters and emotional depth",
    "A truly moving experience, the director did an amazing job",
    "Best movie of the year without a doubt, outstanding in every way",
    "Heartwarming and funny, the whole family enjoyed it",
]

negative = [
    "Terrible movie, complete waste of time and money",
    "Boring plot with no character development whatsoever",
    "I fell asleep halfway through, nothing interesting happens",
    "Awful acting and a predictable story that goes nowhere",
    "One of the worst films I have ever seen, deeply disappointing",
    "The script was painful to watch, no redeeming qualities at all",
    "Completely incoherent plot, made no sense from beginning to end",
    "Poor direction and terrible dialogue, avoid at all costs",
    "Total disaster of a film, the trailer was the best part",
    "Unbearably slow and dull, felt like it would never end",
]

rows = [{"text": t, "label": 1} for t in positive] + [{"text": t, "label": 0} for t in negative]
df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/imdb_reviews.csv", index=False)
print(f"Saved {len(df)} sample reviews to data/imdb_reviews.csv")
print("For real results download the full 50k dataset from Kaggle:")
print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
