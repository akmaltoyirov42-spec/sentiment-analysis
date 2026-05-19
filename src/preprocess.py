import re
import string


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+", " ", text)         # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_series(series):
    return series.apply(clean)
