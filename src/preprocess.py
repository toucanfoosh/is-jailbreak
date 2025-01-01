import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load TF-IDF metadata
with open("./dev/models/tfidf_vocab.json", "r") as f:
    tfidf_meta = json.load(f)

vocabulary = tfidf_meta["vocabulary"]
idf = np.array(tfidf_meta["idf"])
ngram_range = tuple(tfidf_meta["ngram_range"])
max_features = tfidf_meta["max_features"]

tfidf = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features,
    stop_words="english",
    preprocessor=lambda x: x.lower(),
    vocabulary=vocabulary,
)
tfidf.idf_ = idf  # Set preloaded IDF values


def preprocess_input(text):
    vector = tfidf.transform([text]).toarray()
    return vector.tolist()


if __name__ == "__main__":
    import sys

    input_text = sys.argv[1]
    vector = preprocess_input(input_text)
    print(json.dumps(vector))
