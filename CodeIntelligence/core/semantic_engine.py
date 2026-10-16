import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .tokenizer import CodeTokenizer


class SemanticVectorEngine:
    """Sparse TF-IDF cosine search with domain concept expansion."""

    def __init__(self):
        self.tokenizer = CodeTokenizer()
        self.documents = []
        self.vectorizer = None
        self.matrix = None

    def fit(self, documents):
        self.documents = list(documents)
        if not self.documents:
            self.vectorizer = None
            self.matrix = None
            return self

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: self.tokenizer.tokenize(text, expand_synonyms=False),
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            dtype=np.float32,
        )
        try:
            self.matrix = self.vectorizer.fit_transform(self.documents)
        except ValueError as error:
            if "empty vocabulary" not in str(error).casefold():
                raise
            self.vectorizer = None
            self.matrix = None
        return self

    def search(self, query, top_k=50):
        if self.vectorizer is None or self.matrix is None or top_k <= 0:
            return []

        expanded_tokens = self.tokenizer.tokenize_query(query)
        if not expanded_tokens:
            return []
        expanded_query_str = " ".join(expanded_tokens)

        query_vector = self.vectorizer.transform([expanded_query_str])
        scores = (self.matrix @ query_vector.T).tocoo()
        results = sorted(
            ((int(index), float(score)) for index, score in zip(scores.row, scores.data) if score > 0),
            key=lambda result: result[1],
            reverse=True,
        )
        return results[:top_k]
