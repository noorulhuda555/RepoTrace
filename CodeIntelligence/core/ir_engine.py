import math
from collections import Counter
from typing import List, Tuple, Dict, Any

from .tokenizer import CodeTokenizer


class BM25InformationRetrievalEngine:
    """Advanced in-memory BM25Okapi retrieval engine with query term weighting."""

    def __init__(self, k1=1.4, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = CodeTokenizer()
        self.documents = []
        self.term_frequencies: List[Counter] = []
        self.document_frequencies = Counter()
        self.average_length = 0.0

    def fit(self, documents):
        self.documents = list(documents)
        self.term_frequencies = []
        self.document_frequencies = Counter()
        total = 0
        for document in self.documents:
            terms = self.tokenizer.tokenize(document, expand_synonyms=False)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())
            total += len(terms)
        self.average_length = total / max(1, len(self.documents))
        return self

    def search(self, query, top_k=50):
        query_terms = self.tokenizer.tokenize_query(query)
        if not query_terms or not self.documents or top_k <= 0:
            return []

        q_tf = Counter(query_terms)
        count = len(self.documents)
        results = []

        for index, frequencies in enumerate(self.term_frequencies):
            length = sum(frequencies.values())
            score = 0.0
            for term, q_weight in q_tf.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequencies.get(term, 0)
                # BM25 IDF formula with smoothing
                idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                if idf <= 0:
                    continue

                denominator = tf + self.k1 * (1.0 - self.b + self.b * length / max(1.0, self.average_length))
                term_score = idf * (tf * (self.k1 + 1.0) / denominator) * math.log1p(q_weight)
                score += term_score

            if score > 0:
                results.append((index, score))

        results.sort(key=lambda r: r[1], reverse=True)
        return results[:top_k]

    def query(self, query_text, top_k=10):
        return [(self.documents[index], score, {}) for index, score in self.search(query_text, top_k)]
