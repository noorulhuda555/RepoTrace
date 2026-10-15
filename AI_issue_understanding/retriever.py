"""Stage 1: bi-encoder retrieval of candidate labels (pretrained, zero-shot).

Default: BAAI/bge-large-en-v1.5. Any sentence-transformers model works
(e.g. intfloat/e5-large-v2, Alibaba-NLP/gte-large-en-v1.5, BAAI/bge-m3).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from .label_space import LabelSpace

log = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BiEncoderRetriever:
    def __init__(
        self,
        label_space: LabelSpace,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        self.label_space = label_space
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._label_matrix: Optional[np.ndarray] = None

    # -- prefixes differ between embedding families
    def _query_prefix(self) -> str:
        n = self.model_name.lower()
        if "bge" in n and "m3" not in n:
            return _BGE_QUERY_PREFIX
        if "e5" in n:
            return "query: "
        return ""

    def _doc_prefix(self) -> str:
        return "passage: " if "e5" in self.model_name.lower() else ""

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        log.info("Loading bi-encoder %s", self.model_name)
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
        docs = [self._doc_prefix() + l.document for l in self.label_space.labels]
        self._label_matrix = model.encode(
            docs, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False
        )
        self._model = model

    def similarities(self, texts: Sequence[str]) -> np.ndarray:
        """Cosine similarity matrix of shape (len(texts), n_labels)."""
        self._ensure_ready()
        prefix = self._query_prefix()
        emb = self._model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return emb @ self._label_matrix.T

    @staticmethod
    def to_confidence(cos_row: np.ndarray, center_z: float = 2.0, sharpness: float = 1.5) -> np.ndarray:
        """Map raw cosines to [0,1]: z-score across all labels, then a sigmoid.

        Raw cosines from embedding models live in a narrow, model-specific band, so we express each
        label's similarity relative to the issue's similarity to *all* labels. A label ~2 std-devs
        above the mean maps to 0.5.
        """
        z = (cos_row - cos_row.mean()) / (cos_row.std() + 1e-8)
        return 1.0 / (1.0 + np.exp(-sharpness * (z - center_z)))
