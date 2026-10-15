"""IssueAnalyzer: understand what an issue is *about* using only local pretrained HF models.

Pipeline (no training, no external APIs):
  1. Bi-encoder (bge-large) embeds the issue and retrieves the top-N candidate skill labels.
  2. NLI cross-encoder (deberta-v3-large zero-shot) scores each candidate by entailment.
  3. Scores are fused, thresholded and returned together with the issue type.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from .label_space import LabelSpace
from .reranker import NLIReranker
from .retriever import BiEncoderRetriever
from .schemas import IssueInput, IssueUnderstanding, LabelPrediction

log = logging.getLogger(__name__)


class IssueAnalyzer:
    def __init__(
        self,
        label_space: LabelSpace,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        nli_model: Optional[str] = None,  # None -> large on GPU, base on CPU
        device: Optional[str] = None,
        use_reranker: bool = True,
        top_candidates: int = 20,
        nli_weight: float = 0.6,
        threshold: float = 0.5,
        max_labels: int = 8,
        classify_type: bool = True,
    ) -> None:
        import torch

        has_gpu = torch.cuda.is_available() and (device in (None, "cuda"))
        if nli_model is None:
            nli_model = ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0" if has_gpu
                         else "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
        print(f"[issue-understanding] device={'cuda' if has_gpu else 'cpu'} "
              f"nli_model={nli_model if use_reranker else 'disabled'}", flush=True)
        self.label_space = label_space
        self.retriever = BiEncoderRetriever(label_space, embed_model, device)
        self.reranker = NLIReranker(nli_model, device) if use_reranker else None
        self.top_candidates = min(top_candidates, len(label_space))
        self.nli_weight = nli_weight
        self.threshold = threshold
        self.max_labels = max_labels
        self.classify_type = classify_type and use_reranker

    # ------------------------------------------------------------------ scoring
    def score_many(self, issues: Sequence[IssueInput]) -> List[List[LabelPrediction]]:
        """Ranked candidate predictions (best first) for each issue."""
        texts = [i.as_text() for i in issues]
        cos = self.retriever.similarities(texts)  # (n, L)
        results: List[List[LabelPrediction]] = []
        try:
            from tqdm import tqdm
            it = tqdm(list(zip(texts, cos)), desc="scoring issues", unit="issue")
        except ImportError:
            it = zip(texts, cos)
        for text, row in it:
            emb_conf = self.retriever.to_confidence(row)
            cand_idx = np.argsort(-row)[: self.top_candidates]
            nli = None
            if self.reranker is not None:
                nli = self.reranker.score_labels(text, [self.label_space.labels[j].text for j in cand_idx])
            preds = []
            for rank, j in enumerate(cand_idx):
                lab = self.label_space.labels[j]
                e = float(emb_conf[j])
                if nli is None:
                    final, n = e, None
                else:
                    n = float(nli[rank])
                    final = self.nli_weight * n + (1 - self.nli_weight) * e
                preds.append(LabelPrediction(lab.id, lab.text, final, e, n))
            preds.sort(key=lambda p: p.score, reverse=True)
            results.append(preds)
        return results

    def select(self, ranked: List[LabelPrediction], threshold: Optional[float] = None) -> List[LabelPrediction]:
        thr = self.threshold if threshold is None else threshold
        chosen = [p for p in ranked if p.score >= thr][: self.max_labels]
        return chosen or ranked[:1]  # always return at least the best guess

    # --------------------------------------------------------------- public API
    def analyze_many(self, issues: Sequence[IssueInput]) -> List[IssueUnderstanding]:
        all_ranked = self.score_many(issues)
        out = []
        for issue, ranked in zip(issues, all_ranked):
            chosen = self.select(ranked)
            itype, tconf = ("unknown", 0.0)
            if self.classify_type:
                itype, tconf = self.reranker.classify_type(issue.as_text())
            names = ", ".join(p.name for p in chosen[:3])
            summary = f"This {itype.replace('_', ' ')} appears to be about: {names}."
            methods = ["bi-encoder:" + self.retriever.model_name]
            if self.reranker is not None:
                methods.append("nli:" + self.reranker.model_name)
            out.append(IssueUnderstanding(summary, itype, tconf, chosen, ranked, methods))
        return out

    def analyze(self, issue: IssueInput) -> IssueUnderstanding:
        return self.analyze_many([issue])[0]