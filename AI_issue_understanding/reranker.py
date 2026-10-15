"""Stage 2: zero-shot NLI cross-encoder reranking (pretrained, no fine-tuning).

Default: MoritzLaurer/deberta-v3-large-zeroshot-v2.0, a strong open zero-shot classifier.
For every candidate label we ask: does the issue text *entail* "This issue involves <label>."?
Also used to infer the issue type (bug / feature request / ...).
"""
from __future__ import annotations
import logging
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

ISSUE_TYPES = {
    "bug": "This text reports a bug or something that is broken.",
    "feature_request": "This text requests a new feature or an enhancement.",
    "question": "This text asks a question or asks for help using the software.",
    "documentation": "This text is about documentation.",
    "refactoring": "This text is about refactoring or code cleanup.",
    "performance": "This text is about slow performance or resource usage.",
}


class NLIReranker:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 320,
        premise_chars: int = 1000,
        hypothesis_template: str = "This issue involves {}.",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.premise_chars = premise_chars
        self.hypothesis_template = hypothesis_template
        self._model = None
        self._tok = None
        self._ent_idx = 0
        self._neg_idx: Optional[int] = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading NLI model %s on %s", self.model_name, dev)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if dev == "cuda":
            model = model.half()
        self._model = model.to(dev).eval()
        self.device = dev

        id2label = {int(i): str(l).lower() for i, l in self._model.config.id2label.items()}
        ent = [i for i, l in id2label.items() if "entail" in l and not l.startswith("not")]
        neg = [i for i, l in id2label.items() if "contradict" in l or l.startswith("not")]
        if not ent:
            raise RuntimeError(f"{self.model_name} has no entailment label: {id2label}")
        self._ent_idx = ent[0]
        self._neg_idx = neg[0] if neg else None

    def entailment(self, premise: str, hypotheses: Sequence[str]) -> List[float]:
        """P(entailment) for each hypothesis given one premise."""
        import torch

        self._ensure_ready()
        premise = premise[: self.premise_chars]
        out: List[float] = []
        for i in range(0, len(hypotheses), self.batch_size):
            batch = list(hypotheses[i : i + self.batch_size])
            enc = self._tok(
                [premise] * len(batch), batch, truncation="only_first",
                max_length=self.max_length, padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**enc).logits.float()
            if self._neg_idx is not None:  # 2-way softmax: entailment vs not-entailment/contradiction
                two = logits[:, [self._ent_idx, self._neg_idx]]
                probs = torch.softmax(two, dim=-1)[:, 0]
            else:
                probs = torch.softmax(logits, dim=-1)[:, self._ent_idx]
            out.extend(probs.cpu().tolist())
        return out

    def score_labels(self, premise: str, label_texts: Sequence[str]) -> List[float]:
        return self.entailment(premise, [self.hypothesis_template.format(t) for t in label_texts])

    def classify_type(self, premise: str) -> Tuple[str, float]:
        names = list(ISSUE_TYPES)
        probs = self.entailment(premise, [ISSUE_TYPES[n] for n in names])
        best = max(range(len(names)), key=lambda i: probs[i])
        return names[best], float(probs[best])