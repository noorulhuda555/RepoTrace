"""Data structures for the AI Issue Understanding module."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IssueInput:
    """A raw issue as written by a human (title + free text)."""

    title: str
    body: str = ""
    issue_id: Optional[str] = None

    def as_text(self, max_chars: int = 2500) -> str:
        text = self.title.strip()
        if self.body.strip():
            text += "\n\n" + self.body.strip()
        if len(text) > max_chars:
            head = int(max_chars * 0.75)
            text = text[:head] + " ... " + text[-(max_chars - head):]
        return text


@dataclass
class LabelPrediction:
    label_id: str          # original column name in the DB
    name: str              # human-readable label
    score: float           # fused confidence in [0, 1]
    embedding_score: float = 0.0   # bi-encoder confidence
    nli_score: Optional[float] = None  # cross-encoder entailment probability


@dataclass
class IssueUnderstanding:
    summary: str
    issue_type: str                      # bug | feature_request | question | documentation | refactoring | performance
    issue_type_confidence: float
    labels: List[LabelPrediction]        # selected labels, best first
    candidates: List[LabelPrediction] = field(default_factory=list)  # full ranked shortlist
    methods_used: List[str] = field(default_factory=list)

    @property
    def primary(self) -> Optional[LabelPrediction]:
        return self.labels[0] if self.labels else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
