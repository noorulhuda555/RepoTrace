"""Label space (the 217 domain / sub-domain skills) used for zero-shot classification.

Only the *names* of the label columns are used to build the space. The per-issue label
values stored in the DB are never fed to a model; they are used for evaluation only.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_HIER_SPLIT = re.compile(r"\s*(?:::|__|\||/|>|->)\s*")


def humanize(column_name: str) -> str:
    """'WebDev__Servlet_API' -> 'Web Dev: Servlet API'."""
    parts = [p for p in _HIER_SPLIT.split(column_name.strip()) if p]
    cleaned = []
    for p in parts:
        p = re.sub(r"[_\-.]+", " ", p)
        p = _CAMEL.sub(" ", p)
        p = re.sub(r"\s+", " ", p).strip()
        cleaned.append(p)
    return ": ".join(cleaned) if cleaned else column_name


@dataclass(frozen=True)
class Label:
    id: str                 # exact DB column name
    text: str               # humanised name, used in prompts / hypotheses
    description: str = ""   # optional extra context (see descriptions JSON)

    @property
    def document(self) -> str:
        """Text embedded by the bi-encoder to represent the label."""
        return f"{self.text}. {self.description}".strip() if self.description else self.text


class LabelSpace:
    def __init__(self, labels: Sequence[Label]):
        self.labels: List[Label] = list(labels)
        self.index: Dict[str, int] = {l.id: i for i, l in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_columns(
        cls, columns: Iterable[str], descriptions: Optional[Dict[str, str]] = None
    ) -> "LabelSpace":
        descriptions = descriptions or {}
        return cls([Label(c, humanize(c), descriptions.get(c, "")) for c in columns])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(l) for l in self.labels], f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LabelSpace":
        with open(path, "r", encoding="utf-8") as f:
            return cls([Label(**d) for d in json.load(f)])
