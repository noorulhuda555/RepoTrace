from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodeUnit:
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content: str
    parent: str = ""
    language: str = ""
    symbols: List[str] = field(default_factory=list)
    role: str = ""
    is_test: bool = False

    @property
    def qualified_name(self):
        return ".".join(part for part in (self.path, self.parent, self.name) if part)

    @property
    def search_text(self):
        components = [self.path, self.parent, self.name]
        if self.role:
            components.append(self.role)
        if self.kind in {"file", "class"} and self.language == "java":
            components.append(self.content[:500] if self.content else "")
        else:
            components.append(self.content)
        return " ".join(part for part in components if part)


@dataclass
class CodeMatch:
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    score: float
    reason: str
    snippet: str = ""
    signals: Dict[str, float] = field(default_factory=dict)
    role: str = ""
    is_test: bool = False

    def to_dict(self):
        return {
            "path": self.path,
            "kind": self.kind,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 6),
            "reason": self.reason,
            "snippet": self.snippet,
            "role": self.role,
            "is_test": self.is_test,
            "signals": {key: round(value, 6) for key, value in self.signals.items()},
        }

