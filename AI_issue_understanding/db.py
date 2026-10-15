"""Loader for `skillscope_data.db` (NLBSE skill-classification dataset).

The exact column layout can vary between dataset releases, so columns are auto-detected
heuristically and can always be overridden explicitly. Run `python -m
repotrace.issue_understanding inspect --db skillscope_data.db` to see what was detected.
"""
from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_TABLE = "nlbse_tool_competition_data_by_issue"

_TEXT_HINT = re.compile(r"title|body|text|descr|summary|message", re.I)
# columns that hold code-context rather than the human-written issue/PR text
_CODE_HINT = re.compile(r"code|method|class|file|path|func|api|diff|patch|url|link|sha|hash|repo|author|label|domain", re.I)
# numeric columns that are metadata, not skill labels
_META_HINT = re.compile(
    r"(^|[_\s])(id|ids|number|num|no|index|idx|rowid|repo|repository|pr|issue|stars|forks|lines|loc|"
    r"additions|deletions|changed|size|year|month|day|timestamp|date|created|merged|total|count)($|[_\s])",
    re.I,
)


def q(identifier: str) -> str:
    """Quote an SQL identifier."""
    return '"' + identifier.replace('"', '""') + '"'


@dataclass
class DetectedSchema:
    table: str
    text_columns: List[str]
    label_columns: List[str]
    skipped_numeric: List[str]
    all_columns: List[Tuple[str, str]]


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name").fetchall()
    return [r[0] for r in rows]


def _effective_type(conn, table: str, col: str, declared: str) -> str:
    d = (declared or "").upper()
    if d:
        return d
    row = conn.execute(f"SELECT typeof({q(col)}) FROM {q(table)} WHERE {q(col)} IS NOT NULL LIMIT 1").fetchone()
    return (row[0] if row else "").upper()


def detect_schema(
    conn: sqlite3.Connection,
    table: str = DEFAULT_TABLE,
    text_columns: Optional[Sequence[str]] = None,
    label_columns: Optional[Sequence[str]] = None,
    label_regex: Optional[str] = None,
) -> DetectedSchema:
    info = conn.execute(f"PRAGMA table_info({q(table)})").fetchall()
    if not info:
        raise ValueError(f"Table/view '{table}' not found. Available: {list_tables(conn)}")
    cols = [(r["name"], _effective_type(conn, table, r["name"], r["type"])) for r in info]

    # ---- text columns (human-written issue / PR text only; never code-context by default)
    if text_columns:
        text_cols = list(text_columns)
    else:
        text_cols = [
            n for n, t in cols
            if ("CHAR" in t or "TEXT" in t or "CLOB" in t) and _TEXT_HINT.search(n) and not _CODE_HINT.search(n)
        ]

    # ---- label columns
    skipped: List[str] = []
    if label_columns:
        label_cols = list(label_columns)
    else:
        label_cols = []
        rx = re.compile(label_regex) if label_regex else None
        for n, t in cols:
            if n in text_cols or not any(k in t for k in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "BOOL")):
                continue
            if rx is not None:
                (label_cols if rx.search(n) else skipped).append(n)
                continue
            if _META_HINT.search(n):
                skipped.append(n)
                continue
            mn, mx, pos = conn.execute(
                f"SELECT MIN({q(n)}), MAX({q(n)}), SUM({q(n)} > 0) FROM {q(table)}"
            ).fetchone()
            if mn is None or mn < 0 or not pos:
                skipped.append(n)
                continue
            label_cols.append(n)
    return DetectedSchema(table, text_cols, label_cols, skipped, cols)


def load_issues(
    conn: sqlite3.Connection,
    schema: DetectedSchema,
    limit: Optional[int] = None,
    seed: int = 13,
) -> List[Dict]:
    """Return [{'issue_id', 'title', 'body', 'labels': set(column names with value > 0)}]."""
    if not schema.text_columns:
        raise ValueError("No text columns detected; pass --text-cols explicitly.")
    select_cols = schema.text_columns + schema.label_columns
    rows = conn.execute(f"SELECT rowid AS __rid, {', '.join(q(c) for c in select_cols)} FROM {q(schema.table)}").fetchall()
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)

    title_col = next((c for c in schema.text_columns if "title" in c.lower()), schema.text_columns[0])
    out = []
    for r in rows:
        title = (r[title_col] or "").strip()
        body = "\n\n".join(
            (r[c] or "").strip() for c in schema.text_columns if c != title_col and (r[c] or "").strip()
        )
        labels = {c for c in schema.label_columns if (r[c] or 0) > 0}
        out.append({"issue_id": str(r["__rid"]), "title": title, "body": body, "labels": labels})
    return out
