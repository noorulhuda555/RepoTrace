"""CLI:  python -m repotrace.issue_understanding {inspect|evaluate|analyze} ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import db as dbm
from .analyzer import IssueAnalyzer
from .evaluate import evaluate
from .label_space import LabelSpace
from .schemas import IssueInput


def _split(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _common(p):
    p.add_argument("--db", help="path to skillscope_data.db")
    p.add_argument("--table", default=dbm.DEFAULT_TABLE)
    p.add_argument("--text-cols", help="comma-separated text columns (override auto-detect)")
    p.add_argument("--label-cols", help="comma-separated label columns (override auto-detect)")
    p.add_argument("--label-regex", help="regex selecting label columns among numeric columns")


def _model_args(p):
    p.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--nli-model", default=None, help="default: large on GPU, base on CPU")
    p.add_argument("--device", default=None)
    p.add_argument("--no-rerank", action="store_true", help="embedding-only (fast baseline)")
    p.add_argument("--top-candidates", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.5)


def _schema(a):
    conn = dbm.connect(a.db)
    schema = dbm.detect_schema(conn, a.table, _split(a.text_cols), _split(a.label_cols), a.label_regex)
    return conn, schema


def _analyzer(a, space):
    return IssueAnalyzer(
        space, a.embed_model, a.nli_model, a.device, use_reranker=not a.no_rerank,
        top_candidates=a.top_candidates, threshold=a.threshold,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="repotrace.issue_understanding")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="show detected text/label columns")
    _common(pi)

    pe = sub.add_parser("evaluate", help="score the analyzer on issues from the DB")
    _common(pe)
    _model_args(pe)
    pe.add_argument("--limit", type=int, default=50)
    pe.add_argument("--seed", type=int, default=13)
    pe.add_argument("--out", help="write metrics JSON here")

    pa = sub.add_parser("analyze", help="analyze a single issue")
    _common(pa)
    _model_args(pa)
    pa.add_argument("--labels-json", help="label space JSON (alternative to --db)")
    pa.add_argument("--save-labels", help="save the label space JSON for later use")
    pa.add_argument("--title", required=True)
    pa.add_argument("--body", default="")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)

    if a.cmd == "inspect":
        _, s = _schema(a)
        print(f"table: {s.table}\ntext columns ({len(s.text_columns)}): {s.text_columns}")
        print(f"label columns ({len(s.label_columns)}): {s.label_columns[:15]}{' ...' if len(s.label_columns) > 15 else ''}")
        print(f"skipped numeric columns ({len(s.skipped_numeric)}): {s.skipped_numeric}")
        print("\nIf this is wrong, pass --text-cols / --label-cols / --label-regex.")
        return 0

    if a.cmd == "evaluate":
        conn, s = _schema(a)
        records = dbm.load_issues(conn, s, a.limit, a.seed)
        space = LabelSpace.from_columns(s.label_columns)
        print(f"{len(records)} issues, {len(space)} labels, text columns: {s.text_columns}", file=sys.stderr)
        res = evaluate(_analyzer(a, space), records)
        print(json.dumps(res, indent=2))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=2)
        return 0

    if a.cmd == "analyze":
        if a.labels_json:
            space = LabelSpace.load(a.labels_json)
        elif a.db:
            _, s = _schema(a)
            space = LabelSpace.from_columns(s.label_columns)
        else:
            ap.error("analyze needs --db or --labels-json")
        if a.save_labels:
            space.save(a.save_labels)
        r = _analyzer(a, space).analyze(IssueInput(a.title, a.body))
        d = r.to_dict()
        d.pop("candidates")
        print(json.dumps(d, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())