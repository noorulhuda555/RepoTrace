#!/usr/bin/env python3
"""Pipeline: load SkillScope data -> leak-free split -> train improved
classifier -> compare with the original RF baseline -> results_1.json

Default mode is competition floors (P>=0.45, R>=0.76, F1>=0.3248).

    python skill_classifier_test_pipeline.py data\\skillscope_data.db
    python skill_classifier_test_pipeline.py data\\skillscope_data.db --mode competition --output results_1.json
    python skill_classifier_test_pipeline.py data\\skillscope_data.db --mode rf          # legacy RF floors
    python skill_classifier_test_pipeline.py data\\skillscope_data.db --mode f1          # unconstrained F1
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import string
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

from skill_classifier import SkillEnsembleClassifier

try:
    import emoji
except ImportError:
    emoji = None

COMPETITION_TABLE = "nlbse_tool_competition_data_by_issue"
COMPETITION_METADATA_COLUMNS = [
    "Repo Name", "PR #", "issue text", "issue description", "created_at",
    "closed_at", "userlogin", "author_name", "most_recent_commit",
]
ART_METADATA_COLUMNS = [
    "PR #", "Pull Request", "issue text", "issue description", "created_at",
    "closed_at", "userlogin", "author_name", "most_recent_commit", "filename",
    "file_commit", "api", "function_name", "api_domain", "subdomain",
]


# ------------------------------------------------------------------ data ---
def load_dataset(path: Path, table: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    with sqlite3.connect(path) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise ValueError(f"Table {table!r} not found. Available: {sorted(tables)}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def find_label_columns(data: pd.DataFrame, start):
    cols = list(data.columns)
    if start is not None:
        if start not in cols:
            raise ValueError(f"Label-start column {start!r} not found")
        return cols[cols.index(start):]
    for meta in (COMPETITION_METADATA_COLUMNS, ART_METADATA_COLUMNS):
        if cols[: len(meta)] == meta:
            return cols[len(meta):]
    raise ValueError("Unrecognised schema; pass --label-start-column")


# -------------------------------------------------------------- baseline ---
def baseline_clean_text(text):
    """Original clean_text from main.py."""
    if not isinstance(text, str):
        return ""
    text = text.replace('"', "")
    text = re.sub(r"DevTools.*?\(automated\)", "", text)
    text = text.lower()
    if emoji is not None:
        text = emoji.demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return " ".join(w for w in text.split() if len(w) <= 20)


def rf_baseline_predict(docs_fit, Y_fit, docs_pred):
    """Original baseline: TF-IDF(1000) + default RandomForest(random_state=42)."""
    tf = TfidfVectorizer(max_features=1000)
    Xa = tf.fit_transform(docs_fit).toarray()
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(Xa, Y_fit.astype(int))
    return rf.predict(tf.transform(docs_pred).toarray()).astype(bool)


# --------------------------------------------------------------- metrics ---
def sk_metrics(Y, P):
    out = {}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(Y, P, average=avg, zero_division=0)
        out[avg] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    return out


# ------------------------------------------------------------------- run ---
def run(args) -> dict:
    t0 = time.time()
    data = load_dataset(args.dataset, args.table)
    label_cols = find_label_columns(data, args.label_start_column)
    print(f"Rows: {len(data)} | labels: {len(label_cols)}")

    Y = (data[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values > 0)
    title = data["issue text"].fillna("").astype(str)
    body = data["issue description"].fillna("").astype(str)
    repo = (data["Repo Name"].fillna("unknown").astype(str).to_numpy(dtype=object)
            if "Repo Name" in data.columns
            else np.array(["unknown"] * len(data), dtype=object))

    # duplicate-aware split: identical texts never straddle train/test
    norm = lambda s: s.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    groups = pd.factorize(norm(title) + " || " + norm(body).str[:500])[0]
    tr, te = next(GroupShuffleSplit(1, test_size=args.test_size,
                                    random_state=args.seed)
                  .split(np.zeros(len(data)), groups=groups))
    print(f"Train {len(tr)} | Test {len(te)}")

    t_arr, b_arr = title.to_numpy(dtype=object), body.to_numpy(dtype=object)
    base_docs = (title.map(baseline_clean_text) + " "
                 + body.map(baseline_clean_text)).to_numpy(dtype=object)
    Ytr, Yte = Y[tr], Y[te]

    def baseline_factory(a, b):
        return rf_baseline_predict(base_docs[tr][a], Ytr[a], base_docs[tr][b])

    mode = args.mode
    if args.no_constraint and mode == "competition":
        # legacy flag: unconstrained F1 selection
        mode = "f1"
    clf = SkillEnsembleClassifier(
        folds=args.folds, seed=args.seed, n_jobs=args.n_jobs, mode=mode,
        floor_p=args.floor_p, floor_r=args.floor_r, floor_f1=args.floor_f1)
    use_rf_factory = (mode == "rf" and not args.no_constraint)
    clf.fit(t_arr[tr], b_arr[tr], repo[tr], Ytr, groups[tr],
            baseline_factory=baseline_factory if use_rf_factory else None)

    P_imp = clf.predict(t_arr[te], b_arr[te], repo[te])
    improved = sk_metrics(Yte, P_imp)
    print("Improved (micro):", json.dumps(improved["micro"]))

    floors = {"precision": args.floor_p, "recall": args.floor_r, "f1": args.floor_f1}
    beats = {k: improved["micro"][k] + 1e-12 >= floors[k] for k in floors}
    print("Competition floors:", floors)
    print("Beats floors:", beats, "| all:", all(beats.values()))

    baseline = None
    if not args.skip_baseline:
        print("Running baseline RF on the same train/test split ...")
        baseline = sk_metrics(Yte, rf_baseline_predict(base_docs[tr], Ytr, base_docs[te]))
        print("Baseline (micro):", json.dumps(baseline["micro"]))

    result = {
        "files": ["skill_classifier.py", "skill_classifier_test_pipeline.py"],
        "dataset": str(args.dataset),
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_labels": len(label_cols),
        "mode": mode,
        "competition_floors": floors,
        "beats_competition_floors": beats,
        "beats_all_competition_floors": all(beats.values()),
        "precision": improved["micro"]["precision"],
        "recall": improved["micro"]["recall"],
        "f1": improved["micro"]["f1"],
        "improved": improved,
        "baseline_rf_same_split": baseline,
        "classifier": clf.summary(),
        "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    if baseline is not None:
        ks = ("precision", "recall", "f1")
        for avg in ("micro", "macro"):
            result[f"delta_{avg}"] = {k: improved[avg][k] - baseline[avg][k] for k in ks}
            result[f"all_improved_{avg}"] = all(improved[avg][k] > baseline[avg][k] for k in ks)

    if args.save_model:
        Path("model_artifacts").mkdir(exist_ok=True)
        joblib.dump({"clf": clf, "label_columns": label_cols},
                    "model_artifacts/skill_ensemble.pkl", compress=3)
        print("Model saved to model_artifacts/skill_ensemble.pkl")
    return result


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", type=Path,
                    default=Path("data") / "skillscope_data.db")
    ap.add_argument("--table", default=COMPETITION_TABLE)
    ap.add_argument("--label-start-column")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the final RF run on the test split")
    ap.add_argument("--mode", choices=("competition", "rf", "f1"), default="competition",
                    help="competition=beat P/R/F1 floors (default); rf=legacy RF floors; f1=max F1")
    ap.add_argument("--floor-p", type=float, default=0.45)
    ap.add_argument("--floor-r", type=float, default=0.76)
    ap.add_argument("--floor-f1", type=float, default=0.3248)
    ap.add_argument("--no-constraint", action="store_true",
                    help="with default mode, switch to unconstrained F1; with --mode rf, skip RF floors")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results_1.json"))
    a = ap.parse_args()
    if not a.dataset.is_file():
        ap.error(f"Dataset does not exist: {a.dataset}")
    return a


if __name__ == "__main__":
    args = parse_args()
    res = run(args)
    args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")
    for k in ("beats_all_competition_floors", "beats_competition_floors",
              "all_improved_micro", "delta_micro", "all_improved_macro", "delta_macro"):
        if k in res:
            print(k, res[k])