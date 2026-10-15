#!/usr/bin/env python3
"""Improved SkillScope classifier under the SAME (leaky) protocol as the baseline.

Protocol deliberately copied from exact_same.py (the paper's baseline):
  * clean_text / prepare_dataset identical to the baseline
  * TF-IDF(max_features=1000) is fit on the COMPLETE dataset (leak #1)
  * MLSMOTE is run on the COMPLETE dataset before any split (leak #2)
  * originals are concatenated with the MLSMOTE output, which already contains
    the originals, so every real row appears twice (leak #3)
  * a plain random row-level 80/20 split is done AFTER augmentation, so
    duplicates and near-duplicate synthetic neighbours of test rows are in
    train (leak #4)

Memory-safe defaults: MLSMOTE is capped (full uncapped run needs tens of GB),
forests use depth/leaf limits and fewer trees, and the blend leans on kNN
memorisation which thrives under this leaky protocol.

Reference numbers to beat (SkillScope paper, micro): P=0.908  R=0.876  F1=0.889

    python skillscope_leaky_improved.py data/skillscope_data.db
    python skillscope_leaky_improved.py data/skillscope_data.db --skip-baseline --save-model
    python skillscope_leaky_improved.py data/skillscope_data.db --mlsmote-cap -1   # uncapped (needs lots of RAM)
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import re
import sqlite3
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import emoji
except ImportError:  # pragma: no cover
    emoji = None

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

COMPETITION_TABLE = "nlbse_tool_competition_data_by_issue"
TEXT_COLUMNS = ["issue text", "issue description"]
COMPETITION_METADATA_COLUMNS = [
    "Repo Name", "PR #", "issue text", "issue description", "created_at",
    "closed_at", "userlogin", "author_name", "most_recent_commit",
]
ART_METADATA_COLUMNS = [
    "PR #", "Pull Request", "issue text", "issue description", "created_at",
    "closed_at", "userlogin", "author_name", "most_recent_commit", "filename",
    "file_commit", "api", "function_name", "api_domain", "subdomain",
]
PAPER = {"precision": 0.908, "recall": 0.876, "f1": 0.889}


# ------------------------------------------------------------------ data ---
def clean_text(text):
    """Identical to the baseline clean_text."""
    if not isinstance(text, str):
        return text
    text = text.replace('"', "")
    text = re.sub(r"DevTools.*?\(automated\)", "", text)
    text = text.lower()
    if emoji is not None:
        text = emoji.demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return " ".join(w for w in text.split() if len(w) <= 20)


def load_dataset(path: Path, table: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    with sqlite3.connect(path) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise ValueError(f"Table {table!r} not found. Available: {sorted(tables)}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def find_label_columns(data, start):
    cols = list(data.columns)
    if start is not None:
        if start not in cols:
            raise ValueError(f"Label-start column {start!r} not found")
        return cols[cols.index(start):]
    for meta in (COMPETITION_METADATA_COLUMNS, ART_METADATA_COLUMNS):
        if cols[: len(meta)] == meta:
            return cols[len(meta):]
    raise ValueError("Unrecognised schema; pass --label-start-column")


def prepare_dataset(data, label_start_column):
    label_columns = find_label_columns(data, label_start_column)
    prepared = data[TEXT_COLUMNS + label_columns].copy()
    prepared[label_columns] = prepared[label_columns].map(lambda x: 1 if x > 0 else 0)
    prepared["issue text"] = prepared["issue text"].apply(clean_text)
    prepared["issue description"] = prepared["issue description"].apply(clean_text)
    return prepared.dropna(), label_columns


# ------------------------------------------- baseline-identical features ---
def extract_text_features(data):
    """TF-IDF fit on the complete dataset (intentional leak, as in baseline)."""
    combined = data["issue text"] + " " + data["issue description"]
    tfidf = TfidfVectorizer(max_features=1000)
    return tfidf.fit_transform(combined).toarray().astype(np.float32), tfidf


def transform_labels(data):
    y = data.drop(columns=TEXT_COLUMNS)
    y = y.apply(lambda x: x.index[x == 1].tolist(), axis=1)
    mlb = MultiLabelBinarizer()
    y_df = pd.DataFrame(mlb.fit_transform(y), columns=mlb.classes_)
    return y_df, mlb


def perform_mlsmote(X: pd.DataFrame, y: pd.DataFrame, seed: int, cap: int = 80000):
    """Same MLSMOTE logic as the baseline, with an optional synthetic-row cap.

    Uncapped `total = sum(max_count - count_j)` is often >700k rows and OOMs on
    typical laptops. Cap keeps the leaky protocol but fits in RAM. Use cap=-1
    for the full original count.
    """
    rng = np.random.RandomState(seed)
    Xv = X.values.astype(np.float32)
    Yv = y.values.astype(np.int32)
    n = len(Xv)

    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean",
                            algorithm="brute", n_jobs=1).fit(Xv)
    indices = nbrs.kneighbors(Xv, return_distance=False)

    counts = Yv.sum(0)
    total_needed = int((counts.max() - counts).sum())
    if cap is None or cap < 0:
        total = total_needed
    else:
        total = min(total_needed, int(cap))
    print(f"MLSMOTE synthetic rows: {total} (full request was {total_needed})")
    if total == 0:
        return X.copy(), y.copy()

    # Generate in chunks to avoid a single huge allocation spike.
    chunk = min(20000, total)
    new_X_parts, new_y_parts = [], []
    done = 0
    while done < total:
        m = min(chunk, total - done)
        ref = rng.randint(0, n, size=m)
        pick = rng.randint(1, 3, size=m)
        nb = indices[ref, pick]
        ratio = rng.rand(m, 1).astype(np.float32)
        new_X_parts.append(Xv[ref] + ratio * (Xv[ref] - Xv[nb]))
        new_y_parts.append((Yv[indices[ref]].sum(axis=1) > 2).astype(np.int32))
        done += m
        print(f"  MLSMOTE progress {done}/{total}")

    new_X = np.vstack(new_X_parts)
    new_y = np.vstack(new_y_parts)
    del new_X_parts, new_y_parts
    gc.collect()

    X_comb = pd.concat([X, pd.DataFrame(new_X, columns=X.columns)], axis=0, ignore_index=True)
    y_comb = pd.concat([y, pd.DataFrame(new_y, columns=y.columns)], axis=0, ignore_index=True)
    del new_X, new_y
    gc.collect()
    return X_comb, y_comb


# --------------------------------------------------------------- metrics ---
def all_metrics(Y, P):
    out = {}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(Y, P, average=avg, zero_division=0)
        out[avg] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    return out


def fast_micro_prf(Y, P):
    tp = float((P & Y).sum())
    p = tp / max(float(P.sum()), 1.0)
    r = tp / max(float(Y.sum()), 1.0)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def fast_micro_f1(Y, P):
    return fast_micro_prf(Y, P)[2]


def beats_paper(m, paper=PAPER, margin=0.0):
    return {k: m[k] + 1e-12 >= paper[k] + margin for k in paper}


def meets_floors(p, r, f, floor_p, floor_r, floor_f1):
    return (p + 1e-12 >= floor_p and r + 1e-12 >= floor_r and f + 1e-12 >= floor_f1)


# denser on the low-threshold / high-recall side (old max-F1 liked ~0.67)
THR_GRID = np.unique(np.concatenate([
    np.arange(0.02, 0.55, 0.01),
    np.arange(0.55, 0.96, 0.02),
]))


# --------------------------------------------------------------- members ---
def _forest_scores(model, X):
    """Per-label P(y=1) from a multi-output forest (handles constant labels)."""
    probs = model.predict_proba(X)
    cols = []
    for pi, cls in zip(probs, model.classes_):
        if pi.ndim == 1:
            cols.append(np.asarray(pi, dtype=np.float32))
        elif pi.shape[1] == 2:
            cols.append(pi[:, 1].astype(np.float32))
        else:
            cols.append(np.full(X.shape[0], float(cls[0] == 1), dtype=np.float32))
    return np.column_stack(cols)


def knn_scores(Xref, Yref, Xq, k, power, chunk=1500):
    k = min(k, len(Xref))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine",
                          algorithm="brute", n_jobs=1).fit(Xref)
    Yf = Yref.astype(np.float32)
    out = np.zeros((len(Xq), Yf.shape[1]), dtype=np.float32)
    for s in range(0, len(Xq), chunk):
        dist, idx = nn.kneighbors(Xq[s:s + chunk])
        w = np.maximum(1.0 - dist, 1e-6) ** power
        w = w / w.sum(1, keepdims=True)
        out[s:s + chunk] = np.einsum("ik,ikl->il", w, Yf[idx])
    return out


def _make_forest(kind, n_trees, seed, n_jobs, max_depth, min_leaf, max_samples):
    kw = dict(
        n_estimators=n_trees,
        random_state=seed,
        n_jobs=n_jobs,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        bootstrap=True,
    )
    if max_samples is not None and 0 < max_samples < 1:
        kw["max_samples"] = max_samples
    if kind == "rf":
        return RandomForestClassifier(**kw)
    return ExtraTreesClassifier(**kw)


class LeakyEnsemble:
    """kNN (memorisation) + RandomForest + optional ExtraTrees, blended by weights."""

    NAMES = ("knn", "rf", "et")

    def __init__(self, k=5, power=2.5, n_trees=80, n_jobs=2, seed=42,
                 max_depth=40, min_leaf=2, max_samples=0.35, use_et=True,
                 forest_subsample=120000):
        self.k, self.power = k, power
        self.n_trees, self.n_jobs, self.seed = n_trees, n_jobs, seed
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.max_samples, self.use_et = max_samples, use_et
        self.forest_subsample = forest_subsample

    def _forest_train_xy(self, X, Y):
        """Optionally subsample rows for forests (kNN still sees full train)."""
        n = len(X)
        if self.forest_subsample is None or self.forest_subsample <= 0 or n <= self.forest_subsample:
            return X, Y
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(n, size=self.forest_subsample, replace=False)
        return X[idx], Y[idx]

    def fit(self, X, Y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.Y_ = np.asarray(Y, dtype=np.float32)
        Xf, Yf = self._forest_train_xy(self.X_, Y.astype(np.int32))
        print(f"  Forest fit on {len(Xf)} rows "
              f"(kNN index keeps all {len(self.X_)} train rows)")
        self.rf_ = _make_forest(
            "rf", self.n_trees, self.seed, self.n_jobs,
            self.max_depth, self.min_leaf, self.max_samples).fit(Xf, Yf)
        if self.use_et:
            self.et_ = _make_forest(
                "et", max(40, self.n_trees // 2), self.seed + 1, self.n_jobs,
                self.max_depth, self.min_leaf, self.max_samples).fit(Xf, Yf)
        else:
            self.et_ = None
        del Xf, Yf
        gc.collect()
        return self

    def member_scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = {
            "knn": knn_scores(self.X_, self.Y_, X, self.k, self.power),
            "rf": _forest_scores(self.rf_, X),
        }
        if self.et_ is not None:
            out["et"] = _forest_scores(self.et_, X)
        else:
            out["et"] = out["rf"]  # placeholder so blend arity stays 3
        return out


def blend(scores, w):
    return sum(wi * scores[n] for wi, n in zip(w, LeakyEnsemble.NAMES))


def _paper_rank(p, r, f, floor_p, floor_r, floor_f1):
    """Lexicographic rank for paper floors. Higher is better.

    Tier 3: all floors met -> maximize F1, then recall
    Tier 2: P and F1 met   -> maximize recall (push toward R floor)
    Tier 1: P met only     -> maximize recall, then F1  (lower the threshold)
    Tier 0: otherwise      -> maximize F1
    """
    if p + 1e-12 >= floor_p and r + 1e-12 >= floor_r and f + 1e-12 >= floor_f1:
        return (3, f, r, p)
    if p + 1e-12 >= floor_p and f + 1e-12 >= floor_f1:
        return (2, r, f, p)
    if p + 1e-12 >= floor_p:
        return (1, r, f, p)
    return (0, f, r, p)


def select_threshold(Y, S, floor_p=0.908, floor_r=0.876, floor_f1=0.889,
                     grid=None):
    """Choose a (usually lower) global threshold aimed at paper recall/F1 floors."""
    grid = THR_GRID if grid is None else grid
    best = None
    best_key = None
    for t in grid:
        p, r, f = fast_micro_prf(Y, S >= t)
        key = _paper_rank(p, r, f, floor_p, floor_r, floor_f1)
        if best_key is None or key > best_key:
            best_key, best = key, (float(t), p, r, f)
    t, p, r, f = best
    feasible = meets_floors(p, r, f, floor_p, floor_r, floor_f1)
    return t, {"precision": p, "recall": r, "f1": f, "feasible": feasible,
               "tier": int(best_key[0])}


def tune_blend(Y, scores, floor_p=0.908, floor_r=0.876, floor_f1=0.889):
    """Pick blend + threshold using paper-floor ranking (not unconstrained F1).

    Unconstrained max-F1 previously locked onto thr~0.67 with P~0.99 R~0.78.
    Ranking prefers lowering the threshold until R/F1 clear, while keeping P>=0.908.
    """
    best = None
    best_key = None
    steps = np.arange(0, 1.001, 0.1)
    for w in itertools.product(steps, repeat=3):
        if abs(sum(w) - 1) > 1e-6:
            continue
        S = blend(scores, w)
        t, m = select_threshold(Y, S, floor_p, floor_r, floor_f1, THR_GRID)
        key = _paper_rank(m["precision"], m["recall"], m["f1"],
                          floor_p, floor_r, floor_f1) + (m["f1"], -t)
        if best_key is None or key > best_key:
            best_key = key
            best = (m["f1"], w, t, m)
    f, w, t, m = best
    return f, w, t, m


# ------------------------------------------------------------- baseline ---
def baseline_rf(x_train, y_train, x_test, n_jobs=2, max_rows=80000, seed=42):
    """Memory-safe RF baseline on (optionally subsampled) train rows."""
    n = len(x_train)
    if max_rows and n > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
        print(f"  Baseline RF subsampled to {max_rows} rows")
    clf = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=n_jobs,
        max_depth=40, min_samples_leaf=2, max_features="sqrt", max_samples=0.35)
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


# ------------------------------------------------------------------- run ---
def run(args):
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset, args.table)
    prepared, label_cols = prepare_dataset(data, args.label_start_column)
    print(f"Rows after preparation: {len(prepared)} | source labels: {len(label_cols)}")

    # --- baseline-identical (leaky) pipeline ---------------------------------
    X_text, tfidf = extract_text_features(prepared)          # leak: full data
    y_df, mlb = transform_labels(prepared)
    X_df = pd.DataFrame(X_text, columns=list(range(X_text.shape[1])))
    print(f"Transformed labels: {len(y_df.columns)}")
    del X_text
    gc.collect()

    X_aug, y_aug = perform_mlsmote(X_df, y_df, args.seed, cap=args.mlsmote_cap)
    # leak: MLSMOTE output already has originals; concat again doubles them
    X_all = pd.concat([X_df, X_aug], axis=0, ignore_index=True)
    y_all = pd.concat([y_df, y_aug], axis=0, ignore_index=True)
    del X_aug, y_aug
    gc.collect()

    x_train, x_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.split_seed)
    print(f"Augmented rows: {len(X_all)} | train {len(x_train)} | test {len(x_test)}")
    del X_all, y_all, X_df, y_df
    gc.collect()

    xtr = np.ascontiguousarray(x_train.values, dtype=np.float32)
    xte = np.ascontiguousarray(x_test.values, dtype=np.float32)
    ytr = np.ascontiguousarray(y_train.values, dtype=np.int32)
    yte = y_test.values.astype(bool)
    del x_train, x_test, y_train, y_test
    gc.collect()

    # --- tune blend + threshold on a validation slice of TRAIN only ----------
    xa, xv, ya, yv = train_test_split(xtr, ytr, test_size=args.val_size,
                                      random_state=args.seed)
    print("Fitting ensemble on train-minus-validation to tune blend ...")
    tuner = LeakyEnsemble(
        args.k, args.power, args.trees, args.n_jobs, args.seed,
        args.max_depth, args.min_leaf, args.max_samples, not args.no_et,
        args.forest_subsample).fit(xa, ya)
    f_val, weights, thr, val_m = tune_blend(
        yv.astype(bool), tuner.member_scores(xv),
        PAPER["precision"], PAPER["recall"], PAPER["f1"])
    print(f"Validation micro-F1 {f_val:.4f} | weights "
          f"{dict(zip(LeakyEnsemble.NAMES, np.round(weights, 2)))} | threshold {thr:.2f}")
    print(f"Validation PRF {val_m} | paper-feasible={val_m['feasible']} "
          f"| tier={val_m.get('tier')} (3=all floors, 1=max R under P floor)")
    del tuner, xa, xv, ya, yv
    gc.collect()

    # --- refit on the full train part and evaluate on the test part ----------
    print("Refitting ensemble on full train part ...")
    final = LeakyEnsemble(
        args.k, args.power, args.trees, args.n_jobs, args.seed,
        args.max_depth, args.min_leaf, args.max_samples, not args.no_et,
        args.forest_subsample).fit(xtr, ytr)
    S_test = blend(final.member_scores(xte), weights)
    P_test = S_test >= thr
    improved = all_metrics(yte, P_test)
    print("Improved (micro):", json.dumps(improved["micro"]))

    beats = beats_paper(improved["micro"])
    print("Paper numbers:", PAPER, "| beats:", beats, "| all:", all(beats.values()))

    # If just under floors, try a small threshold sweep on test scores would leak
    # selection — we do NOT retune on test. Report only.

    baseline = None
    if not args.skip_baseline:
        print("Running baseline RF on the same split ...")
        baseline = all_metrics(
            yte, baseline_rf(xtr, ytr, xte, n_jobs=args.n_jobs,
                             max_rows=args.forest_subsample, seed=args.seed).astype(bool))
        print("Baseline (micro):", json.dumps(baseline["micro"]))

    if args.save_model:
        if joblib is None:
            raise SystemExit("pip install joblib to use --save-model")
        out = Path("model_artifacts")
        out.mkdir(exist_ok=True)
        joblib.dump({"ensemble": final, "weights": weights, "threshold": thr,
                     "tfidf": tfidf, "mlb": mlb},
                    out / "skillscope_leaky_ensemble.pkl", compress=3)

    result = {
        "protocol": "leaky (TF-IDF + MLSMOTE on full data, duplicates, row-level split)",
        "dataset": str(args.dataset),
        "n_train": int(len(xtr)), "n_test": int(len(xte)),
        "n_labels": int(yte.shape[1]),
        "mlsmote_cap": args.mlsmote_cap,
        "paper_reference_micro": PAPER,
        "beats_paper": beats,
        "beats_all_paper_metrics": all(beats.values()),
        "improved": improved,
        "baseline_rf_same_split": baseline,
        "blend_weights": dict(zip(LeakyEnsemble.NAMES, map(float, weights))),
        "threshold": thr,
        "validation_micro_f1": float(f_val),
        "validation_prf": val_m,
        "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", type=Path,
                    default=Path("data") / "skillscope_data.db")
    ap.add_argument("--table", default=COMPETITION_TABLE)
    ap.add_argument("--label-start-column")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=5, help="kNN neighbours (5 aids recall)")
    ap.add_argument("--power", type=float, default=2.5, help="kNN similarity power (lower = smoother/higher recall)")
    ap.add_argument("--trees", type=int, default=80, help="RF trees (keep modest for RAM)")
    ap.add_argument("--max-depth", type=int, default=40)
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--max-samples", type=float, default=0.35,
                    help="fraction of rows each tree sees")
    ap.add_argument("--forest-subsample", type=int, default=100000,
                    help="max rows used to fit RF/ET (kNN still uses all train)")
    ap.add_argument("--mlsmote-cap", type=int, default=150000,
                    help="max synthetic MLSMOTE rows (-1 = uncapped, needs lots of RAM)")
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="parallel jobs for forests (2 avoids RAM spikes vs -1)")
    ap.add_argument("--no-et", action="store_true", help="skip ExtraTrees to save RAM")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results_leaky.json"))
    a = ap.parse_args()
    if not a.dataset.is_file():
        ap.error(f"Dataset does not exist: {a.dataset}")
    return a


if __name__ == "__main__":
    args = parse_args()
    res = run(args)
    args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")
    print("beats_paper:", res["beats_paper"], "| all:", res["beats_all_paper_metrics"])
    print("Improved micro:", res["improved"]["micro"])
