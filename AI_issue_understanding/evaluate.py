"""Evaluate the analyzer against the ground-truth labels stored in skillscope_data.db.

Ground-truth labels are used here ONLY for scoring; they are never given to the models.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .schemas import IssueInput, LabelPrediction


def _prf(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ranking_metrics(ranked: List[List[LabelPrediction]], truth: List[set], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    m: Dict[str, float] = {}
    pairs = [(r, t) for r, t in zip(ranked, truth) if t]  # only issues with >=1 gold label
    m["n_eval_issues"] = len(pairs)
    for k in ks:
        hit = prec = rec = 0.0
        for r, t in pairs:
            top = {p.label_id for p in r[:k]}
            inter = len(top & t)
            hit += inter > 0
            prec += inter / k
            rec += inter / len(t)
        n = max(len(pairs), 1)
        m[f"hit@{k}"], m[f"precision@{k}"], m[f"recall@{k}"] = hit / n, prec / n, rec / n
    mrr = 0.0
    for r, t in pairs:
        for i, p in enumerate(r, 1):
            if p.label_id in t:
                mrr += 1 / i
                break
    m["mrr"] = mrr / max(len(pairs), 1)
    return m


def threshold_metrics(ranked, truth, threshold: float, max_labels: int = 8) -> Dict[str, float]:
    tp = fp = fn = 0
    per_label: Dict[str, List[int]] = {}
    sample_f = []
    for r, t in zip(ranked, truth):
        pred = {p.label_id for p in [p for p in r if p.score >= threshold][:max_labels]} or {r[0].label_id}
        a, b, c = len(pred & t), len(pred - t), len(t - pred)
        tp, fp, fn = tp + a, fp + b, fn + c
        sample_f.append(_prf(a, b, c)[2])
        for l in pred | t:
            s = per_label.setdefault(l, [0, 0, 0])
            s[0] += l in pred and l in t
            s[1] += l in pred and l not in t
            s[2] += l not in pred and l in t
    mp, mr, mf = _prf(tp, fp, fn)
    macro = float(np.mean([_prf(*v)[2] for v in per_label.values()])) if per_label else 0.0
    return {
        "threshold": threshold, "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "macro_f1": macro, "samples_f1": float(np.mean(sample_f)) if sample_f else 0.0,
    }


def evaluate(analyzer, records: Sequence[dict], thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> Dict:
    issues = [IssueInput(r["title"], r["body"], r["issue_id"]) for r in records]
    truth = [set(r["labels"]) for r in records]
    ranked = analyzer.score_many(issues)
    sweep = [threshold_metrics(ranked, truth, t, analyzer.max_labels) for t in thresholds]
    return {
        "ranking": ranking_metrics(ranked, truth),
        "threshold_sweep": sweep,
        "best_threshold_by_micro_f1": max(sweep, key=lambda s: s["micro_f1"]),
    }
