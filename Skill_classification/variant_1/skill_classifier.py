#!/usr/bin/env python3
"""SkillEnsembleClassifier - improved multi-label classifier for SkillScope.

Default selection targets competition-style floors:
  precision >= 0.45, recall >= 0.76, micro-F1 >= 0.3248
by choosing blend weights and a (usually lower) global threshold on OOF scores.
"""

from __future__ import annotations

import itertools
import re
import warnings

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

THR_GRID = np.arange(0.03, 0.91, 0.02)
COARSE_GRID = np.arange(0.05, 0.91, 0.05)
MEMBER_NAMES = ("cnb", "knn", "lr", "repo")  # fixed order used in blends

# Competition-style operating-point floors (beat these on micro metrics).
DEFAULT_FLOOR_P = 0.45
DEFAULT_FLOOR_R = 0.76
DEFAULT_FLOOR_F1 = 0.3248


# ==========================================================================
# Text normalisation
# ==========================================================================
def normalize_text(text, max_len: int = 6000) -> str:
    if not isinstance(text, str):
        return ""
    text = text[:max_len]
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[0-9a-fA-F]{7,40}\b", " hashtoken ", text)
    text = re.sub(r"\s+at\s+[\w.$]+\([\w.]+:\d+\)", " stacktraceline ", text)
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    return (text + " " + split).lower()


# ==========================================================================
# Metrics and threshold utilities
# ==========================================================================
def prf_micro(Y, P):
    tp = float((P & Y).sum())
    p = tp / max(float(P.sum()), 1.0)
    r = tp / max(float(Y.sum()), 1.0)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def objective(Y, P) -> float:
    tp = (P & Y).sum(0)
    fp = P.sum(0) - tp
    fn = Y.sum(0) - tp
    sup = Y.sum(0) > 0
    macro = (2 * tp / np.maximum(2 * tp + fp + fn, 1))[sup].mean()
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * TP / max(2 * TP + FP + FN, 1)
    return float(0.5 * (micro + macro))


def meets_floors(p, r, f, floor_p, floor_r, floor_f1, margin=0.0):
    return (p + 1e-12 >= floor_p + margin
            and r + 1e-12 >= floor_r + margin
            and f + 1e-12 >= floor_f1 + margin)


def best_global_threshold(Y, S, grid=THR_GRID):
    scores = [prf_micro(Y, S >= t)[2] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def competition_threshold(Y, S, floor_p=DEFAULT_FLOOR_P, floor_r=DEFAULT_FLOOR_R,
                          floor_f1=DEFAULT_FLOOR_F1, grid=THR_GRID, margin=0.0):
    """Pick global thr maximizing micro-F1 among points that beat competition floors.

    If no threshold is fully feasible, prefer points that clear precision and F1 floors
    and get as close as possible on recall (then F1). Never falls back to a high-P /
    low-R unconstrained optimum when a high-recall point with P>=floor_p exists.
    """
    best_ok = None          # (f1, recall, precision, t)
    best_partial = None     # P and F1 ok, maximize recall then f1
    best_p_ok = None        # only P ok, maximize recall then f1
    best_any = None         # unconstrained F1 fallback

    for t in grid:
        p, r, f = prf_micro(Y, S >= t)
        key_f = (f, r, p, float(t))
        key_r = (r, f, p, float(t))
        if best_any is None or key_f > best_any:
            best_any = key_f
        if p + 1e-12 >= floor_p + margin:
            if best_p_ok is None or key_r > best_p_ok:
                best_p_ok = key_r
            if f + 1e-12 >= floor_f1 + margin:
                if best_partial is None or key_r > best_partial:
                    best_partial = key_r
                if r + 1e-12 >= floor_r + margin:
                    if best_ok is None or key_f > best_ok:
                        best_ok = key_f

    if best_ok is not None:
        f, r, p, t = best_ok
        return t, True, {"precision": p, "recall": r, "f1": f}
    if best_partial is not None:
        r, f, p, t = best_partial
        return t, False, {"precision": p, "recall": r, "f1": f}
    if best_p_ok is not None:
        r, f, p, t = best_p_ok
        return t, False, {"precision": p, "recall": r, "f1": f}
    f, r, p, t = best_any
    return t, False, {"precision": p, "recall": r, "f1": f}


def constrained_global_threshold(Y, S, rows_ref, floor_p, floor_r, margin=0.003):
    """Legacy RF-style constraint on a subset of rows (kept for --mode rf)."""
    best_t, best_f = None, -1.0
    for t in THR_GRID:
        P = S >= t
        pb, rb, _ = prf_micro(Y[rows_ref], P[rows_ref])
        if pb >= floor_p + margin and rb >= floor_r + margin:
            f = prf_micro(Y, P)[2]
            if f > best_f:
                best_t, best_f = float(t), f
    if best_t is None:
        return best_global_threshold(Y, S)[0], False
    return best_t, True


def tune_thresholds(S, Y, t_global, shrink_k=15):
    L = S.shape[1]
    thr = np.full(L, t_global)
    for j in range(L):
        npos = int(Y[:, j].sum())
        if npos == 0:
            thr[j] = 2.0
            continue
        P = S[:, j][:, None] >= THR_GRID[None, :]
        y = Y[:, j][:, None]
        tp = (P & y).sum(0)
        fp = P.sum(0) - tp
        fn = npos - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        if f1.max() <= 0:
            continue
        lam = npos / (npos + shrink_k)
        thr[j] = lam * THR_GRID[int(np.argmax(f1))] + (1 - lam) * t_global
    return thr


def apply_thresholds(S, thr):
    P = S >= thr
    empty = ~P.any(axis=1)
    if empty.any():
        ratio = np.where(thr <= 1.0, S / np.maximum(thr, 1e-9), -np.inf)
        P[np.where(empty)[0], ratio[empty].argmax(axis=1)] = True
    return P


def _rank_blend(Y, S, floor_p, floor_r, floor_f1, grid=COARSE_GRID):
    """Higher is better. Feasible competition points beat all infeasible ones."""
    t, ok, m = competition_threshold(Y, S, floor_p, floor_r, floor_f1, grid=grid)
    # Lexicographic: feasible first, then F1, then recall, then precision.
    return (1 if ok else 0, m["f1"], m["recall"], m["precision"], -t), t, ok, m


# ==========================================================================
# Features
# ==========================================================================
class TextFeaturizer:
    def __init__(self, char_weight=0.7, repo_weight=1.25):
        self.char_weight, self.repo_weight = char_weight, repo_weight

    @staticmethod
    def _docs(titles, bodies):
        t = [normalize_text(x) for x in titles]
        b = [normalize_text(x) for x in bodies]
        return np.array([f"{a} {a} {c}" for a, c in zip(t, b)], dtype=object)

    def fit(self, titles, bodies, repos):
        docs = self._docs(titles, bodies)
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=40000, sublinear_tf=True,
                                     dtype=np.float32).fit(docs)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, max_features=40000,
                                     sublinear_tf=True, dtype=np.float32).fit(docs)
        self.repo_ = OneHotEncoder(handle_unknown="ignore").fit(
            np.asarray(repos, dtype=object).reshape(-1, 1))
        return self

    def transform(self, titles, bodies, repos):
        docs = self._docs(titles, bodies)
        Xw = self.word_.transform(docs)
        Xc = self.char_.transform(docs)
        Xr = self.repo_.transform(np.asarray(repos, dtype=object).reshape(-1, 1))
        X = sp.hstack([Xw, self.char_weight * Xc, self.repo_weight * Xr]).tocsr()
        return X, Xw.tocsr()


# ==========================================================================
# Scorers
# ==========================================================================
def _fit_lr(X, y, C, pos_power=0.65, pos_cap=12.0):
    pos = int(y.sum())
    if pos == 0:
        return 0.0
    if pos == len(y):
        return 1.0
    # Slightly stronger positive weight -> higher recall scores.
    pw = float(min(pos_cap, max(1.0, ((len(y) - pos) / pos) ** pos_power)))
    return LogisticRegression(C=C, solver="liblinear",
                              class_weight={0: 1.0, 1: pw}).fit(X, y)


class LRHeads:
    def __init__(self, C=8.0, n_jobs=-1, pos_power=0.65, pos_cap=12.0):
        self.C, self.n_jobs = C, n_jobs
        self.pos_power, self.pos_cap = pos_power, pos_cap

    def fit(self, X, Y):
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_lr)(X, Y[:, j].astype(int), self.C, self.pos_power, self.pos_cap)
            for j in range(Y.shape[1]))
        return self

    def predict(self, X):
        out = np.zeros((X.shape[0], len(self.models_)))
        for j, m in enumerate(self.models_):
            out[:, j] = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m
        return out


class NBHeads:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, Xw, Y):
        self.models_ = []
        for j in range(Y.shape[1]):
            y = Y[:, j].astype(int)
            if y.sum() == 0:
                self.models_.append(0.0)
            elif y.sum() == len(y):
                self.models_.append(1.0)
            else:
                self.models_.append(ComplementNB(alpha=self.alpha).fit(Xw, y))
        return self

    def predict(self, Xw):
        out = np.zeros((Xw.shape[0], len(self.models_)))
        for j, m in enumerate(self.models_):
            out[:, j] = m.predict_proba(Xw)[:, 1] if hasattr(m, "predict_proba") else m
        return out


class RepoPrior:
    def __init__(self, alpha=5.0):
        self.alpha = alpha

    def fit(self, repos, Y):
        repos = np.asarray(repos, dtype=object)
        Yf = Y.astype(float)
        self.glob_ = Yf.mean(0)
        self.table_ = {}
        for r in np.unique(repos):
            m = repos == r
            self.table_[r] = (Yf[m].sum(0) + self.alpha * self.glob_) / (m.sum() + self.alpha)
        return self

    def predict(self, repos):
        return np.vstack([self.table_.get(r, self.glob_) for r in repos])


class KNNTransfer:
    def __init__(self, k=25, boost=1.75, chunk=400):
        self.k, self.boost, self.chunk = k, boost, chunk

    def fit(self, Xw, Y, repos):
        self.XT_ = Xw.T.tocsc()
        self.Y_ = Y.astype(np.float32)
        self.repos_ = np.asarray(repos, dtype=object)
        return self

    def predict(self, Xq, repos):
        repos = np.asarray(repos, dtype=object)
        k = max(1, min(self.k, self.Y_.shape[0] - 1))
        out = np.zeros((Xq.shape[0], self.Y_.shape[1]))
        for s in range(0, Xq.shape[0], self.chunk):
            sims = (Xq[s:s + self.chunk] @ self.XT_).toarray()
            same = repos[s:s + self.chunk][:, None] == self.repos_[None, :]
            sims = sims * np.where(same, self.boost, 1.0)
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            # Milder power than ^2 keeps more neighbours and lifts recall.
            w = np.take_along_axis(sims, idx, axis=1) ** 1.5
            w = w / (w.sum(1, keepdims=True) + 1e-9)
            out[s:s + self.chunk] = np.einsum("ik,ikl->il", w, self.Y_[idx])
        return out


# ==========================================================================
# Ensemble
# ==========================================================================
def _fit_members(X, Xw, Y, repos, n_jobs):
    return {
        "cnb": NBHeads().fit(Xw, Y),
        "knn": KNNTransfer().fit(Xw, Y, repos),
        "lr": LRHeads(n_jobs=n_jobs).fit(X, Y),
        "repo": RepoPrior().fit(repos, Y),
    }


def _score_members(fitted, X, Xw, repos):
    return {
        "cnb": fitted["cnb"].predict(Xw),
        "knn": fitted["knn"].predict(Xw, repos),
        "lr": fitted["lr"].predict(X),
        "repo": fitted["repo"].predict(repos),
    }


def _blend(scores, w):
    return sum(wi * scores[n] for wi, n in zip(w, MEMBER_NAMES))


class SkillEnsembleClassifier:
    def __init__(self, folds=4, seed=42, n_jobs=-1, verbose=True,
                 mode="competition",
                 floor_p=DEFAULT_FLOOR_P, floor_r=DEFAULT_FLOOR_R,
                 floor_f1=DEFAULT_FLOOR_F1):
        """mode:
          competition - beat floor_p/r/f1 (default; high-recall operating point)
          rf          - legacy: beat RF fold-1 floors then max F1
          f1          - unconstrained micro/macro F1 blend (old behaviour)
        """
        self.folds, self.seed, self.n_jobs, self.verbose = folds, seed, n_jobs, verbose
        self.mode = mode
        self.floor_p, self.floor_r, self.floor_f1 = floor_p, floor_r, floor_f1

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def fit(self, titles, bodies, repos, Y, groups, baseline_factory=None):
        Y = np.asarray(Y).astype(bool)
        repos = np.asarray(repos, dtype=object)
        groups = np.asarray(groups)
        rng = np.random.RandomState(self.seed)

        self.feat_ = TextFeaturizer().fit(titles, bodies, repos)
        X, Xw = self.feat_.transform(titles, bodies, repos)
        self._log(f"Feature matrix: {X.shape} | mode={self.mode} "
                  f"floors P>={self.floor_p} R>={self.floor_r} F1>={self.floor_f1}")

        # ---- OOF member scores -------------------------------------------
        oof = {n: np.zeros(Y.shape) for n in MEMBER_NAMES}
        ref_rows, ref_pr = None, None
        for f, (a, b) in enumerate(GroupKFold(self.folds).split(X, groups=groups)):
            self._log(f"OOF fold {f + 1}/{self.folds}")
            fitted = _fit_members(X[a], Xw[a], Y[a], repos[a], self.n_jobs)
            sc = _score_members(fitted, X[b], Xw[b], repos[b])
            for n in MEMBER_NAMES:
                oof[n][b] = sc[n]
            if f == 0 and baseline_factory is not None:
                self._log("Measuring reference model on fold 1 ...")
                Pb = np.asarray(baseline_factory(a, b)).astype(bool)
                pb, rb, fb = prf_micro(Y[b], Pb)
                ref_rows, ref_pr = b, (pb, rb)
                self._log(f"Reference (micro) P={pb:.4f} R={rb:.4f} F1={fb:.4f}")

        self.member_report_ = {
            n: {"oof_best_threshold": best_global_threshold(Y, oof[n], COARSE_GRID)[0],
                "oof_micro_f1": best_global_threshold(Y, oof[n], COARSE_GRID)[1]}
            for n in MEMBER_NAMES}
        self._log("Member OOF micro-F1:",
                  {n: round(v["oof_micro_f1"], 4) for n, v in self.member_report_.items()})

        # ---- blend weights -----------------------------------------------
        cache = {}

        def score_w(w):
            key = tuple(np.round(w, 2))
            if key not in cache:
                S = _blend(oof, w)
                if self.mode == "competition":
                    cache[key] = _rank_blend(
                        Y, S, self.floor_p, self.floor_r, self.floor_f1)[0]
                else:
                    t, _ = best_global_threshold(Y, S, COARSE_GRID)
                    cache[key] = (objective(Y, S >= t),)
            return cache[key]

        cands = [w for w in itertools.product(np.arange(0, 1.01, 0.2), repeat=4)
                 if abs(sum(w) - 1) < 1e-6]
        best_w = max(cands, key=score_w)
        improved = True
        while improved:
            improved = False
            for i, j in itertools.permutations(range(4), 2):
                w = list(best_w)
                if w[i] < 0.1 - 1e-9:
                    continue
                w[i] -= 0.1
                w[j] += 0.1
                if score_w(w) > score_w(best_w):
                    best_w, improved = tuple(w), True
        self.weights_ = np.array(best_w, dtype=float)
        S_oof = _blend(oof, self.weights_)
        self._log("Blend weights:", dict(zip(MEMBER_NAMES, np.round(self.weights_, 2))))

        # ---- thresholds ----------------------------------------------------
        if self.mode == "competition":
            t_glob, feasible, oof_at_t = competition_threshold(
                Y, S_oof, self.floor_p, self.floor_r, self.floor_f1, grid=THR_GRID)
            self._log(f"Competition threshold {t_glob:.3f} feasible={feasible} "
                      f"OOF P={oof_at_t['precision']:.4f} R={oof_at_t['recall']:.4f} "
                      f"F1={oof_at_t['f1']:.4f}")
            # High-recall target: keep a single global threshold (per-label hurts R).
            self.use_per_label_ = False
            self.thresholds_ = np.full(Y.shape[1], t_glob, dtype=float)
            self.competition_oof_ = oof_at_t
        elif self.mode == "rf" and ref_pr is not None:
            t_glob, feasible = constrained_global_threshold(
                Y, S_oof, ref_rows, ref_pr[0], ref_pr[1])
            self._log(f"RF-constrained threshold {t_glob:.3f} (feasible: {feasible})")
            self.competition_oof_ = None
            ug = np.unique(groups)
            rng.shuffle(ug)
            half = np.isin(groups, ug[: len(ug) // 2])

            def held_out_score(P, Yb):
                sc = objective(Yb, P)
                p, r, _ = prf_micro(Yb, P)
                sc -= max(0.0, ref_pr[0] - p) + max(0.0, ref_pr[1] - r)
                return sc

            s_pl, s_gl = [], []
            for A, B in ((half, ~half), (~half, half)):
                tg, _ = best_global_threshold(Y[A], S_oof[A])
                tg2, ok = constrained_global_threshold(
                    Y[A], S_oof[A], np.arange(A.sum()), ref_pr[0], ref_pr[1])
                tg = tg2 if ok else tg
                th = tune_thresholds(S_oof[A], Y[A], tg)
                s_pl.append(held_out_score(apply_thresholds(S_oof[B], th), Y[B]))
                s_gl.append(held_out_score(
                    apply_thresholds(S_oof[B], np.full(Y.shape[1], tg)), Y[B]))
            self.use_per_label_ = bool(np.mean(s_pl) >= np.mean(s_gl))
            self._log(f"Split-half per-label {np.mean(s_pl):.4f} vs global "
                      f"{np.mean(s_gl):.4f} -> "
                      f"{'per-label' if self.use_per_label_ else 'global'}")
            self.thresholds_ = (tune_thresholds(S_oof, Y, t_glob) if self.use_per_label_
                                else np.full(Y.shape[1], t_glob))
        else:
            t_glob, _f1 = best_global_threshold(Y, S_oof)
            feasible = None
            self._log(f"Unconstrained F1 threshold {t_glob:.3f}")
            self.competition_oof_ = None
            self.use_per_label_ = False
            self.thresholds_ = np.full(Y.shape[1], t_glob, dtype=float)

        self.thresholds_[Y.sum(0) == 0] = 2.0
        self.global_threshold_ = float(t_glob)
        self.reference_ = (None if ref_pr is None
                           else {"precision": ref_pr[0], "recall": ref_pr[1]})
        self.constraint_feasible_ = feasible
        pred_oof = apply_thresholds(S_oof, self.thresholds_)
        self.oof_micro_ = dict(zip(("precision", "recall", "f1"), prf_micro(Y, pred_oof)))
        self.oof_objective_ = objective(Y, pred_oof)
        self.beats_floors_oof_ = meets_floors(
            self.oof_micro_["precision"], self.oof_micro_["recall"], self.oof_micro_["f1"],
            self.floor_p, self.floor_r, self.floor_f1)
        self._log(f"OOF micro at selected thr: { {k: round(v, 4) for k, v in self.oof_micro_.items()} } "
                  f"| beats floors: {self.beats_floors_oof_}")

        # ---- final members on all training data ---------------------------
        self._log("Fitting final members on all training data ...")
        self.members_ = _fit_members(X, Xw, Y, repos, self.n_jobs)
        return self

    def predict_scores(self, titles, bodies, repos):
        repos = np.asarray(repos, dtype=object)
        X, Xw = self.feat_.transform(titles, bodies, repos)
        return _blend(_score_members(self.members_, X, Xw, repos), self.weights_)

    def predict(self, titles, bodies, repos):
        return apply_thresholds(self.predict_scores(titles, bodies, repos),
                                self.thresholds_)

    def summary(self):
        return {
            "mode": self.mode,
            "floors": {"precision": self.floor_p, "recall": self.floor_r, "f1": self.floor_f1},
            "blend_weights": dict(zip(MEMBER_NAMES, map(float, self.weights_))),
            "global_threshold": float(self.global_threshold_),
            "per_label_thresholds_used": self.use_per_label_,
            "reference_floor_oof": self.reference_,
            "constraint_feasible": self.constraint_feasible_,
            "oof_micro": self.oof_micro_,
            "beats_floors_oof": self.beats_floors_oof_,
            "oof_objective": float(self.oof_objective_),
            "members_oof": self.member_report_,
        }
