"""Train and evaluate the round win probability model.

Honest evaluation matters more than the model here, because the data is small (a few hundred rounds):
  * Leave-one-demo-out: every score is on a match the model never saw. States from the same round
    are near-duplicates, so a random split would leak and look far better than it is.
  * Baselines: the T win rate (a constant) and a logistic regression on players alive. The model
    has to beat "count the players" to be worth anything.
  * Calibration: when it says 70%, does T win about 70% of the time? Swing ratings are differences
    between probabilities, so they're only as good as the calibration.

The shipped model is the average of two:
  * XGBoost with monotone constraints (features.MONOTONE), shallow and regularised. It can learn
    interactions (a post-plant 2v2 with no kits) but on a few hundred rounds it's coarse: some kills
    land in the same tree leaves before and after, so they'd get zero swing.
  * A logistic regression on the same standardised features, with its coefficient signs held to the
    same monotone directions. Smooth, so every kill moves it.
On the first 103 rounds the logistic model alone was as good as XGBoost, and the average beat both
while halving the kills that got no swing. With more data XGBoost should pull ahead; the CV report
shows all three so that's visible.

Explanations: XGBoost's built-in TreeSHAP (pred_contribs) and the linear model's exact contributions
(coefficient x standardised value), averaged, in log-odds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..errors import AnalystError
from .features import FEATURES, LABELS, MONOTONE, training_rows

# Shallow and strongly regularised: the best-calibrated of a handful of settings compared
# leave-one-demo-out on 5 public demos; deeper trees were overconfident at the extremes.
PARAMS = dict(
    n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_child_weight=10, reg_lambda=5.0,
    objective="binary:logistic", eval_metric="logloss", random_state=0,
)
MODEL_FILE, LINEAR_FILE, META_FILE = "winprob_xgb.json", "winprob_linear.json", "winprob_meta.json"


class ModelMissing(AnalystError):
    pass


def _xgb(**over: Any):
    from xgboost import XGBClassifier

    return XGBClassifier(**{**PARAMS, **over}, monotone_constraints=tuple(MONOTONE[f] for f in FEATURES))


class Linear:
    """Standardised logistic regression, stored as plain numbers (no pickles)."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Linear:
        """L2-regularised logistic regression whose coefficient signs obey features.MONOTONE.

        Plain logistic regression can learn a backwards sign from a few hundred rounds (it learned
        "more round time left hurts T", the time-pressure effect reversed by confounding), which would
        make the averaged model non-monotone. Same loss as scikit-learn's, solved with bounds instead.
        """
        from scipy.optimize import minimize

        mean, scale = X.mean(axis=0), X.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        Z = (X - mean) / scale
        signs = np.array([MONOTONE[f] for f in FEATURES])

        def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
            z = Z @ w[:-1] + w[-1]
            p = 1 / (1 + np.exp(-z))
            nll = np.sum(np.logaddexp(0, z) - y * z)
            g = p - y
            grad = np.append(C * (Z.T @ g) + w[:-1], C * g.sum())
            return C * nll + 0.5 * w[:-1] @ w[:-1], grad

        bounds = [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in signs] + [(None, None)]
        w = minimize(loss, np.zeros(len(FEATURES) + 1), jac=True, method="L-BFGS-B", bounds=bounds).x
        return cls(mean, scale, w[:-1], float(w[-1]))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale * self.coef

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.contributions(X).sum(axis=1) + self.intercept)))

    def to_json(self) -> dict[str, Any]:
        return {"features": FEATURES, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Linear:
        return cls(np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), d["intercept"])


class WinProbModel:
    def __init__(self, xgb, linear: Linear):
        self.xgb, self.linear = xgb, linear

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> WinProbModel:
        return cls(_xgb().fit(X, y), Linear.fit(X, y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return (self.xgb.predict_proba(X)[:, 1] + self.linear.predict_proba(X)) / 2

    def contributions(self, X: np.ndarray) -> np.ndarray:
        from xgboost import DMatrix

        tree = self.xgb.get_booster().predict(DMatrix(X, feature_names=FEATURES), pred_contribs=True)[:, :-1]
        return (tree + self.linear.contributions(X)) / 2


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    """Predicted vs actual T win rate per probability bucket."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            out.append({"bucket": f"{lo:.1f}-{hi:.1f}", "states": int(m.sum()),
                        "predicted": round(float(p[m].mean()), 3), "actual": round(float(y[m].mean()), 3)})
    return out


def cross_validate(states: pl.DataFrame) -> dict[str, Any]:
    """Leave-one-demo-out scores for the shipped model, its two halves and both baselines."""
    from sklearn.linear_model import LogisticRegression

    df = training_rows(states)
    demos = df["demo_id"].unique().sort().to_list()
    if len(demos) < 2:
        raise ValueError("Need at least 2 ingested demos to evaluate on a match the model hasn't seen.")
    X, y = df.select(FEATURES).to_numpy().astype(float), df["t_win"].to_numpy()
    groups = df["demo_id"].to_numpy()
    names = ("constant", "players_alive", "logistic", "xgboost", "model")
    preds = {n: np.zeros(len(y)) for n in names}
    alive_cols = [FEATURES.index("t_alive"), FEATURES.index("ct_alive")]
    for d in demos:
        test, train = groups == d, groups != d
        preds["constant"][test] = y[train].mean()
        lr = LogisticRegression().fit(X[train][:, alive_cols], y[train])
        preds["players_alive"][test] = lr.predict_proba(X[test][:, alive_cols])[:, 1]
        m = WinProbModel.fit(X[train], y[train])
        preds["xgboost"][test] = m.xgb.predict_proba(X[test])[:, 1]
        preds["logistic"][test] = m.linear.predict_proba(X[test])
        preds["model"][test] = m.predict_proba(X[test])
    return {
        "demos": len(demos),
        "rounds": df.select("demo_id", "round").unique().height,
        "states": len(y),
        # AUC is meaningless for a constant (it only ranks folds against each other), so it's left out.
        "scores": {n: {**_metrics(y, p), **({"auc": None} if n == "constant" else {})} for n, p in preds.items()},
        "calibration": calibration(y, preds["model"]),
        "per_demo": {d: _metrics(y[groups == d], preds["model"][groups == d]) for d in demos},
    }


def train(states: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Cross-validate, then fit the final model on everything and save it with its report."""
    report = cross_validate(states)
    df = training_rows(states)
    X, y = df.select(FEATURES).to_numpy().astype(float), df["t_win"].to_numpy()
    model = WinProbModel.fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.xgb.save_model(out_dir / MODEL_FILE)
    (out_dir / LINEAR_FILE).write_text(json.dumps(model.linear.to_json(), indent=2))
    contribs = model.contributions(X)
    importance = {f: round(float(np.abs(contribs[:, i]).mean()), 4) for i, f in enumerate(FEATURES)}
    meta = {
        **report,
        "features": FEATURES,
        "monotone": MONOTONE,
        "params": PARAMS,
        "linear_coefficients": dict(zip(FEATURES, [round(float(c), 3) for c in model.linear.coef])),
        "importance_mean_abs_contribution": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
        "trained_on": sorted(df["demo_id"].unique().to_list()),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return meta


def load(out_dir: Path) -> tuple[WinProbModel, dict[str, Any]]:
    from xgboost import XGBClassifier

    if not (out_dir / MODEL_FILE).exists() or not (out_dir / LINEAR_FILE).exists():
        raise ModelMissing("No win probability model yet. Train one with: cs2-agent model train")
    xgb = XGBClassifier()
    xgb.load_model(out_dir / MODEL_FILE)
    linear = Linear.from_json(json.loads((out_dir / LINEAR_FILE).read_text()))
    meta = json.loads((out_dir / META_FILE).read_text()) if (out_dir / META_FILE).exists() else {}
    return WinProbModel(xgb, linear), meta


def predict(model: WinProbModel, states: pl.DataFrame) -> np.ndarray:
    """P(T wins) for every state. Decided states (a side wiped out, a defuse) are pinned to 0 or 1."""
    p = model.predict_proba(states.select(FEATURES).to_numpy().astype(float))
    t_alive, ct_alive = states["t_alive"].to_numpy(), states["ct_alive"].to_numpy()
    planted = states["bomb_planted"].to_numpy().astype(bool)
    defused = (states["event"] == "defuse").to_numpy()
    p = np.where(ct_alive == 0, 1.0, p)                       # CTs wiped: T win
    p = np.where((t_alive == 0) & ~planted, 0.0, p)           # Ts wiped, no bomb: CT win
    return np.where(defused, 0.0, p)


def explain(model: WinProbModel, X: np.ndarray) -> np.ndarray:
    return model.contributions(X)


def top_reasons(contribs: np.ndarray, row: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
    """The k features moving this estimate most, each as e.g. "low T equipment -> CT".

    A feature's direction is fixed (MONOTONE), so whether it pushes toward T or CT says whether its value
    is high or low compared with a typical state: low T equipment can only ever push toward CT.
    """
    order = np.argsort(-np.abs(contribs))[:k]
    out = []
    for i in order:
        f, c = FEATURES[i], float(contribs[i])
        sign = MONOTONE[f] or 1
        level = "high" if c * sign > 0 else "low"
        if f == "bomb_planted":
            label = "bomb planted" if row[i] else "bomb not planted"
        else:
            label = f"{level} {LABELS[f]}"
        out.append({"feature": LABELS[f], "level": level, "value": round(float(row[i]), 1),
                    "pushes": "T" if c > 0 else "CT", "reason": f"{label} -> {'T' if c > 0 else 'CT'}",
                    "log_odds": round(c, 3)})
    return out
