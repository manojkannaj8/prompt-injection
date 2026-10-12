"""Numbers shown on the dashboard, each derived from a saved file and tagged with its denominator."""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data, replay, rules

GPT4O = "gpt-4o-2024-05-13"


@lru_cache(maxsize=None)
def overview():
    idx, R, E = data.runs_index(), data.risk_scores(), data.exposure()
    s = {}
    # data volume (runs_index.parquet, risk_scores.parquet, stage2_exposure.parquet)
    s["n_runs"], s["n_configs"] = int(len(idx)), int(idx.config.nunique())
    s["n_attacked_runs"], s["n_clean_runs"] = int(idx.attacked.sum()), int((~idx.attacked).sum())
    s["n_suites"] = int(idx.suite.nunique())
    s["n_attack_styles"] = int(idx.loc[idx.attacked, "attack"].nunique())
    s["n_tool_outputs"] = int(len(R))
    s["n_injected_outputs"] = int(R.has_injection.sum())
    s["n_harmless_outputs"] = int((R.has_injection == 0).sum())
    s["n_exposure"] = int(len(E))
    s["n_obeyed"] = int(E.obeyed.sum())
    s["obey_rate"] = float(E.obeyed.mean())
    s["n_agents_stage2"] = int(E.config.nunique())
    # decisions under the documented rule
    order = rules.ORDER
    s["decisions_exposure"] = {d: int((E.decision == d).sum()) for d in order}
    G = R[R.in_gate]
    s["decisions_gate"] = {d: int((G.decision == d).sum()) for d in order}
    s["n_gate"] = int(len(G))
    s["decisions_all"] = {d: int((R.decision == d).sum()) for d in order}
    lev = E.groupby("decision").obeyed.agg(n="size", obeyed="sum", obey_rate="mean")
    s["levels"] = {d: {"n": int(lev.n[d]), "obeyed": int(lev.obeyed[d]), "obey_rate": float(lev.obey_rate[d]),
                       "share_of_obeyed": float(lev.obeyed[d] / E.obeyed.sum())} for d in order}
    s["n_alerts"] = int(len(data.alerts()))
    # replay at the documented threshold (replay_all_configs.csv)
    rp = data.replay_all()
    s["n_replay_agents"] = int(len(rp))
    s["asr_pooled_before"] = replay.pooled(rp, "ASR no guard", "n_attacked")
    s["asr_pooled_after"] = replay.pooled(rp, "ASR guard", "n_attacked")
    s["task_pooled_before"] = replay.pooled(rp, "task no guard", "n_clean")
    s["task_pooled_after"] = replay.pooled(rp, "task guard", "n_clean")
    s["task_change_median"] = float(rp["task change (pts)"].median())
    s["n_agents_lose_5"] = int((rp["task change (pts)"] < -5).sum())
    g = rp.set_index("config").loc[GPT4O]
    s["gpt4o"] = {"asr_before": float(g["ASR no guard"]), "asr_after": float(g["ASR guard"]),
                  "task_before": float(g["task no guard"]), "task_after": float(g["task guard"]),
                  "n_attacked": int(g.n_attacked), "n_clean": int(g.n_clean)}
    loo = data.replay_loo()
    s["loo_asr_before"] = replay.pooled(loo, "ASR no guard", "n_att")
    s["loo_asr_after"] = replay.pooled(loo, "ASR guard", "n_att")
    s["loo_worst_task_change"] = float(loo["task change (pts)"].min())
    s["loo_threshold"] = float(loo.threshold.iloc[0])
    return s


@lru_cache(maxsize=None)
def stage2_table():
    """table_stage2_results.csv with a `source` column saying whether a row is a saved Part 3 result or a Part 6 refit."""
    t = data.table("table_stage2_results").copy()
    t["source"] = np.where(t.model.str.contains("saved"), "Saved Part 3 scores", "Part 6 refit")
    return t


def pr_auc(model, protocol="Unseen injection task", text="raw", metric="PR-AUC"):
    t = stage2_table()
    r = t[(t.model == model) & (t.protocol == protocol) & (t.text == text)]
    return float(r[metric].iloc[0]) if len(r) else float("nan")


def threshold_metrics(t):
    """Guard at risk >= t against 'the agent obeyed', on the exposure rows (same definition as Part 6 cell 8f)."""
    E = data.exposure()
    y, pred = E.obeyed.values == 1, E.risk.values >= t
    tp, fp, fn, tn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum()), int((~y & ~pred).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false positive rate": fp / max(fp + tn, 1), "blocked": tp + fp}


def _thin(a, b, n=300):
    k = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return a[k], b[k]


@lru_cache(maxsize=None)
def curves():
    """PR and ROC curves on unseen injection tasks, from the saved out-of-fold scores (thinned for plotting)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    y = data.exposure().obeyed.values
    out = []
    for name, s, src in [("Model-only lookup", data.oof_refit("base"), "Part 6 refit"),
                         ("Logistic Regression", data.oof_refit("lr"), "Part 6 refit"),
                         ("XGBoost (saved Part 3 scores)", data.oof_saved(), "Saved Part 3 scores")]:
        if s is None:
            continue
        pr, rc, _ = precision_recall_curve(y, s)
        fp, tp, _ = roc_curve(y, s)
        rc_t, pr_t = _thin(rc[::-1], pr[::-1])
        fp_t, tp_t = _thin(fp, tp)
        out.append({"model": name, "source": src, "recall": rc_t, "precision": pr_t, "fpr": fp_t, "tpr": tp_t,
                    "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))})
    return out


def ece(p, t, bins=10):
    """Expected calibration error with 10 equal-width bins (Part 6 cell 8e)."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == i).mean() * abs(p[b == i].mean() - t[b == i].mean()) for i in range(bins) if (b == i).any()))


@lru_cache(maxsize=None)
def calibration():
    """Reliability points (10 equal-size bins) for the raw saved XGBoost score and the calibrated P(obey)."""
    E = data.exposure()
    y = E.obeyed.values
    out = []
    for name, s in [("XGBoost raw score (saved)", data.oof_saved()), ("Calibrated P(obey)", E.p_obey.values)]:
        q = pd.qcut(s, 10, duplicates="drop")
        d = pd.DataFrame({"s": s, "y": y}).groupby(q, observed=True).agg(predicted=("s", "mean"), observed=("y", "mean"), n=("y", "size"))
        out.append({"name": name, "points": d.reset_index(drop=True), "ece": ece(s, y)})
    return out
