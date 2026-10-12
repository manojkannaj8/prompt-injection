"""Read-only loaders for the saved research files.

Frames are cached per process and shared, so callers must not modify them in place
(filter or .copy() first). Row order is never changed: several files are aligned row by row.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths, rules


class MissingArtefact(FileNotFoundError):
    """A saved research file the page needs is not in the project folder."""

    def __init__(self, name):
        super().__init__(name)
        self.name = name


def _p(name):
    p = paths.path(name)
    if not p.exists():
        raise MissingArtefact(name)
    return p


@lru_cache(maxsize=None)
def runs_index():
    """One row per saved run. Row number = run_id. The raw-file path column is dropped (never shown)."""
    idx = pd.read_parquet(_p("runs_index.parquet")).drop(columns=["path"], errors="ignore")
    idx["attacked"] = idx.injection_task.notna() & idx.attack.notna() & (idx.attack != "none")
    return idx


@lru_cache(maxsize=None)
def risk_scores():
    """One row per tool output with the saved p_inj, p_obey and risk, plus the decision under the documented rule.

    `row` is the position in samples_features.parquet (the two files are aligned row by row).
    `in_gate` marks outputs up to and including the exposure point of their run, as in both notebooks.
    """
    R = pd.read_parquet(_p("risk_scores.parquet"))
    R["row"] = np.arange(len(R))
    exp_step = R[R.is_exposure == 1].set_index("run_id").step.rename("exp_step")
    Rj = R.merge(exp_step, left_on="run_id", right_index=True, how="left")
    R["in_gate"] = (Rj.exp_step.isna() | (Rj.step <= Rj.exp_step)).values
    R["decision"] = rules.decide_array(R.p_inj.values, R.risk.values)
    R["kind"] = np.where(R.is_exposure == 1, "exposure (first injected output)",
                 np.where(R.has_injection == 1, "later injected output",
                  np.where(R.attacked, "harmless output, attacked run", "harmless output, clean run")))
    return R


@lru_cache(maxsize=None)
def runs_with_gate():
    """runs_index plus `gate`: the highest risk seen up to and including the exposure point (Day 2, cell 0)."""
    idx, R = runs_index().copy(), risk_scores()
    idx["gate"] = R[R.in_gate].groupby("run_id").risk.max().reindex(idx.index).fillna(0.0)
    return idx


EXPOSURE_COLS = ["run_id", "config", "suite", "attack", "step", "tool_name", "obeyed", "fold_inj", "fold_user",
                 "fold_attack", "fold_cfg", "n_email", "n_email_new", "n_iban", "n_iban_new", "n_url", "n_url_new",
                 "n_cue_words", "out_len"]


@lru_cache(maxsize=None)
def exposure():
    """The Stage 2 rows (first injected output of each attacked run) with the saved scores attached. No text."""
    E = pd.read_parquet(_p("stage2_exposure.parquet"), columns=EXPOSURE_COLS).reset_index(drop=True)
    R = risk_scores()
    Rx = R[R.is_exposure == 1]
    if not (Rx.run_id.values == E.run_id.values).all():
        raise ValueError("risk_scores.parquet and stage2_exposure.parquet are no longer aligned")
    for c in ["p_inj", "p_obey", "risk", "decision", "row"]:
        E[c] = Rx[c].values
    return E


@lru_cache(maxsize=None)
def agent_obey_rates():
    """Measured obey rate of each agent on its exposure rows: the value the pipeline uses as P(obey) when
    no content-based Stage 2 score exists (README, 'Things Part 6 had to work out')."""
    g = exposure().groupby("config").obeyed.agg(obey_rate="mean", n="size")
    return g.sort_index()


@lru_cache(maxsize=None)
def overall_obey_rate():
    return float(exposure().obeyed.mean())


@lru_cache(maxsize=None)
def alerts():
    """All saved alerts with their reasons. `alert_id` is the row number in the saved file."""
    A = pd.read_parquet(_p("part6/alerts_with_reasons.parquet")).reset_index(drop=True)
    A.insert(0, "alert_id", ["A%05d" % i for i in range(len(A))])
    R = risk_scores()
    A = A.merge(R[["run_id", "step", "row", "attack", "user_task", "injection_task"]], on=["run_id", "step"], how="left", validate="one_to_one")
    A["why_resist"] = A.why_resist.fillna("")
    return A


@lru_cache(maxsize=None)
def alert_examples():
    return pd.read_csv(_p("part6/alerts_examples.csv")).fillna("")


@lru_cache(maxsize=None)
def notinject():
    return pd.read_parquet(_p("notinject_scores.parquet"), columns=["prompt", "category", "split", "p_inj"])


@lru_cache(maxsize=None)
def replay_all():
    return pd.read_csv(_p("replay_all_configs.csv"))


@lru_cache(maxsize=None)
def replay_loo():
    return pd.read_csv(_p("replay_leave_one_agent_out.csv"))


@lru_cache(maxsize=None)
def table(name):
    """A Part 6 result table by its file stem, e.g. table('table_stage2_results')."""
    return pd.read_csv(_p(f"part6/{name}.csv"))


@lru_cache(maxsize=None)
def summary():
    with open(_p("part6/part6_summary.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def oof_saved():
    """Stage 2 XGBoost out-of-fold scores saved in Part 3 (the reference for the replay)."""
    return np.load(_p("oof_xgb_fold_inj.npy"))


@lru_cache(maxsize=None)
def oof_refit(model):
    """Out-of-fold scores of a Part 6 refit on unseen injection tasks: 'base', 'lr' or 'xgb'.
    Returns None when part6/cache has been deleted (the README allows that)."""
    name = f"part6/cache/oof_{model}_raw_inj.npy"
    return np.load(paths.path(name)) if paths.available(name) else None


@lru_cache(maxsize=None)
def _texts():
    """Tool-output and user-task text, dictionary-encoded so repeated texts are held once."""
    import pyarrow.parquet as pq
    cols = ["tool_output", "user_task_text"]
    return pq.read_table(_p("samples_features.parquet"), columns=cols, read_dictionary=cols)


def text_for_row(row):
    """(tool_output, user_task_text) of one tool output, by its position in samples_features.parquet."""
    t = _texts()
    row = int(row)
    if not 0 <= row < t.num_rows:
        raise IndexError("row outside the saved samples")
    rec = t.slice(row, 1).to_pylist()[0]
    return rec["tool_output"] or "", rec["user_task_text"] or ""


def stage1_training_frame():
    """The three columns the final Stage 1 detector is fitted on (Day 2 cell 2, Part 6 cell 5)."""
    return pd.read_parquet(_p("samples_features.parquet"), columns=["tool_output", "tool_name", "has_injection"])
