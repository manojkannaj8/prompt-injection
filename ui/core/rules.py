"""The decision rule, copied from the research notebooks and not changed here.

T_HIGH    = 0.20  cell 1 of `ML project Day3 - Part 6.ipynb`: the value that reproduces replay_all_configs.csv.
P_INJ_LOG = 0.5   cell 6 of the same notebook: the SUSPICIOUS rule.

The dashboard never edits these. The "what if" sliders in the UI pass their own value to `decide`
and are labelled as illustrations.
"""
import math

T_HIGH = 0.20
P_INJ_LOG = 0.5
HIGH, SUSPICIOUS, SAFE = "HIGH RISK", "SUSPICIOUS", "SAFE"
ORDER = [HIGH, SUSPICIOUS, SAFE]
ACTION = {HIGH: "block the tool output, or ask the user before continuing",
          SUSPICIOUS: "let it through and log it",
          SAFE: "let it through"}


def _check(p, name):
    if p is None or isinstance(p, bool) or not isinstance(p, (int, float)) or math.isnan(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a probability between 0 and 1")
    return float(p)


def risk(p_inj, p_obey):
    """Risk score = P(injection) x P(obey)."""
    return _check(p_inj, "P(injection)") * _check(p_obey, "P(obey)")


def decide(p_inj, risk_score, t_high=T_HIGH, p_inj_log=P_INJ_LOG):
    """HIGH RISK if risk >= t_high; else SUSPICIOUS if P(injection) >= p_inj_log; else SAFE."""
    _check(p_inj, "P(injection)"); _check(risk_score, "risk")
    if risk_score >= t_high:
        return HIGH
    return SUSPICIOUS if p_inj >= p_inj_log else SAFE


def decide_array(p_inj, risk_score, t_high=T_HIGH, p_inj_log=P_INJ_LOG):
    """Vectorised `decide`, written the way cell 6 of the Part 6 notebook writes it."""
    import numpy as np
    return np.where(risk_score >= t_high, HIGH, np.where(p_inj >= p_inj_log, SUSPICIOUS, SAFE))
