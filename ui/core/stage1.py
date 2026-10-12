"""Live Stage 1 scoring: does this tool output contain an injection?

The trained detector was never saved by the research notebooks, so it is refitted here once with the
recipe of `ML project Day2.ipynb` cell 2 ("Final Stage 1 detector, trained on all AgentDojo tool outputs",
the same fit as LR1["raw"] in Part 6 cell 5) and kept in ui/cache. Nothing is written to the research files.

The refit is checked against two saved artefacts before it is used:
  - part6/table_lr_coefficients_stage1_raw.csv  (every coefficient)
  - notinject_scores.parquet                    (the 339 saved P(injection) values)

Stage 2 (will the agent obey?) cannot be run on new text: it needs MiniLM sentence embeddings and the
flag-extraction code of Parts 1-4, and neither is in this folder. See `live_risk`.
"""
import json
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, paths, rules

RECIPE = "day2-cell2-v1"
MAX_CHARS = 40_000                       # the longest saved tool output is 32,458 characters
NO_TOOL = "(no tool / unknown)"          # scored with an all-zero tool block, as the NotInject check did
AVERAGE_AGENT = "Average agent (overall obey rate)"
_MODEL_FILE, _META_FILE = paths.CACHE / "stage1_detector.joblib", paths.CACHE / "stage1_detector.json"
_lock = threading.Lock()
_model = None


class InputError(ValueError):
    """The text or option supplied by the user cannot be scored."""


@dataclass
class Stage1Model:
    tf: object
    oh: object
    clf: object
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self):
        return len(self.tf.vocabulary_)

    @property
    def tools(self):
        return list(self.oh.categories_[0])


def _fingerprint():
    import sklearn
    return {"recipe": RECIPE, "sklearn": sklearn.__version__,
            "source_bytes": paths.path("samples_features.parquet").stat().st_size}


def _fit():
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    S = data.stage1_training_frame()
    codes, uniq = pd.factorize(S.tool_output)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True, dtype=np.float32).fit(uniq)
    oh = OneHotEncoder(handle_unknown="ignore").fit(S[["tool_name"]].fillna("none"))
    X = hstack([tf.transform(uniq)[codes], oh.transform(S[["tool_name"]].fillna("none"))]).tocsr()
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, solver="liblinear").fit(X, S.has_injection.values)
    return Stage1Model(tf, oh, clf, {"n_training_rows": int(len(S)), "n_distinct_texts": int(len(uniq)),
                                     "n_features": int(X.shape[1])})


def self_check(m):
    """Compare the refit with what the notebooks saved. Returns the largest differences found."""
    from scipy.sparse import hstack
    out = {}
    names = ["word:" + w for w in m.tf.get_feature_names_out()] + ["tool=" + t for t in m.tools]
    try:
        saved = data.table("table_lr_coefficients_stage1_raw").set_index("feature").coef
        mine = pd.Series(m.clf.coef_[0], index=names)
        out["same_features"] = bool(set(saved.index) == set(mine.index))
        out["coef_max_abs_diff"] = float((mine - saved.reindex(mine.index)).abs().max()) if out["same_features"] else None
    except data.MissingArtefact:
        out["same_features"], out["coef_max_abs_diff"] = None, None
    try:
        N = data.notinject()
        Xn = hstack([m.tf.transform(N.prompt), m.oh.transform(pd.DataFrame({"tool_name": ["none"] * len(N)}))]).tocsr()
        out["notinject_max_abs_diff"] = float(np.abs(m.clf.predict_proba(Xn)[:, 1] - N.p_inj.values).max())
        out["notinject_n"] = int(len(N))
    except data.MissingArtefact:
        out["notinject_max_abs_diff"], out["notinject_n"] = None, 0
    out["passed"] = bool((out["coef_max_abs_diff"] is None or out["coef_max_abs_diff"] < 1e-6)
                         and (out["notinject_max_abs_diff"] is None or out["notinject_max_abs_diff"] < 1e-4)
                         and out["same_features"] is not False)
    return out


def is_cached():
    if not (_MODEL_FILE.exists() and _META_FILE.exists()):
        return False
    try:
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
        return all(meta.get(k) == v for k, v in _fingerprint().items())
    except Exception:
        return False


def load(force_refit=False):
    """The Stage 1 detector: from ui/cache when it matches this sklearn version and data file, else refitted
    (about a minute on a laptop CPU)."""
    global _model
    import joblib
    with _lock:
        if _model is not None and not force_refit:
            return _model
        if not force_refit and is_cached():
            try:
                d = joblib.load(_MODEL_FILE)
                _model = Stage1Model(d["tf"], d["oh"], d["clf"], json.loads(_META_FILE.read_text(encoding="utf-8")))
                return _model
            except Exception:
                pass                                              # unreadable cache: fall through and refit
        m = _fit()
        m.meta.update(_fingerprint())
        m.meta["check"] = self_check(m)
        paths.CACHE.mkdir(parents=True, exist_ok=True)
        joblib.dump({"tf": m.tf, "oh": m.oh, "clf": m.clf}, _MODEL_FILE, compress=3)
        _META_FILE.write_text(json.dumps(m.meta, indent=1), encoding="utf-8")
        _model = m
        return m


def clean_text(text):
    """Validate text typed or pasted into the UI. It is only ever tokenised, never executed or rendered as markup."""
    if not isinstance(text, str):
        raise InputError("The tool output must be text.")
    text = text.replace("\x00", "").strip()
    if not text:
        raise InputError("Enter or paste some tool-output text to analyse.")
    if len(text) > MAX_CHARS:
        raise InputError(f"The text is {len(text):,} characters long. The limit is {MAX_CHARS:,}.")
    return text


def score(text, tool_name=NO_TOOL, top=8):
    """P(injection) for one text, with the words that pushed the score up or down.

    A word's contribution is coefficient x TF-IDF value in log-odds, the same quantity Part 6 cell 6
    reports as `why_injection`. The contributions, the tool term and the intercept add up to the score.
    """
    m = load()
    text = clean_text(text)
    if tool_name != NO_TOOL and tool_name not in m.tools:
        raise InputError("Unknown tool name.")
    x = m.tf.transform([text]).tocsr()
    nW = x.shape[1]
    coef, intercept = m.clf.coef_[0], float(m.clf.intercept_[0])
    contrib = x.multiply(coef[:nW]).tocsr()
    names = m.tf.get_feature_names_out()
    words = pd.DataFrame({"term": names[contrib.indices], "contribution": contrib.data,
                          "tfidf": x.data, "coefficient": coef[contrib.indices]})
    tool_term = float(coef[nW + m.tools.index(tool_name)]) if tool_name != NO_TOOL else 0.0
    margin = float(words.contribution.sum()) + tool_term + intercept
    p = float(1.0 / (1.0 + np.exp(-margin)))

    tokens = [g for g in m.tf.build_analyzer()(text) if " " not in g]
    known = sum(t in m.tf.vocabulary_ for t in tokens)
    warnings = []
    if len(words) == 0:
        warnings.append("None of the words in this text are in the detector's vocabulary, so the score comes only from "
                        "the tool and the model's starting point. Treat it as uninformative.")
    elif tokens and known / len(tokens) < 0.5:
        warnings.append(f"Only {known} of {len(tokens)} words are in the detector's vocabulary. The text looks unlike the "
                        "AgentDojo tool outputs the detector was trained on, so the score is less reliable.")
    if len(tokens) < 5:
        warnings.append("Very short text: there is little evidence for the detector to work with.")
    return {"p_inj": p, "margin": margin, "intercept": intercept, "tool_term": tool_term, "text": text,
            "tool_name": tool_name, "n_tokens": len(tokens), "n_known_tokens": int(known), "n_terms": int(len(words)),
            "top_up": words[words.contribution > 0].nlargest(top, "contribution").reset_index(drop=True),
            "top_down": words[words.contribution < 0].nsmallest(top, "contribution").reset_index(drop=True),
            "warnings": warnings}


def agent_choices():
    """Agents that can be selected for a live analysis: the 28 with a measured obey rate, plus the average."""
    return [AVERAGE_AGENT] + list(data.agent_obey_rates().index)


def p_obey_for(agent):
    """P(obey) used for new text: the agent's measured obey rate on its saved exposure rows."""
    if agent == AVERAGE_AGENT:
        return data.overall_obey_rate()
    rates = data.agent_obey_rates()
    if agent not in rates.index:
        raise InputError("Unknown agent configuration.")
    return float(rates.obey_rate[agent])


def live_risk(text, agent=AVERAGE_AGENT, tool_name=NO_TOOL):
    """Live risk for new text = live Stage 1 score x the agent's measured obey rate, then the documented rule.

    This is the fallback the pipeline itself uses wherever no content-based Stage 2 score exists
    (non-exposure rows of risk_scores.parquet, and the NotInject check of Day 2).
    """
    s1 = score(text, tool_name)
    p_obey = p_obey_for(agent)
    r = rules.risk(s1["p_inj"], p_obey)
    return {**s1, "agent": agent, "p_obey": p_obey, "p_obey_source": "agent's measured obey rate (not a content-based Stage 2 prediction)",
            "risk": r, "decision": rules.decide(s1["p_inj"], r), "mode": "live"}
