"""Shared look and building blocks of the dashboard. Rendering only: no model code lives here."""
import html
import logging
import re
import traceback
from functools import wraps

import plotly.graph_objects as go
import streamlit as st

from core import data, paths, rules, stage1

log = logging.getLogger("dashboard")

# ---- palette (validated for the dark chart surface with the dataviz palette checker) ----
PAGE, SURFACE, SURFACE2 = "#0d1117", "#151a22", "#1b222c"
INK, INK2, MUTED = "#e6ebf2", "#b3bcc9", "#8892a0"
GRID, AXIS = "#232a35", "#39424f"
BLUE, ORANGE, AQUA, GRAY = "#3987e5", "#d95926", "#199e70", "#8a929e"     # series: identity
CRITICAL, WARNING, GOOD = "#d03b3b", "#fab219", "#0ca30c"                  # status: reserved for decisions
DECISION_COLOR = {rules.HIGH: CRITICAL, rules.SUSPICIOUS: WARNING, rules.SAFE: GOOD}
DECISION_ICON = {rules.HIGH: "⛔", rules.SUSPICIOUS: "⚠️", rules.SAFE: "✅"}
MODEL_COLOR = {"Model-only lookup": GRAY, "Logistic Regression": AQUA, "XGBoost": BLUE,
               "XGBoost (saved Part 3 scores)": BLUE}

CSS = f"""
<style>
:root {{ --page:{PAGE}; --surface:{SURFACE}; --surface2:{SURFACE2}; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED};
        --line:rgba(255,255,255,.09); --blue:{BLUE}; --crit:{CRITICAL}; --warn:{WARNING}; --good:{GOOD}; }}
html, body, [class*="css"] {{ font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }}
/* top padding must clear Streamlit's fixed top bar (3.75rem, taller on a host: Share / GitHub icons) */
.block-container {{ padding-top: 4.75rem; padding-bottom: 4rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; }}
h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
h2 {{ font-size: 1.3rem !important; font-weight: 600 !important; margin-top: .6rem !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
.pi-eyebrow {{ font-size:.76rem; line-height:1.5; letter-spacing:.14em; text-transform:uppercase; color:var(--blue); font-weight:600; margin-bottom:.15rem; }}
.pi-lead {{ color:var(--ink2); font-size:1.02rem; line-height:1.6; max-width:70ch; margin:.2rem 0 1.1rem 0; }}
.pi-card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 18px; height:100%; }}
.pi-card .lab {{ color:var(--muted); font-size:clamp(.62rem, .85vw, .74rem); letter-spacing:.06em; text-transform:uppercase; font-weight:600; hyphens:auto; }}
/* numbers shrink on narrow screens instead of breaking in the middle */
.pi-card .val {{ color:var(--ink); font-size:clamp(1.1rem, 2.1vw, 1.85rem); font-weight:650; line-height:1.15; margin:.3rem 0 .15rem 0; letter-spacing:-.02em; word-break:normal; overflow-wrap:normal; }}
.pi-card .val small {{ font-size:1rem; font-weight:500; color:var(--ink2); }}
.pi-card .sub {{ color:var(--ink2); font-size:.84rem; line-height:1.45; }}
.pi-card.crit {{ border-top:3px solid var(--crit); }} .pi-card.warn {{ border-top:3px solid var(--warn); }}
.pi-card.good {{ border-top:3px solid var(--good); }} .pi-card.blue {{ border-top:3px solid var(--blue); }}
.pi-chip {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:650; letter-spacing:.05em;
           border:1px solid var(--line); white-space:nowrap; }}
.pi-chip.crit {{ background:rgba(208,59,59,.16); color:#ff8f8f; border-color:rgba(208,59,59,.5); }}
.pi-chip.warn {{ background:rgba(250,178,25,.13); color:#fcc552; border-color:rgba(250,178,25,.45); }}
.pi-chip.good {{ background:rgba(12,163,12,.14); color:#5fd65f; border-color:rgba(12,163,12,.5); }}
.pi-chip.blue {{ background:rgba(57,135,229,.14); color:#8fbcf5; border-color:rgba(57,135,229,.45); }}
.pi-chip.gray {{ background:rgba(255,255,255,.05); color:var(--ink2); }}
.pi-verdict {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:18px 22px; }}
.pi-verdict .big {{ font-size:1.7rem; font-weight:680; letter-spacing:-.01em; }}
.pi-verdict.crit {{ border-left:5px solid var(--crit); }} .pi-verdict.warn {{ border-left:5px solid var(--warn); }}
.pi-verdict.good {{ border-left:5px solid var(--good); }}
.pi-eq {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; background:var(--surface2); border:1px solid var(--line);
         border-radius:10px; padding:12px 16px; color:var(--ink); font-size:1.02rem; overflow-x:auto; }}
.pi-note {{ background:rgba(57,135,229,.08); border:1px solid rgba(57,135,229,.3); border-radius:10px; padding:10px 14px;
           color:var(--ink2); font-size:.9rem; line-height:1.5; }}
.pi-note.warn {{ background:rgba(250,178,25,.07); border-color:rgba(250,178,25,.35); }}
.pi-step {{ display:flex; gap:12px; align-items:flex-start; margin:.5rem 0; }}
.pi-step .n {{ flex:0 0 28px; height:28px; border-radius:50%; background:rgba(57,135,229,.18); color:#8fbcf5; font-weight:650;
              display:flex; align-items:center; justify-content:center; font-size:.85rem; }}
.pi-step .t {{ color:var(--ink2); line-height:1.55; }} .pi-step .t b {{ color:var(--ink); }}
.pi-q {{ color:var(--ink); font-weight:600; margin:.55rem 0 .1rem 0; font-size:.92rem; }}
.pi-a {{ color:var(--ink2); line-height:1.55; font-size:.92rem; margin-bottom:.2rem; }}
.pi-term {{ border-bottom:1px dotted var(--muted); cursor:help; }}
.pi-src {{ color:var(--muted); font-size:.78rem; margin-top:.2rem; }}
.pi-flow {{ display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; }}
.pi-flow .box {{ flex:1 1 150px; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.pi-flow .box b {{ display:block; color:var(--ink); font-size:.9rem; }} .pi-flow .box span {{ color:var(--ink2); font-size:.8rem; line-height:1.4; }}
.pi-flow .arrow {{ align-self:center; color:var(--muted); }}
div[data-testid="stExpander"] details {{ border-radius:10px; border-color:var(--line); background:rgba(255,255,255,.015); }}
div[data-testid="stExpander"] summary p {{ font-size:.9rem; color:var(--ink2); }}
@media (max-width: 760px) {{ .pi-card .val {{ font-size:1.45rem; }} .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x), quote=True)


# ---------------------------------------------------------------- explanation level
def level():
    return st.session_state.get("level", "Beginner")


def is_beginner():
    return level() == "Beginner"


def pick(beginner, technical):
    return beginner if is_beginner() else technical


def term(word, definition=None):
    """A glossary term with its plain-English definition as a tooltip (used the first time a term appears)."""
    if definition is None:
        from content.glossary import lookup
        definition = lookup(word)
    return f'<span class="pi-term" title="{esc(definition)}">{esc(word)}</span>'


# ---------------------------------------------------------------- page furniture
def header(eyebrow, title, beginner, technical):
    st.markdown(f'<div class="pi-eyebrow">{esc(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="pi-lead">{pick(beginner, technical)}</div>', unsafe_allow_html=True)


def card(label, value, sub="", tone=""):
    """A metric tile. `value` and `sub` may contain trusted inline HTML written in this code base."""
    return (f'<div class="pi-card {tone}"><div class="lab">{esc(label)}</div><div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def cards(items):
    """items: list of (label, value, sub, tone). Rendered as one responsive row."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(card(*it), unsafe_allow_html=True)


def chip(text, tone="gray"):
    return f'<span class="pi-chip {tone}">{esc(text)}</span>'


def decision_chip(decision):
    tone = {rules.HIGH: "crit", rules.SUSPICIOUS: "warn", rules.SAFE: "good"}[decision]
    return f'<span class="pi-chip {tone}">{DECISION_ICON[decision]} {esc(decision)}</span>'


PROVENANCE = {
    "live": ("LIVE MODEL PREDICTION", "blue", "Computed just now by the Stage 1 detector on the text you entered."),
    "historical": ("HISTORICAL · SAVED RESULT", "gray", "Loaded from the saved research files. Nothing was recomputed."),
    "illustration": ("ILLUSTRATION · NOT A PREDICTION", "warn", "Made-up numbers to show how the calculation works."),
    "saved": ("SAVED PART 3–5 RESULT", "gray", "Produced by the earlier notebooks and stored as a file."),
    "refit": ("PART 6 REFIT", "blue", "Produced by models refitted in the Part 6 notebook on the same grouped folds."),
    "whatif": ("WHAT-IF · RECOMPUTED FROM SAVED SCORES", "warn", "Recalculated from saved scores with the notebook's replay rule."),
}


def provenance(kind):
    label, tone, tip = PROVENANCE[kind]
    return f'<span class="pi-chip {tone}" title="{esc(tip)}">{label}</span>'


def note(text, warn=False):
    st.markdown(f'<div class="pi-note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def source(*names):
    st.markdown('<div class="pi-src">Data source: ' + " · ".join(esc(n) for n in names) + "</div>", unsafe_allow_html=True)


def steps(items):
    out = "".join(f'<div class="pi-step"><div class="n">{i}</div><div class="t">{t}</div></div>' for i, t in enumerate(items, 1))
    st.markdown(out, unsafe_allow_html=True)


QUESTIONS = ["What am I looking at?", "Why does it matter?", "How does it work?", "How should I interpret it?",
             "What does my result tell me?", "What are its limitations?"]


def explain(key, expanded=False, label="What does this mean?"):
    """The on-demand six-question explanation of one dashboard component, at the selected level."""
    from content.explain import get
    e = get(key)
    answers = e["B"] if is_beginner() else e["T"]
    with st.expander(f"{label}  ·  {e['title']}", expanded=expanded, icon=":material/help:"):
        st.markdown("".join(f'<div class="pi-q">{q}</div><div class="pi-a">{a}</div>' for q, a in zip(QUESTIONS, answers)),
                    unsafe_allow_html=True)
        st.caption(f"{level()} explanation. Change the level in the sidebar.")


def chart_title(technical, plain, beginner_note, technical_note):
    """Two-level chart heading: the original technical title, and a plain-English reading of the same chart."""
    if is_beginner():
        st.markdown(f"### {plain}")
        st.caption(f"Technical title: {technical}")
        st.markdown(f'<div class="pi-a">{beginner_note}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"### {technical}")
        st.markdown(f'<div class="pi-a">{technical_note}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- charts
def style(fig, height=380, legend=True, **layout):
    base = dict(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=12.5, family="Inter, Segoe UI, system-ui, sans-serif"),
        margin=dict(l=10, r=16, t=28, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#0b0f14", bordercolor=AXIS, font=dict(color=INK, size=12.5)), bargap=0.28)
    base.update(layout)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False, "responsive": True})


def bar(x, y, color=BLUE, text=None, hover=None, name=None, orientation="v"):
    return go.Bar(x=x, y=y, name=name, orientation=orientation, text=text, textposition="outside", cliponaxis=False,
                  textfont=dict(color=INK2, size=11.5), hovertemplate=hover,
                  marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4))


def png(name, caption=None):
    """Show a figure saved by the notebooks, or say plainly that it is missing."""
    if not paths.available(name):
        st.info(f"The saved figure `{name}` is not in the project folder, so it cannot be shown.")
        return
    st.image(str(paths.path(name)), caption=caption or f"Original notebook figure: {name}", width="stretch")


def original_figure(name, label="Show the original figure saved by the notebook"):
    with st.expander(label, icon=":material/image:"):
        png(name)


# ---------------------------------------------------------------- text from the benchmark
_MASKS = [(re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
          (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"), "[account number]"),
          (re.compile(r"(?:https?://|www\.)[^\s'\"<>]+", re.I), "[link]"),
          (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card-like number]")]


def mask_sensitive(text):
    """Hide things that look like emails, account numbers and links. The benchmark data is synthetic,
    but the dashboard still avoids putting contact-like strings on a projector by default."""
    for rx, rep in _MASKS:
        text = rx.sub(rep, text)
    return text


def show_text(text, limit=1800, key=None):
    """Render benchmark or user text as inert plain text (never as markdown or HTML)."""
    masked = st.session_state.get("mask_text", True)
    t = mask_sensitive(text) if masked else text
    if len(t) > limit:
        st.code(t[:limit] + "\n… [truncated for display]", language=None, wrap_lines=True)
        with st.expander(f"Show all {len(t):,} characters"):
            st.code(t, language=None, wrap_lines=True)
    else:
        st.code(t, language=None, wrap_lines=True)


def mask_toggle():
    st.toggle("Mask emails, account numbers and links in displayed text", key="mask_text",
              value=st.session_state.get("mask_text", True),
              help="The benchmark text is synthetic (AgentDojo), so these are not real people or accounts. "
                   "Masking only changes what is displayed; scores always use the original text.")


# ---------------------------------------------------------------- risk breakdown (shared by live and saved results)
def risk_breakdown(p_inj, p_obey, kind, agent=None, p_obey_note=None, t_high=rules.T_HIGH, key="rb"):
    """Guided walk from the two probabilities to the decision. `kind` is a PROVENANCE key."""
    r = rules.risk(p_inj, p_obey)
    d = rules.decide(p_inj, r, t_high=t_high)
    tone = {rules.HIGH: "crit", rules.SUSPICIOUS: "warn", rules.SAFE: "good"}[d]
    st.markdown(
        f'<div class="pi-verdict {tone}">{provenance(kind)}'
        f'<div class="big" style="margin-top:.5rem">{DECISION_ICON[d]} {d}</div>'
        f'<div class="pi-a">Recommended handling: {esc(rules.ACTION[d])}.</div></div>', unsafe_allow_html=True)
    st.write("")
    cards([("Injection probability", f"{p_inj:.1%}", pick("How likely the text contains a planted instruction (Stage 1).",
                                                         "Stage 1 Logistic Regression, P(injection)."), "blue"),
           ("Obedience probability", f"{p_obey:.1%}", p_obey_note or pick("How likely this agent would follow it (Stage 2).",
                                                                        "Stage 2 P(obey), isotonic-calibrated."), "blue"),
           ("Risk score", f"{r:.3f}", f"threshold {t_high:.2f}", tone)])
    st.write("")
    fig = go.Figure()
    for lab, v, c in [("Risk score", r, DECISION_COLOR[d]), ("Obedience probability", p_obey, BLUE), ("Injection probability", p_inj, BLUE)]:
        fig.add_trace(go.Bar(y=[lab], x=[v], orientation="h", marker=dict(color=c, cornerradius=4), text=[f"{v:.3f}"], textposition="outside",
                             cliponaxis=False, textfont=dict(color=INK2), hovertemplate=f"{lab}: %{{x:.3f}}<extra></extra>", showlegend=False))
    # each threshold is drawn only on the row it applies to (rows are categories 0, 1, 2 from the bottom)
    for x, row, colr, lab in [(t_high, 0, CRITICAL, f"HIGH RISK line: risk ≥ {t_high:.2f}"),
                              (rules.P_INJ_LOG, 2, WARNING, f"SUSPICIOUS line: P(injection) ≥ {rules.P_INJ_LOG}")]:
        fig.add_shape(type="line", x0=x, x1=x, y0=row - 0.46, y1=row + 0.46, line=dict(color=colr, width=3))
        fig.add_annotation(x=x, y=row + 0.46, text=lab, showarrow=False, xanchor="left", yanchor="bottom", xshift=4,
                           font=dict(color=INK2, size=11), bgcolor="rgba(13,17,23,0.82)")
    style(fig, height=270, legend=False, bargap=0.5, xaxis=dict(range=[0, 1.08], title="probability / score (0 to 1)"))
    show(fig, key=f"{key}_bars")

    st.markdown(f'<div class="pi-eq">risk = P(injection) × P(obey) = {p_inj:.3f} × {p_obey:.3f} = <b>{r:.3f}</b></div>', unsafe_allow_html=True)
    cmp1 = "≥" if r >= t_high else "<"
    cmp2 = "≥" if p_inj >= rules.P_INJ_LOG else "<"
    rule_steps = [f"<b>Is the risk at least {t_high:.2f}?</b> {r:.3f} {cmp1} {t_high:.2f} → {'yes, so the decision is <b>HIGH RISK</b>.' if r >= t_high else 'no, so it is not HIGH RISK.'}"]
    if r < t_high:
        rule_steps.append(f"<b>Is an injection likely (P(injection) at least {rules.P_INJ_LOG})?</b> {p_inj:.3f} {cmp2} {rules.P_INJ_LOG} → "
                          + ("yes, so the decision is <b>SUSPICIOUS</b>: something was planted, but this agent is unlikely to act on it."
                             if p_inj >= rules.P_INJ_LOG else "no, so the decision is <b>SAFE</b>."))
    steps(rule_steps)
    if agent is not None and p_obey < t_high:
        note(f"With an obedience probability of {p_obey:.3f}, the risk can never reach {t_high:.2f} whatever the text says, because "
             f"risk cannot be larger than P(obey). The strongest possible outcome here is SUSPICIOUS.", warn=True)
    return d


# ---------------------------------------------------------------- errors and resources
def safe_page(fn):
    """Show a short, path-free message instead of a traceback when a page fails."""
    @wraps(fn)
    def wrapper():
        try:
            fn()
        except data.MissingArtefact as ex:
            st.error(f"This page needs the saved file `{ex.name}`, which is not in the project folder. "
                     "Restore it (or re-run the notebook that writes it) and reload.")
        except Exception as ex:  # noqa: BLE001
            log.error("page %s failed\n%s", fn.__name__, traceback.format_exc())
            st.error(f"This page could not be displayed ({type(ex).__name__}). The details are in the terminal "
                     "where the dashboard was started. The other pages are unaffected.")
    return wrapper


@st.cache_resource(show_spinner="Preparing the Stage 1 detector. The first run refits it from the saved data and takes about a minute…")
def get_stage1():
    return stage1.load()
