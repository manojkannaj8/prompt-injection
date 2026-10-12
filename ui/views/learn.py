"""Learning centre: a seven-lesson path, an interactive risk-score explainer and a searchable glossary."""
import streamlit as st

import components as ui
from content import glossary
from content.lessons import lessons
from views import charts, widgets

TAGS = {"measured": ("MEASURED FINDING", "good"), "interpretation": ("INTERPRETATION", "blue"),
        "hypothesis": ("FUTURE HYPOTHESIS · NOT TESTED", "warn")}


def _widget(name, key):
    if name == "real_injection":
        widgets.real_injection()
    elif name == "attack_per_agent":
        ui.show(charts.attack_per_agent(), key=key)
        st.caption("Attack success of the benchmark's main attack against each undefended model. Source: part6/table_attack_success_per_agent.csv.")
    elif name == "pipeline":
        widgets.pipeline()
    elif name == "risk_calc":
        widgets.risk_calc(key)
    elif name == "decision_rule":
        widgets.decision_rule()
    elif name == "tfidf_demo":
        widgets.tfidf_demo(key)
    elif name == "leak_demo":
        widgets.leak_demo()
    elif name == "confusion":
        widgets.confusion_guard()
    elif name == "threshold_lab":
        widgets.threshold_lab(key)
    elif name == "calibration":
        ui.show(charts.calibration(), key=key)
        st.caption("Points on the diagonal are perfectly calibrated. Grey: raw model score. Blue: after correction.")
    elif name == "protocols":
        ui.show(charts.protocols("PR-AUC"), key=key)
        st.caption("Stage 2 PR-AUC under each testing rule. The leftmost group (random split) is the unfair comparison; the others hide a whole group.")
    elif name == "shap_top":
        ui.show(charts.shap_top("raw", 14), key=key)
        st.caption("Average influence of each input on the obedience score. Orange bars are words from the benchmark's fixed attack sentence.")
    elif name == "family":
        ui.show(charts.family(), key=key)
    elif name == "tradeoff":
        ui.show(charts.tradeoff_gpt4o(), key=key)
        st.caption("Each point is one way of protecting GPT-4o. Top-left is best. Replay of saved runs, not a live test.")


def _goto(i):
    st.session_state["lesson"] = st.session_state["lesson_pick"] = i


def _picked():
    st.session_state["lesson"] = st.session_state["lesson_pick"]


def _toggle_done(i):
    done = st.session_state["lessons_done"]
    (done.discard if i in done else done.add)(i)


def _lessons():
    L = lessons()
    st.session_state.setdefault("lesson", 0)
    st.session_state.setdefault("lessons_done", set())
    done = st.session_state["lessons_done"]
    i = st.session_state["lesson"]
    st.session_state["lesson_pick"] = i

    st.progress(len(done) / len(L), text=f"{len(done)} of {len(L)} lessons marked as understood")
    labels = [f"{'✓ ' if k in done else ''}Lesson {k + 1}: {l['title']}" for k, l in enumerate(L)]
    st.selectbox("Lesson", range(len(L)), format_func=lambda k: labels[k], key="lesson_pick", on_change=_picked)

    les = L[i]
    st.markdown(f'<div class="pi-eyebrow">Lesson {i + 1} of {len(L)} · about {les["minutes"]} minutes</div>', unsafe_allow_html=True)
    st.header(les["title"])
    st.markdown(f'<div class="pi-lead"><b>After this lesson you will:</b> {les["goal"][0].lower() + les["goal"][1:]}</div>', unsafe_allow_html=True)

    for j, (heading, body, tech, widget, tag) in enumerate(les["sections"]):
        with st.container(border=True):
            head = f"#### {heading}"
            st.markdown(head)
            if tag:
                st.markdown(ui.chip(*TAGS[tag]), unsafe_allow_html=True)
            st.markdown(body)
            if widget:
                _widget(widget, f"ls_{i}_{j}")
            if tech and not ui.is_beginner():
                st.caption(f"Technical note: {tech}")

    st.write("")
    c1, c2, c3 = st.columns([1, 1.4, 1])
    c1.button("← Previous lesson", disabled=i == 0, width="stretch", key="ls_prev", on_click=_goto, args=(i - 1,))
    is_done = i in done
    c2.button("✓ Understood" if is_done else "Mark this lesson as understood", type="secondary" if is_done else "primary", width="stretch",
              key="ls_done", on_click=_toggle_done, args=(i,))
    c3.button("Next lesson →", disabled=i == len(L) - 1, width="stretch", key="ls_next", on_click=_goto, args=(i + 1,))
    if len(done) == len(L):
        st.success("You have worked through all seven lessons. You should now be able to say what the project does, why each stage exists, how to "
                   "read its main graphs, why it can be wrong, what it has shown, and what is still unbuilt. The Guided demo is a good next step.")


def _glossary():
    q = st.text_input("Search the glossary", key="gl_q", max_chars=60, placeholder="Try: SHAP, threshold, leakage, embedding…")
    hits = glossary.search(q)
    st.caption(f"{len(hits)} of {len(glossary.entries())} terms")
    if not hits:
        st.info("No term matches that search. Try a shorter word.")
        return
    for e in hits:
        with st.expander(e["term"]):
            st.markdown(f'<div class="pi-q">In simple English</div><div class="pi-a">{ui.esc(e["simple"])}</div>'
                        f'<div class="pi-q">Why this project uses it</div><div class="pi-a">{ui.esc(e["why"])}</div>'
                        f'<div class="pi-q">Example from this project</div><div class="pi-a">{ui.esc(e["example"])}</div>'
                        + (f'<div class="pi-q">Deeper technical explanation</div><div class="pi-a">{ui.esc(e["deeper"])}</div>' if e["deeper"] else ""),
                        unsafe_allow_html=True)


def render():
    ui.header("Learning centre", "Understand the system",
              "New to machine learning? Start here. Seven short lessons take you from \"what is an AI agent?\" to reading every graph on this "
              "dashboard, using this project's own numbers as the examples. No programming or statistics needed.",
              "Seven lessons, an interactive risk-score explainer and a glossary. All figures are the project's measured results; each section is "
              "tagged as a measured finding, an interpretation or an untested hypothesis where that distinction matters.")
    t1, t2, t3 = st.tabs(["Learning path", "Risk-score explainer", "Glossary"])
    with t1:
        _lessons()
    with t2:
        st.subheader("How two probabilities become one decision")
        widgets.risk_calc("lc_calc")
    with t3:
        _glossary()
