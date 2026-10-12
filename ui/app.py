"""Prompt-injection defence dashboard.

Start it from the project folder with:   python run_dashboard.py
or from this folder with:                streamlit run app.py

The app reads the saved research files and writes only to ui/cache. See ui/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Prompt-Injection Defence Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

import components as ui  # noqa: E402
from core import paths  # noqa: E402
from views import alerts, analyze, evaluation, explainability, learn, method, overview, tour, tradeoff  # noqa: E402

ui.inject_css()

PAGES = {
    "overview": st.Page(ui.safe_page(overview.render), title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    "analyze": st.Page(ui.safe_page(analyze.render), title="Risk analysis", icon=":material/policy:", url_path="risk-analysis"),
    "alerts": st.Page(ui.safe_page(alerts.render), title="Alerts explorer", icon=":material/notifications_active:", url_path="alerts"),
    "explain": st.Page(ui.safe_page(explainability.render), title="Explainability", icon=":material/psychology:", url_path="explainability"),
    "evaluation": st.Page(ui.safe_page(evaluation.render), title="Model evaluation", icon=":material/monitoring:", url_path="evaluation"),
    "tradeoff": st.Page(ui.safe_page(tradeoff.render), title="Security vs usefulness", icon=":material/balance:", url_path="security-vs-usefulness"),
    "method": st.Page(ui.safe_page(method.render), title="Methodology & limits", icon=":material/account_tree:", url_path="methodology"),
    "learn": st.Page(ui.safe_page(learn.render), title="Understand the system", icon=":material/school:", url_path="learn"),
    "tour": st.Page(ui.safe_page(tour.render), title="Guided demo", icon=":material/slideshow:", url_path="guided-demo"),
}
st.session_state["_pages"] = PAGES

page = st.navigation({
    "Dashboard": [PAGES["overview"], PAGES["analyze"], PAGES["alerts"]],
    "Research results": [PAGES["explain"], PAGES["evaluation"], PAGES["tradeoff"], PAGES["method"]],
    "Learn": [PAGES["learn"], PAGES["tour"]],
})

with st.sidebar:
    st.markdown('<div class="pi-eyebrow" style="margin-top:.4rem">Explanation level</div>', unsafe_allow_html=True)
    st.radio("Explanation level", ["Beginner", "Technical"], key="level", horizontal=True, label_visibility="collapsed",
             help="Beginner: plain English, terms explained. Technical: formulas, model names and methodology. Applies to every page.")
    st.caption("Explain like I'm new to ML" if ui.is_beginner() else "Formulas, metrics and methodology")
    st.divider()
    st.markdown('<div class="pi-a" style="font-size:.8rem"><b>Research prototype.</b> Results are replays of saved AgentDojo runs, not a live defence. '
                'Encoder, Integrated Gradients and external-detector comparison are not built.</div>', unsafe_allow_html=True)
    missing = paths.missing()
    if missing:
        st.warning(f"{len(missing)} saved file(s) are missing. Pages that need them will say so.")
        with st.expander("Missing files"):
            st.write("\n".join(f"- {m}" for m in missing))

page.run()
