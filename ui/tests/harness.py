"""Renders one dashboard page for the headless tests: session_state['page'] picks it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import components as ui  # noqa: E402
from views import alerts, analyze, evaluation, explainability, learn, method, overview, tour, tradeoff  # noqa: E402

PAGES = {"overview": overview, "analyze": analyze, "alerts": alerts, "explain": explainability, "evaluation": evaluation,
         "tradeoff": tradeoff, "method": method, "learn": learn, "tour": tour}
st.session_state.setdefault("level", "Beginner")
ui.inject_css()
ui.safe_page(PAGES[st.session_state.get("page", "overview")].render)()
