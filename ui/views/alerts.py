"""D. Alerts explorer: the saved alerts, filtered and paged on the server so the browser only receives one page."""
import math

import streamlit as st

import components as ui
from content.facts import facts
from core import data, rules
from views import record

SORTS = {"Highest risk first": ("risk", False), "Lowest risk first": ("risk", True), "Highest injection probability first": ("p_inj", False),
         "Highest obedience probability first": ("p_obey", False), "Alert ID": ("alert_id", True)}


def filter_alerts(A, decisions=None, agents=None, suites=None, outcomes=None, risk_range=(0.0, 1.0), query=""):
    """Pure filter used by the page and by the tests."""
    m = (A.risk >= risk_range[0]) & (A.risk <= risk_range[1])
    if decisions:
        m &= A.decision.isin(decisions)
    if agents:
        m &= A.config.isin(agents)
    if suites:
        m &= A.suite.isin(suites)
    if outcomes:
        m &= A.outcome.isin(outcomes)
    q = (query or "").strip()
    if q:
        hay = A.snippet.str.cat([A.tool_name, A.alert_id, A.why_injection, A.why_obey], sep=" ")
        m &= hay.str.contains(q, case=False, regex=False)
    return A[m]


def render():
    f = facts()
    A = data.alerts()
    ui.header("Alerts", "Alerts explorer",
              f"Every alert the guard would have raised on the saved runs: {f.alerts} in total. Filter the list, then select a row to see the full "
              "text, the scores and the reasons. These are historical records, not live alerts.",
              f"part6/alerts_with_reasons.parquet: in-gate tool outputs with decision ≠ SAFE ({f.gate_high} HIGH RISK, {f.gate_susp} SUSPICIOUS). "
              "Filtering, sorting and paging run server-side; one page is sent to the browser.")
    st.markdown(ui.provenance("historical"), unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
        decisions = c1.multiselect("Decision", [rules.HIGH, rules.SUSPICIOUS], key="al_dec", placeholder="Both")
        agents = c2.multiselect("Agent configuration", sorted(A.config.unique()), key="al_agent", placeholder="All agents")
        outcomes = c3.multiselect("What really happened", sorted(A.outcome.unique()), key="al_out", placeholder="Any outcome")
        c4, c5, c6 = st.columns([1.2, 1.6, 1.2])
        suites = c4.multiselect("Task suite", sorted(A.suite.unique()), key="al_suite", placeholder="All suites")
        risk_range = c5.slider("Risk score range", 0.0, 1.0, (0.0, 1.0), 0.01, key="al_risk")
        query = c6.text_input("Search text, tool or reasons", key="al_q", max_chars=80, placeholder="e.g. send_money, delete, A00042")
        c7, c8, c9 = st.columns([1.2, 1.6, 1.2])
        sort = c7.selectbox("Sort by", list(SORTS), key="al_sort")
        size = c9.selectbox("Rows per page", [25, 50, 100], key="al_size")
        with c8:
            st.write("")
            ui.mask_toggle()

    F = filter_alerts(A, decisions, agents, suites, outcomes, risk_range, query)
    col, asc = SORTS[sort]
    F = F.sort_values(col, ascending=asc, kind="stable")
    n = len(F)
    if n == 0:
        st.info("No alerts match these filters. Widen the risk range or clear a filter.")
        ui.explain("alerts_table")
        return
    pages = max(1, math.ceil(n / size))
    sig = (tuple(decisions), tuple(agents), tuple(suites), tuple(outcomes), risk_range, query, sort, size)
    if st.session_state.get("al_sig") != sig:                      # filters changed: go back to the first page
        st.session_state["al_sig"], st.session_state["al_page"] = sig, 1
    c1, c2 = st.columns([3, 1])
    page = c2.number_input(f"Page (1 to {pages:,})", 1, pages, key="al_page", step=1)
    lo = (page - 1) * size
    P = F.iloc[lo:lo + size]
    n_high = int((F.decision == rules.HIGH).sum())
    c1.markdown(f"**{n:,}** matching alerts of {len(A):,} · {n_high:,} HIGH RISK · {n - n_high:,} SUSPICIOUS · showing {lo + 1:,} to {lo + len(P):,}")

    masked = st.session_state.get("mask_text", True)
    view = P[["alert_id", "decision", "risk", "p_inj", "p_obey", "config", "suite", "tool_name", "outcome", "snippet"]].copy()
    view["decision"] = view.decision.map(lambda d: f"{ui.DECISION_ICON[d]} {d}")
    if masked:
        view["snippet"] = view.snippet.map(ui.mask_sensitive)
    sel = st.dataframe(
        view, hide_index=True, width="stretch", height=min(38 * len(view) + 40, 640), on_select="rerun", selection_mode="single-row", key=f"al_table_{page}",
        column_config={
            "alert_id": st.column_config.TextColumn("Alert ID", width="small"),
            "decision": st.column_config.TextColumn("Decision", width=130),
            "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f", width="small"),
            "p_inj": st.column_config.NumberColumn("P(injection)", format="%.2f", width="small"),
            "p_obey": st.column_config.NumberColumn("P(obey)", format="%.2f", width="small"),
            "config": st.column_config.TextColumn("Agent"), "suite": st.column_config.TextColumn("Suite", width="small"),
            "tool_name": st.column_config.TextColumn("Tool"), "outcome": st.column_config.TextColumn("What really happened"),
            "snippet": st.column_config.TextColumn("Text (snippet)", width="large")})
    ui.source("part6/alerts_with_reasons.parquet")
    ui.explain("alerts_table")

    rows = sel.selection.rows if sel and sel.selection else []
    if not rows:
        st.info("Select a row (click the box at its left edge) to open the alert's full details and reasons.")
        return
    a = P.iloc[rows[0]]
    st.divider()
    st.subheader(f"Alert {a.alert_id}")
    st.caption(a.alert)
    record.render(record.from_alert(a), key="al_rec")
