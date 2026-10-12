"""B. Interactive risk analysis: live Stage 1 scoring of new text, and a separate explorer of saved results."""
import pandas as pd
import streamlit as st

import components as ui
from content.facts import facts
from core import data, rules, stage1
from views import charts, record, widgets

EXAMPLES = ["Benchmark attack (saved AgentDojo tool output)", "Harmless tool output (saved AgentDojo, clean run)",
            "Harmless question with a trigger word (NotInject)"]


def _example_text(name):
    if name == EXAMPLES[0]:
        r = data.alert_examples().iloc[0]
        A = data.alerts()
        row = A[(A.run_id == r.run_id) & (A.step == r.step)].row.iloc[0]
        return data.text_for_row(row)[0]
    if name == EXAMPLES[1]:
        R = data.risk_scores()
        clean = R[(R.kind == "harmless output, clean run") & (R.tool_name == "get_day_calendar_events")]
        row = (clean if len(clean) else R[R.kind == "harmless output, clean run"]).row.iloc[0]
        return data.text_for_row(row)[0]
    N = data.notinject()
    return N.sort_values("p_inj", ascending=False).prompt.iloc[0]


def _load_example():
    st.session_state["live_text"] = _example_text(st.session_state["live_example"])[: stage1.MAX_CHARS]
    st.session_state.pop("live_result", None)


def _live_tab():
    f = facts()
    ui.note("<b>What is live here, and what is not.</b> The injection probability is a real prediction made now by the Stage 1 detector. "
            "The obedience probability is <b>not</b> predicted from your text: Stage 2 needs a sentence-embedding step and feature code from the "
            "early notebooks that are not part of the saved project. For new text the dashboard uses the selected agent's measured obey rate from "
            "the saved runs, which is the same fallback the research pipeline uses wherever no Stage 2 score exists.")
    st.write("")
    c1, c2 = st.columns([2, 1])
    with c2:
        st.selectbox("Start from an example", EXAMPLES, key="live_example", help="Loads real text from the saved data into the box.")
        st.button("Load example into the box", on_click=_load_example, width="stretch")
        agents = stage1.agent_choices()
        rates = data.agent_obey_rates().obey_rate
        agent = st.selectbox("Agent configuration", agents, key="live_agent",
                             format_func=lambda a: f"{a} · obeys {data.overall_obey_rate():.0%}" if a == stage1.AVERAGE_AGENT else f"{a} · obeys {rates[a]:.0%}",
                             help="Sets the obedience probability: the share of planted instructions this agent followed in the saved runs.")
        m = ui.get_stage1()
        tool = st.selectbox("Tool that returned the text", [stage1.NO_TOOL] + m.tools, key="live_tool",
                            help="Stage 1 has a small learned adjustment per tool. Leave as unknown for text that did not come from a benchmark tool.")
    with c1:
        text = st.text_area("Tool-output text to analyse", key="live_text", height=265, max_chars=stage1.MAX_CHARS,
                            placeholder="Paste the text an AI agent received from a tool: an email body, a file's contents, a web page, a transaction list…")
        go = st.button("Analyse", type="primary")
    st.caption("The user's original task is not an input here. Only Stage 2 uses it (through the embedding and the novelty flags), and Stage 2 is not "
               "available for new text. Saved examples on the next tab show the user task and a full Stage 2 score.")

    if go:
        try:
            st.session_state["live_result"] = stage1.live_risk(text, agent, tool)
        except stage1.InputError as ex:
            st.session_state.pop("live_result", None)
            st.warning(str(ex))
    res = st.session_state.get("live_result")
    if not res:
        st.info("Enter some text and press **Analyse**. Nothing is stored or sent anywhere; scoring happens on this computer.")
        return

    st.divider()
    st.subheader("Result")
    ui.risk_breakdown(res["p_inj"], res["p_obey"], "live", agent=res["agent"], key="live",
                      p_obey_note=f"Measured obey rate of {res['agent']} in the saved runs. Not a prediction for this text.")
    for w in res["warnings"]:
        st.warning(w)

    st.markdown("#### What was analysed")
    c1, c2 = st.columns([2, 1])
    with c1:
        ui.show_text(res["text"], limit=1200)
    with c2:
        st.dataframe([{"Field": "Agent configuration", "Value": res["agent"]}, {"Field": "Tool", "Value": res["tool_name"]},
                      {"Field": "Words read", "Value": f"{res['n_tokens']:,}"},
                      {"Field": "Words the detector knows", "Value": f"{res['n_known_tokens']:,}"},
                      {"Field": "Vocabulary terms matched", "Value": f"{res['n_terms']:,}"}], hide_index=True, width="stretch")

    st.markdown("#### Why Stage 1 gave this injection score")
    if len(res["top_up"]) or len(res["top_down"]):
        ui.chart_title("Stage 1 term contributions (coefficient × TF-IDF, log-odds)", "The words that moved the injection score",
                       "Orange words pushed the score towards 'this contains a planted instruction'. Blue words pushed it towards 'this is ordinary tool data'.",
                       "Exact additive decomposition of the Logistic Regression logit. Top 8 in each direction.")
        ui.show(charts.word_contributions(res["top_up"], res["top_down"]), key="live_words")
        words_total = res["margin"] - res["intercept"] - res["tool_term"]
        st.markdown(f'<div class="pi-eq">score before squashing = all words {words_total:+.2f} + tool {res["tool_term"]:+.2f} + starting point {res["intercept"]:+.2f} '
                    f'= {res["margin"]:+.2f} → P(injection) = {res["p_inj"]:.3f}</div>', unsafe_allow_html=True)
        st.caption("The starting point is strongly negative because most tool outputs in the training data are harmless.")
    else:
        st.info("No vocabulary terms matched, so there are no word contributions to show.")
    ui.explain("live_result")
    ui.explain("stage1_words")
    ui.note("<b>Limits of this live result.</b> There is no per-input SHAP explanation for new text: SHAP was computed for the saved Stage 2 rows only. "
            "The detector was trained on benchmark tool outputs, where injections are English sentences inside structured data, so ordinary prose "
            "can score high. If the text is a saved AgentDojo output, the live score is from a model that has already seen it; the honest out-of-fold "
            "score is on the <i>Saved examples</i> tab.", warn=True)
    with st.expander("How the live Stage 1 detector was verified", icon=":material/verified:"):
        chk = m.meta.get("check", {})
        st.markdown(
            f"- Refitted with the recipe of `ML project Day2.ipynb` cell 2 on {m.meta.get('n_training_rows', 0):,} saved tool outputs "
            f"({m.meta.get('n_distinct_texts', 0):,} distinct texts, {m.meta.get('n_features', 0):,} features). The research files are not modified.\n"
            f"- Largest difference from the saved coefficient table: `{chk.get('coef_max_abs_diff')}`\n"
            f"- Largest difference from the {chk.get('notinject_n')} saved NotInject scores: `{chk.get('notinject_max_abs_diff')}`\n"
            f"- Check passed: **{chk.get('passed')}**")


def _saved_tab():
    f = facts()
    st.markdown(ui.provenance("historical"), unsafe_allow_html=True)
    st.caption("Real tool outputs from the saved runs with the scores the research notebooks stored. Both stages are available here, "
               "and the outcome of each run is known.")
    mode = st.radio("Choose a saved example", ["The ten worked examples from the Part 6 notebook", "Look up any saved run"], horizontal=True, key="sv_mode")
    if mode.startswith("The ten"):
        ex = data.alert_examples()
        labels = [f"{r.case} · {r.config} · risk {r.risk:.2f}" for r in ex.itertuples()]
        i = st.selectbox("Worked example", range(len(ex)), format_func=lambda k: labels[k], key="sv_ex")
        A = data.alerts()
        hit = A[(A.run_id == ex.run_id[i]) & (A.step == ex.step[i])]
        if hit.empty:
            st.warning("This worked example is not in the saved alerts file.")
            return
        st.divider()
        record.render(record.from_alert(hit.iloc[0]), key="sv_rec")
    else:
        R = data.risk_scores()
        n_runs = int(data.runs_index().shape[0])
        run_id = st.number_input(f"Run number (0 to {n_runs - 1:,})", 0, n_runs - 1, 18416, 1, key="sv_run")
        rows = R[R.run_id == run_id]
        if rows.empty:
            st.info("This run produced no tool outputs, so there is nothing to score. Try another run number.")
            return
        run = data.runs_index().iloc[int(run_id)]
        st.caption(f"Agent {run.config} · suite {run.suite} · {'attacked run (' + str(run.attack) + ')' if run.attacked else 'clean run'} · "
                   f"task {'completed' if run.utility else 'not completed'}")
        view = rows[["step", "tool_name", "kind", "p_inj", "p_obey", "risk", "decision"]].rename(columns={
            "step": "Step", "tool_name": "Tool", "kind": "Kind of output", "p_inj": "P(injection)", "p_obey": "P(obey)", "risk": "Risk", "decision": "Decision"})
        st.dataframe(view.round(3), hide_index=True, width="stretch")
        step = st.selectbox("Open a step", list(rows.step), key="sv_step")
        st.divider()
        record.render(record.from_risk_row(rows[rows.step == step].iloc[0]), key="sv_rec2")


def render():
    f = facts()
    ui.header("Risk analysis", "Interactive risk analysis",
              "Paste text that an AI agent might receive from a tool and see how the guard would judge it. The first tab runs the real detector "
              "on your text. The second tab shows real examples from the saved experiments, where both steps of the guard are available.",
              f"Live: refitted final Stage 1 Logistic Regression × agent obey rate, decision by the documented rule (risk ≥ {f.t_high}; "
              f"P(injection) ≥ {f.p_log}). Historical: out-of-fold p_inj and calibrated p_obey from risk_scores.parquet with saved SHAP reasons.")
    ui.mask_toggle()
    t1, t2, t3 = st.tabs(["Live analysis", "Saved examples (historical)", "How the risk score works"])
    with t1:
        _live_tab()
    with t2:
        _saved_tab()
    with t3:
        widgets.risk_calc("an_calc")
