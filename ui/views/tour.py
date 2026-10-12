"""Guided demonstration: an eight-step walkthrough for presenting the project to someone new to ML.

Every number and example is a saved project artefact. Step 3 onwards follows one real saved alert.
"""
import streamlit as st

import components as ui
from content.facts import facts
from core import data
from views import charts, method, widgets

STEPS = ["The prompt-injection problem", "The defence architecture", "A real saved example", "The risk calculation",
         "The final alert and its explanation", "Model evaluation results", "Security versus usefulness", "Limitations and unfinished work"]


def _example():
    """The second worked example of the Part 6 notebook: a correct block on GPT-4 (saved, historical)."""
    ex = data.alert_examples().iloc[1]
    A = data.alerts()
    return A[(A.run_id == ex.run_id) & (A.step == ex.step)].iloc[0], ex.case


def _say(beginner, technical):
    st.markdown(f'<div class="pi-note"><b>In one breath:</b> {ui.pick(beginner, technical)}</div>', unsafe_allow_html=True)
    st.write("")


def _step(i, f):
    if i == 0:
        _say("AI agents act on what they read. If an attacker hides an instruction in an email or a file, the agent may follow it instead of its user.",
             "Indirect prompt injection: instructions embedded in tool output are treated as user intent by the agent.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("An **AI agent** uses tools to get things done: it reads your inbox, looks up transactions, sends messages. Whatever a tool "
                        "returns goes straight into the agent's working text. An attacker who can put words there, by sending an email or editing a "
                        "shared file, can try to give the agent new orders. That is **prompt injection**.")
            widgets.real_injection()
        with c2:
            ui.cards([("Most easily fooled model", f.asr_max, "of attacks succeeded, no defence", "crit")])
            st.write("")
            ui.cards([("Most resistant model", f.asr_min, "same attack, same tasks", "good")])
            st.caption(f"Main benchmark attack against {f.asr_n} undefended models. Source: part6/table_attack_success_per_agent.csv.")
    elif i == 1:
        _say("The guard asks two questions about every piece of text: is an instruction planted here, and would this agent obey it? It multiplies the two answers into a risk score.",
             "Stage 1 P(injection) × calibrated Stage 2 P(obey) = risk; three-way decision at fixed thresholds.")
        widgets.pipeline()
        st.write("")
        c1, c2 = st.columns(2)
        c1.markdown(ui.card("Stage 1", "Is an instruction planted here?", f"Looks at the words of the tool output. Catches {f.s1_recall} of planted instructions on unseen tasks.", "blue"), unsafe_allow_html=True)
        c2.markdown(ui.card("Stage 2", "Will this agent obey it?", f"Looks at the text, the user's task and which agent is reading. Only {f.obey_pct} of planted instructions were obeyed.", "blue"), unsafe_allow_html=True)
        st.write("")
        widgets.decision_rule()
    elif i == 2:
        a, case = _example()
        _say("Here is one real moment from the recorded runs. The user asked for something ordinary, and the tool's answer came back with an instruction hidden inside.",
             "One exposure row from the saved runs, scored out-of-fold. Nothing on this step is recomputed.")
        st.markdown(ui.provenance("historical") + " " + ui.chip(f"run {int(a.run_id)} · step {int(a.step)}") + " " + ui.chip(a.config), unsafe_allow_html=True)
        tool_text, task_text = data.text_for_row(a.row)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What the user asked the agent to do**")
            ui.show_text(task_text, limit=600)
            st.markdown(f"**The agent then called the tool** `{a.tool_name}` ({a.suite} tasks)")
        with c2:
            st.markdown("**What came back from the tool**")
            ui.show_text(tool_text, limit=1100)
        st.caption("Synthetic benchmark text (AgentDojo). Displayed as inert text. In the saved run, this attack succeeded.")
    elif i == 3:
        a, _ = _example()
        _say(f"Stage 1 is almost certain an instruction is planted. Stage 2 says this agent is very likely to obey. Multiplied, the risk is far above the {f.t_high} line.",
             f"Saved out-of-fold p_inj and isotonic-calibrated p_obey; risk ≥ {f.t_high} → HIGH RISK.")
        ui.risk_breakdown(float(a.p_inj), float(a.p_obey), "historical", key="tour_rb",
                          p_obey_note="Stage 2 XGBoost, calibrated, scored by a model that had not seen this attack goal (saved).")
        ui.note("The risk score is a ranking tool. It is not a guarantee that this attack would succeed that often.", warn=True)
    elif i == 4:
        a, case = _example()
        _say("The guard does not just say 'stop'. It gives its reasons: which words looked like an instruction, and why this agent was expected to obey.",
             "Stage 1 reasons: coefficient × TF-IDF. Stage 2 reasons: out-of-fold TreeSHAP of the Part 6 refit (top 3 up, top 2 down).")
        st.markdown(ui.decision_chip(a.decision) + " " + ui.chip(a.outcome) + " " + ui.chip(f"alert {a.alert_id}"), unsafe_allow_html=True)
        st.markdown(f'<div class="pi-eq">{ui.esc(a.alert)}</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Why it looks like an injection**")
            ui.show(charts.reasons(charts.parse_reasons(a.why_injection.replace('"', "")), "push on the injection score (log-odds)"), key="tour_r1")
        with c2:
            st.markdown("**Why this agent may obey, and what lowers the risk**")
            ui.show(charts.reasons(charts.parse_reasons(a.why_obey) + charts.parse_reasons(a.why_resist), "push on the obedience score (SHAP, log-odds)"), key="tour_r2")
        st.caption(f"Notebook's label for this case: “{case}”. The strongest reason is the agent itself: this model obeys more often than any other in the data. "
                   "Reasons describe the model's arithmetic, not the cause of the agent's behaviour.")
    elif i == 5:
        _say(f"Tested fairly, on attack goals it had never seen, the obedience model scores {f.xgb_inj} where guessing scores {f.obey3}. Tested unfairly it looks much better, which is why the fair test matters.",
             f"Stage 2 PR-AUC: lookup {f.base_inj}, LR {f.lr_inj}, XGBoost {f.xgb_inj} on unseen injection tasks; random split {f.xgb_rnd} (leaky).")
        ui.cards([("Guessing", f.obey3, "share of planted instructions that were obeyed", ""), ("Agent identity only", f.base_inj, "lookup baseline", ""),
                  ("Logistic Regression", f.lr_inj, "reads the content", "blue"), ("XGBoost", f.xgb_inj, "reads the content and its meaning", "blue")])
        st.write("")
        ui.show(charts.protocols("PR-AUC"), key="tour_proto")
        st.caption("Stage 2 PR-AUC under five testing rules (Part 6 refits). The leftmost group is the unfair random split, shown for comparison only. "
                   "On a new attack style the two models are level and clearly weaker.")
    elif i == 6:
        _say(f"On GPT-4o the guard would have cut successful attacks from {f.g4_asr_before} to {f.g4_asr_after}, while normal tasks went from {f.g4_task_before} to {f.g4_task_after}. Other defences either stop less or break normal work.",
             f"Replay at risk ≥ {f.t_high}. Pooled over {f.replay_agents} agents: ASR {f.asr_before} → {f.asr_after}; {f.lose5} agents lose more than 5 utility points.")
        ui.show(charts.tradeoff_gpt4o(), key="tour_trade")
        ui.cards([("All agents, attacks", f"{f.asr_before} <small>→</small> {f.asr_after}", f"pooled over {f.replay_agents} agents", "blue"),
                  ("Never-seen agents, attacks", f"{f.asr_before} <small>→</small> {f.loo_after}", "guard trained without the agent", "blue"),
                  ("Agents that lose > 5 points of tasks", f.lose5, "at the single shared threshold", "crit")])
        st.write("")
        ui.note("This is a replay of recorded runs. It is an estimate, not a demonstration of a live, production-ready guarantee.", warn=True)
    else:
        _say("The idea works on this benchmark. The models also lean on shortcuts, the tests have gaps, and three planned parts are not built. That is where the project honestly stands.",
             "Wrapper shortcut, [WRAP] placeholder leak, agent-identity dependence, replay-only evaluation; encoder, Integrated Gradients and external-detector comparison not started.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What the project found out about its own weaknesses**")
            ui.steps([f"The model partly memorised the benchmark's fixed attack sentence ({f.wrap_raw} of word influence).",
                      f"Hiding that sentence did not fix it: the placeholder became the clue ({f.wrap_masked}).",
                      f"{f.sh_agent} of the obedience score is just 'which agent is this'.",
                      "The hand-built 'new bank account / new email / new link' flags do almost nothing.",
                      "Everything is a replay of saved runs from one benchmark. No one attacked the guard."])
        with c2:
            st.markdown("**What has not been built**")
            for name, status, desc in method.STATUS:
                if status in ("Not started", "Not done"):
                    st.markdown(f'{ui.chip(status.upper(), "crit")} &nbsp;<b>{ui.esc(name)}</b><div class="pi-a">{ui.esc(desc)}</div>', unsafe_allow_html=True)
        st.write("")
        pages = st.session_state.get("_pages", {})
        st.markdown("**Where to go next**")
        cols = st.columns(4)
        for c, (k, lab) in zip(cols, [("analyze", "Try your own text"), ("alerts", "Browse the alerts"), ("learn", "Take the lessons"), ("method", "Read the methodology")]):
            if k in pages:
                c.page_link(pages[k], label=lab, icon=":material/arrow_forward:")


def _go(step):
    st.session_state["tour_step"] = max(0, min(len(STEPS) - 1, step))


def render():
    f = facts()
    ui.header("Guided demo", "A guided walk through the project",
              "Eight short steps, designed for showing the project to someone who has never studied machine learning. Use Next and Previous. "
              "Everything shown is a real saved result; nothing is invented for the demo.",
              "Eight-step presentation path over the saved artefacts. Steps 3 to 5 follow one saved exposure row (out-of-fold scores, saved SHAP reasons).")
    st.session_state.setdefault("tour_step", 0)
    i = st.session_state["tour_step"]
    n = len(STEPS)

    st.progress((i + 1) / n, text=f"Step {i + 1} of {n}")
    dots = " ".join(ui.chip(f"{k + 1}", "blue" if k == i else "gray") for k in range(n))
    st.markdown(dots, unsafe_allow_html=True)
    st.header(f"{i + 1}. {STEPS[i]}")
    _step(i, f)

    st.divider()
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.button("← Previous", disabled=i == 0, width="stretch", key="tour_prev", on_click=_go, args=(i - 1,))
    c2.markdown(f'<div style="text-align:center;color:var(--muted);padding-top:.5rem">'
                f'{"Next: " + STEPS[i + 1] if i < n - 1 else "End of the guided demo"}</div>', unsafe_allow_html=True)
    if i < n - 1:
        c3.button("Next →", type="primary", width="stretch", key="tour_next", on_click=_go, args=(i + 1,))
    else:
        c3.button("Start again", width="stretch", key="tour_restart", on_click=_go, args=(0,))
