"""F. Security versus usefulness: the saved replay results, per configuration, defended and undefended."""
import streamlit as st

import components as ui
from content.facts import facts
from core import data, replay, rules
from views import charts


def _inspector(rp, f):
    st.subheader("Inspect one agent configuration")
    cfgs = list(rp.sort_values("ASR no guard", ascending=False).config)
    cfg = st.selectbox("Agent configuration", cfgs, key="tr_cfg",
                       format_func=lambda c: f"{c}{'  (has a built-in defence)' if replay.is_defended(c) else ''}")
    r = rp.set_index("config").loc[cfg]
    lo, hi = replay.task_ci(cfg)
    st.markdown(ui.provenance("saved"), unsafe_allow_html=True)
    ui.cards([("Attack success, no guard", f"{r['ASR no guard']:.1%}", f"{int(r.n_attacked)} attacked runs, main attack", ""),
              ("Attack success, with guard", f"{r['ASR guard']:.1%}", f"cut by {r['ASR cut (pts)']:.1f} points", "blue"),
              ("Task success, no guard", f"{r['task no guard']:.1%}", f"{int(r.n_clean)} clean runs", ""),
              ("Task success, with guard", f"{r['task guard']:.1%}", f"change {r['task change (pts)']:+.1f} points · 95% interval {lo:.0%} to {hi:.0%}",
               "crit" if r["task change (pts)"] < -5 else "blue")])
    if r["task change (pts)"] < -5:
        ui.note(f"This agent loses more than 5 points of task success at the single {f.t_high} threshold. It has a high usual obey rate, so a harmless "
                "output with a moderate Stage 1 score is enough to cross the line. Per-agent thresholds are future work and have not been built.", warn=True)
    st.write("")
    ui.chart_title(f"Replay across thresholds · {cfg}", "What if the HIGH RISK line were somewhere else?",
                   "Orange: share of attacks that still succeed. Blue: share of normal tasks still completed. Moving the line left blocks more "
                   "(both lines fall); moving it right blocks less (both rise). A good threshold keeps orange low and blue high.",
                   "Recomputed from saved scores with the notebooks' replay function. Equals replay_all_configs.csv at 0.20. The curve is tuned on the runs it is drawn on.")
    st.markdown(ui.provenance("whatif"), unsafe_allow_html=True)
    t = st.slider("What-if threshold (does not change the configured 0.20)", 0.01, 0.95, rules.T_HIGH, 0.01, key="tr_t")
    a, u, _, _ = replay.replay(cfg, t)
    ui.show(charts.sweep(cfg, t), key="tr_sweep")
    st.caption(f"At threshold {t:.2f}: attack success {a:.1%}, task success {u:.1%}. "
               + ("This is the configured threshold." if abs(t - rules.T_HIGH) < 1e-9 else f"The configured threshold remains {f.t_high}."))
    ui.source("risk_scores.parquet", "runs_index.parquet", "replay_all_configs.csv")
    ui.explain("threshold_sweep")


def render():
    f = facts()
    rp, loo = data.replay_all(), data.replay_loo()
    ui.header("Security vs usefulness", "Does the guard stop attacks without stopping work?",
              "A defence has two jobs that pull against each other: stop attacks, and stay out of the way of normal tasks. This page shows both "
              "numbers together, for one AI model in detail and then for every agent that was tested.",
              f"Replay of saved runs. A run is blocked when the highest risk up to its exposure point reaches the threshold. Attack success on "
              f"`important_instructions` runs, utility on clean runs. Configured threshold {f.t_high}.")
    ui.note("<b>Replay, not a live guarantee.</b> These results estimate what the guard would have done on recorded runs. A blocked agent might retry "
            "or fail another way, and no attacker tried to evade this guard. They do not demonstrate a production-ready security guarantee.", warn=True)

    tabs = st.tabs(["GPT-4o: the main figure", "Every agent", "Agents the guard never saw"])

    with tabs[0]:
        ui.chart_title("Security vs usefulness on GPT-4o (replay of saved runs)", "Finding a defence that stops attacks but not normal work",
                       f"Each point is one way of protecting the same AI model. Best is the top-left corner: few successful attacks, many normal "
                       f"tasks done. With no defence, {f.g4_asr_before} of attacks succeed. The benchmark's built-in detector cuts attacks to {f.det_asr} "
                       f"but also stops most normal work ({f.det_task}). This project's guard (orange) reaches {f.g4_asr_after} while keeping {f.g4_task_after}. "
                       f"Do not read small vertical differences as real: the orange bar shows how uncertain the task figure is.",
                       f"Grey: saved GPT-4o configurations as run. Blue line: guard replay for thresholds 0.01 to 0.60. Orange: guard at {f.t_high} with a "
                       f"Wilson 95% interval on utility (n = {f.g4_clean} clean runs).")
        ui.show(charts.tradeoff_gpt4o(), key="tr_main")
        st.markdown(ui.provenance("saved") + " grey points &nbsp; " + ui.provenance("whatif") + " blue line and orange point", unsafe_allow_html=True)
        ui.source("replay_all_configs.csv", "risk_scores.parquet", "runs_index.parquet")
        g = rp.set_index("config")
        rows = []
        for cfg, lab in charts.GPT4O_SETUPS.items():
            if cfg in g.index:
                rows.append({"Setting (main attack)": f"GPT-4o, {lab.lower()}", "Attack success": f"{g.loc[cfg, 'ASR no guard']:.1%}",
                             "Task success, no attack": f"{g.loc[cfg, 'task no guard']:.1%}", "Source": "Saved run"})
        rows.append({"Setting (main attack)": f"GPT-4o, our guard at risk ≥ {f.t_high}", "Attack success": f.g4_asr_after, "Task success, no attack": f.g4_task_after,
                     "Source": "Replay (replay_all_configs.csv)"})
        rows.append({"Setting (main attack)": "GPT-4o, our guard, threshold chosen on held-out folds", "Attack success": "5.7%",
                     "Task success, no attack": "71.5% (95% CI 63.0 to 78.8)", "Source": "Saved Part 5 result quoted from RESULTS.md; its code is not in the folder"})
        st.dataframe(rows, hide_index=True, width="stretch")
        ui.original_figure("fig_security_vs_usefulness.png", "Show the original figure saved by the Part 5 notebook (fig_security_vs_usefulness.png)")
        ui.explain("tradeoff_gpt4o")

    with tabs[1]:
        ui.cards([("Attack success, pooled", f"{f.asr_before} <small>→</small> {f.asr_after}", f"{f.replay_agents} agents, guard at {f.t_high}", "blue"),
                  ("Task success, pooled", f"{f.task_before} <small>→</small> {f.task_after}", "clean runs, weighted by runs", ""),
                  ("Median task change", f"{f.task_median} <small>points</small>", "the typical agent loses nothing", "good"),
                  ("Agents losing > 5 points", f.lose5, f"{f.worst1} and {f.worst2}", "crit")])
        st.write("")
        hi = st.session_state.get("tr_cfg")
        c1, c2 = st.columns([2.3, 1])
        with c1:
            ui.chart_title(f"Attack success per configuration: no guard vs guard at risk ≥ {f.t_high}", "Attacks before and after, agent by agent",
                           "Grey dot: no guard. Blue dot: with the guard. A long line means the guard removed many attacks. Agents that already "
                           "resisted well have short lines because there was little left to fix.",
                           "replay_all_configs.csv, sorted by undefended attack success.")
            ui.show(charts.dumbbell(rp, "ASR no guard", "ASR guard", f"Guard at {f.t_high}", highlight=hi), key="tr_db")
        with c2:
            ui.chart_title("Cost in task success", "The price in normal tasks",
                           "Bars to the left mean normal tasks were lost. Most agents lose nothing.", "Change in utility on clean runs, points. Same row order.")
            ui.show(charts.task_change(rp, highlight=hi), key="tr_tc")
        ui.source("replay_all_configs.csv")
        ui.explain("tradeoff_agents")
        with st.expander("Full table: every configuration, defended and undefended", icon=":material/table:"):
            t = rp.copy()
            for c in ["ASR no guard", "ASR guard", "task no guard", "task guard"]:
                t[c] = (t[c] * 100).round(1)
            st.dataframe(t.rename(columns={"config": "Agent configuration", "n_attacked": "Attacked runs", "n_clean": "Clean runs",
                                           "ASR no guard": "Attack success %, no guard", "ASR guard": "Attack success %, guard",
                                           "task no guard": "Task success %, no guard", "task guard": "Task success %, guard"}),
                         hide_index=True, width="stretch", height=480)
        st.divider()
        _inspector(rp, f)

    with tabs[2]:
        ui.cards([("Attack success, pooled", f"{f.asr_before} <small>→</small> {f.loo_after}", "guard trained without the agent it protects", "blue"),
                  ("Worst task-success loss", f"{f.loo_worst} <small>points</small>", "no agent loses more than this", "good"),
                  ("Threshold used", f.loo_t, "chosen on the other agents' clean runs", "")])
        st.write("")
        ui.chart_title("Leave-one-agent-out replay (Day 2 results): the guard is trained without the agent it protects", "Protecting an AI model the guard has never seen",
                       f"The guard was rebuilt 28 times, each time without one agent, then tested on that agent. Attacks still fall from {f.asr_before} to {f.loo_after}.",
                       "Stage 2 LR without the agent feature; per-agent threshold = lowest t costing ≤ 1 point of utility on the other agents' clean runs.")
        ui.show(charts.dumbbell(loo, "ASR no guard", "ASR guard", "Guard that never saw this agent"), key="tr_loo")
        st.markdown(ui.provenance("saved"), unsafe_allow_html=True)
        ui.source("replay_leave_one_agent_out.csv")
        ui.original_figure("part6/fig_leave_one_agent_out.png")
        ui.explain("loo")
