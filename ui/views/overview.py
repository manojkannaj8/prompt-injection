"""A. Overview dashboard: project statistics, each with its own dataset and denominator."""
import streamlit as st

import components as ui
from content.facts import facts
from core import rules, stats
from views import charts


def render():
    s, f = stats.overview(), facts()
    ui.header("Overview", "Prompt-injection defence dashboard",
              f"AI agents read emails, files and web pages. Attackers can hide instructions in that text, a trick called "
              f"{ui.term('prompt injection')}. This project built a two-step guard that estimates how risky each piece of text is, "
              f"and measured it on {f.runs} recorded test runs. Everything on this page is a real number from those saved runs.",
              f"Two-stage risk model for indirect prompt injection, evaluated by replaying {f.runs} saved AgentDojo runs "
              f"({f.configs} agent configurations). Stage 1: P(injection). Stage 2: calibrated P(obey). risk = product, HIGH RISK at ≥ {f.t_high}.")
    ui.note("<b>Research prototype.</b> All results come from saved benchmark runs that were re-examined after the fact. "
            "They are not measurements of a live, deployed defence. The fine-tuned encoder, Integrated Gradients and a direct "
            "external-detector comparison are <b>not built</b>.", warn=True)

    # ---------------------------------------------------------------- data volume
    st.header("What was analysed")
    ui.cards([
        ("Saved runs", f.runs, f"{f.attacked_runs} attacked · {f.clean_runs} clean", ""),
        ("Agent configurations", f.configs, f"{f.agents} of them have attacked runs with an obey label", ""),
        ("Tool outputs analysed", f.outputs, f"{f.injected} contain an injection · {f.harmless} harmless", ""),
        ("Injection exposures", f.exposure, f"first planted instruction of each attacked run · {f.obey_pct} were obeyed", "blue"),
    ])
    ui.source("runs_index.parquet", "risk_scores.parquet", "stage2_exposure.parquet")
    ui.explain("data_volume")

    # ---------------------------------------------------------------- decisions
    st.header("What the guard decided")
    opts = {f"First encounters with a planted instruction ({f.exposure})": "exposure",
            f"Everything the guard sees up to that point ({f.gate_n})": "gate"}
    choice = st.radio("Count decisions over", list(opts), horizontal=True, key="ov_denominator",
                      help="The first option is where we know whether the agent really obeyed. The second adds the harmless outputs that came before.")
    which = opts[choice]
    counts = s["decisions_exposure"] if which == "exposure" else s["decisions_gate"]
    total = sum(counts.values())
    ui.cards([(f"{ui.DECISION_ICON[d]} {d}", f"{counts[d]:,}", f"{counts[d] / total:.1%} · {rules.ACTION[d]}", tone)
              for d, tone in zip(rules.ORDER, ["crit", "warn", "good"])])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        ui.chart_title("Decisions under the documented rule", "How the guard sorted the text it read",
                       f"Each bar is a box the guard puts text into. Red is stopped, amber is let through but logged, green is let through.",
                       f"HIGH RISK: risk ≥ {f.t_high}. SUSPICIOUS: P(injection) ≥ {f.p_log} and risk < {f.t_high}. Counts over the selected denominator.")
        ui.show(charts.decisions(counts, "exposure rows" if which == "exposure" else "outputs up to the exposure point"), key="ov_dec")
    with c2:
        ui.chart_title("Obey rate by decision level (exposure rows)", "Was the guard right to worry?",
                       f"Of the texts in each box, the share where the agent really did follow the planted instruction. "
                       f"A good guard has a tall red bar and short amber and green bars.",
                       "Mean of `obeyed` per decision level. Defined on exposure rows only, whatever denominator is selected on the left.")
        ui.show(charts.obey_by_level(s["levels"]), key="ov_obey")
    ui.source("risk_scores.parquet", "stage2_exposure.parquet", "part6/table_decision_levels.csv")
    ui.explain("decisions")

    # ---------------------------------------------------------------- models
    st.header("How good are the two models?")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.subheader(ui.pick("Step 1: spotting a planted instruction", "Stage 1 detection (unseen user tasks, out-of-fold)"))
        ui.cards([("ROC-AUC", f.s1_roc, ui.pick("1.0 would be perfect ranking", "Part 6 refit, grouped 5-fold"), "blue"),
                  ("Injections caught", f.s1_recall, ui.pick("at the 0.5 cut-off", "recall at p ≥ 0.5"), "")])
        st.write("")
        ui.cards([("False alarms", f.s1_fa, ui.pick("harmless texts flagged", "FPR at p ≥ 0.5"), ""),
                  ("Saved scores", f"{f.s1_prec} <small>/ {f.s1_rec}</small>", ui.pick("precision / recall of the scores the guard uses",
                                                                                      "precision / recall of p_inj in risk_scores.parquet"), "")])
        ui.source("part6/table_stage1_raw_vs_masked.csv", "part6/table_threshold_metrics.csv")
        ui.explain("stage1_perf")
    with c2:
        ui.chart_title("Stage 2 PR-AUC, unseen injection tasks", "Step 2: predicting whether the agent will obey",
                       f"Higher bars are better. The line is what pure guessing would score. Knowing only which agent is running already beats "
                       f"guessing; reading the content nearly doubles that.",
                       f"Grouped 5-fold by injection task, {f.exposure} exposure rows, base rate {f.obey3}. The saved Part 3 XGBoost drives the risk scores; the others are Part 6 refits.")
        ui.show(charts.stage2_main(), key="ov_s2")
        ui.source("part6/table_stage2_results.csv")
        ui.explain("stage2_perf")

    # ---------------------------------------------------------------- guard effect
    st.header("What the guard would have changed")
    c1, c2 = st.columns(2)
    with c1:
        ui.chart_title(f"Attack success, no guard vs guard at risk ≥ {f.t_high} (replay)", "Attacks that got through",
                       f"Grey: nothing guarding the agent. Blue: with this guard. Across {f.replay_agents} agents, successful attacks fall from "
                       f"{f.asr_before} to {f.asr_after}.",
                       f"Main attack (`important_instructions`). Pooled rate weighted by attacked runs; GPT-4o n = {f.g4_attacked}.")
        ui.show(charts.guard_effect("attack"), key="ov_att")
        ui.explain("attack_before_after")
    with c2:
        ui.chart_title(f"Task success on clean runs, no guard vs guard at risk ≥ {f.t_high} (replay)", "Normal work that still got done",
                       f"The two bars should be close: that means the guard rarely interrupts honest tasks. For GPT-4o it goes from "
                       f"{f.g4_task_before} to {f.g4_task_after}.",
                       f"Utility on non-attacked runs. Pooled rate weighted by clean runs; GPT-4o n = {f.g4_clean} (Wilson 95% interval about ±8 points).")
        ui.show(charts.guard_effect("task"), key="ov_task")
        ui.explain("task_before_after")
    ui.source("replay_all_configs.csv")
    ui.note(f"These are <b>replay estimates</b>: saved runs re-examined with the guard's scores. {f.lose5} agents lose more than 5 points of task "
            f"success at this single threshold. See <i>Security vs Usefulness</i> for every agent.")
