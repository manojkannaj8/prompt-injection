"""G. Methodology and limitations: the six-part pipeline, the rules, and an honest project-status table."""
import streamlit as st

import components as ui
from content.facts import facts
from views import widgets

# Status of each planned component, as recorded in PROGRESS.md (11 October 2026). Nothing here is upgraded by the UI.
STATUS = [
    ("1. Input and data collection", "Done", f"Saved AgentDojo runs indexed. The raw run files are not in the project folder."),
    ("2. Preprocessing and feature engineering", "Done", "Tool outputs, exposure rows, MiniLM embeddings, flags, masked copy, four sets of grouped folds."),
    ("3. Model development: lookup baseline, Logistic Regression, XGBoost", "Done", "XGBoost was refitted in Part 6 because the original model was not saved."),
    ("3. Model development: fine-tuned encoder", "Not started", "Needs a GPU build of PyTorch or a Colab GPU session. No encoder outputs exist."),
    ("3. Baselines: direct external-detector comparison", "Not started", "ProtectAI DeBERTa or Prompt Guard has not been scored on this project's samples."),
    ("4. Risk prediction", "Done", "Calibrated P(obey) and risk for every tool output."),
    ("5. Decision and replay analysis", "Done", "Replay for 28 agents, leave-one-agent-out replay, NotInject over-defence check."),
    ("6. Explanations and alerts: SHAP, coefficients, reasons, faithfulness", "Done", "For Logistic Regression and XGBoost."),
    ("6. Explanations: Integrated Gradients", "Not started", "Depends on the fine-tuned encoder, which does not exist."),
    ("Written dissertation results chapter and remaining diagrams", "Not done", "RESULTS.md is the starting point."),
    ("This dashboard", "Done", "Reads the saved files. Live scoring covers Stage 1 only."),
]
TONE = {"Done": "good", "Not started": "crit", "Not done": "crit", "Partly done": "warn"}


def render():
    f = facts()
    ui.header("Methodology", "How the project works, and where it stops",
              "The project is a chain of six steps that turns recorded AI-agent runs into a risk score, a decision and an explanation. This page "
              "walks through the chain, states the exact rules, and lists plainly what has not been built.",
              "Six-part pipeline over saved AgentDojo runs: indexing, feature construction with grouped folds, Stage 1 and Stage 2 models, "
              "calibrated risk, threshold replay, and attribution. Status below mirrors PROGRESS.md.")

    st.header("The six-part research pipeline")
    widgets.pipeline()
    st.write("")
    parts = [
        ("1. Input and data collection",
         f"The starting material is {f.runs} saved runs from the AgentDojo benchmark, covering {f.configs} agent configurations across {f.suites} "
         f"task areas. {f.attacked_runs} runs were attacked and {f.clean_runs} were clean. Each run records whether the task was completed and "
         "whether the attack succeeded.",
         "runs_index.parquet: config, suite, user_task, injection_task, attack, utility, security. Row number = run_id."),
        ("2. Preprocessing and feature engineering",
         f"Each run is cut into the pieces of text its tools returned: {f.outputs} tool outputs. Each is labelled (does it contain a planted "
         f"instruction?). The first planted instruction of each attacked run is an 'exposure' ({f.exposure} of them), labelled with whether the agent "
         "obeyed. The text is turned into numbers in four ways: word weights, meaning vectors, simple counts, and the tool and agent names.",
         "samples_features.parquet, stage2_exposure.parquet, emb_task.npy / emb_out.npy / sim.npy. Grouped folds: fold_inj, fold_user, fold_attack, fold_cfg. "
         "Task ids, attack and suite are never model inputs."),
        ("3. Model development",
         "Two questions, two models. Stage 1 decides whether a planted instruction is present. Stage 2 predicts whether this agent would obey it. "
         "Three Stage 2 models were compared: a lookup of each agent's usual behaviour, Logistic Regression, and XGBoost.",
         "Stage 1: liblinear LR, balanced, TF-IDF 1-2 grams + tool. Stage 2: LR and XGBoost (300 trees, depth 6, lr 0.1). All scores out-of-fold on grouped splits."),
        ("4. Risk prediction",
         "Every tool output gets an injection probability, an obedience probability, and their product: the risk score. The obedience probability "
         "is corrected (calibrated) so that its percentages match reality.",
         "risk_scores.parquet. p_obey on exposure rows = saved XGBoost score through isotonic regression fitted on the other four folds; elsewhere the agent's obey rate."),
        ("5. Decision and replay analysis",
         f"Rules turn the risk score into HIGH RISK, SUSPICIOUS or SAFE. The saved runs are then replayed to estimate how many attacks the guard "
         "would have stopped and how many normal tasks it would have interrupted.",
         f"T_HIGH = {f.t_high}; gate = max risk up to the exposure step. replay_all_configs.csv, replay_leave_one_agent_out.csv, notinject_scores.parquet."),
        ("6. Output explanations and alerts",
         f"Each of the {f.alerts} alerts gets reasons: which words made it look like an injection and what made the agent likely to obey. The "
         "explanations are themselves tested for honesty.",
         "TreeSHAP (out-of-fold), LR coefficients, template-masking shortcut check, deletion faithfulness test. Outputs in part6/."),
    ]
    for title, plain, tech in parts:
        with st.expander(title, expanded=False):
            st.markdown(plain)
            if not ui.is_beginner():
                st.caption(tech)

    st.header("The two stages")
    c1, c2 = st.columns(2)
    c1.markdown(ui.card("Stage 1 · is an injection present?", "P(injection)",
                        f"Reads the tool output only. Logistic Regression on word weights and the tool name. ROC-AUC {f.s1_roc} on unseen user tasks. "
                        "Mostly detects ordinary sentences sitting inside structured tool data.", "blue"), unsafe_allow_html=True)
    c2.markdown(ui.card("Stage 2 · will this agent obey?", "P(obey)",
                        f"Reads the tool output, the user task and the agent identity. XGBoost with calibrated output. PR-AUC {f.xgb_saved} (saved model) on "
                        f"unseen injection tasks, where guessing scores {f.obey3}.", "blue"), unsafe_allow_html=True)
    st.write("")
    ui.original_figure("report_figures/fig_architecture.png", "Show the architecture diagram from the project report")

    st.header("The risk formula and the decision thresholds")
    st.markdown('<div class="pi-eq">risk = P(injection) × P(obey)</div>', unsafe_allow_html=True)
    st.write("")
    widgets.decision_rule()
    st.caption(f"The {f.t_high} threshold is the value that reproduces replay_all_configs.csv for all 28 agents; it was not stored by Parts 1-5. "
               "The SUSPICIOUS rule was first written as code in Part 6. This dashboard uses both unchanged.")

    st.header("Project status")
    ui.note("Items marked <b>Not started</b> or <b>Not done</b> do not exist. No result on this dashboard comes from them.", warn=True)
    st.write("")
    rows = "".join(f'<tr><td style="padding:7px 10px">{ui.esc(n)}</td><td style="padding:7px 10px">{ui.chip(s.upper(), TONE[s])}</td>'
                   f'<td style="padding:7px 10px;color:var(--ink2)">{ui.esc(d)}</td></tr>' for n, s, d in STATUS)
    st.markdown(f'<div class="pi-card"><table style="width:100%;border-collapse:collapse;font-size:.9rem">'
                f'<tr style="color:var(--muted);text-align:left"><th style="padding:7px 10px">Component</th><th style="padding:7px 10px">Status</th>'
                f'<th style="padding:7px 10px">Note</th></tr>{rows}</table></div>', unsafe_allow_html=True)

    st.header("Limitations")
    ui.steps([
        "<b>Replay is an estimate.</b> A blocked agent might retry, recover or fail in another way. No live guard was run.",
        "<b>One benchmark, repetitive attacks.</b> One fixed wrapper covers most attacked runs and the models use it. Results may not hold against attacks written to avoid the guard.",
        "<b>The explained XGBoost is a refit,</b> not the model whose scores drive the replay. The two agree closely but not exactly.",
        "<b>Stage 2 reasons exist only for exposure rows.</b> A false alarm on a harmless output can only show the Stage 1 words and the agent's usual obey rate.",
        "<b>The obey label is per run.</b> It says the attacker's goal was reached somewhere in the run, not that this output caused it.",
        "<b>No hyperparameter search.</b> One fixed XGBoost setting was used in Part 6.",
        "<b>Parts 1-4 code is not in the folder.</b> Their outputs were checked wherever a number could be re-derived; the code itself could not be.",
        "<b>Live scoring in this dashboard covers Stage 1 only.</b> For new text the obedience probability is the agent's historical obey rate.",
    ])
