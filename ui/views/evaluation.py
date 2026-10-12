"""E. Model evaluation: the saved result tables, with each row marked as a saved result or a Part 6 refit."""
import streamlit as st

import components as ui
from content.facts import facts
from core import data, paths, stats
from views import charts, widgets


def render():
    f = facts()
    ui.header("Model evaluation", "How well do the models work?",
              f"Scores for the second, harder question: will the agent obey? The key idea on this page is that <b>how</b> you test matters as much "
              f"as the score. Tests that hide a whole group from the model are the honest ones. The 'random split' is shown only to demonstrate "
              f"how much an unfair test flatters the model. For reference, pure guessing scores {f.obey3} on {ui.term('PR-AUC')}.",
              f"Stage 2 on {f.exposure} exposure rows (base rate {f.obey3}). Grouped 5-fold by injection task (primary), user task and attack style; "
              f"leave-one-agent-out over 28 agents; stratified random 5-fold as a leaky comparison.")
    st.markdown(ui.provenance("saved") + " " + ui.provenance("refit"), unsafe_allow_html=True)
    st.caption("Both kinds of result appear on this page. Every table has a Source column that says which is which.")

    tabs = st.tabs(["Models and test protocols", "PR and ROC curves", "Calibration", "Decisions and errors", "False alarms", "Data balance"])

    with tabs[0]:
        metric = st.radio("Metric", ["PR-AUC", "ROC-AUC"], horizontal=True, key="ev_metric")
        text = st.radio("Text version", ["raw", "masked"], horizontal=True, key="ev_text",
                        format_func=lambda x: "Raw text" if x == "raw" else "Template-masked text (three grouped protocols only)")
        ui.chart_title(f"Random splits flatter the model: Stage 2 {metric} by protocol", "The same models, tested five different ways",
                       f"Each group of bars is one way of testing. The leftmost group mixes everything, which lets the model recognise things it has "
                       f"already seen, and gives XGBoost {f.xgb_rnd}. When whole attack goals are hidden it scores {f.xgb_inj}. Trust the 'unseen' groups.",
                       "Part 6 refits. Random split is a comparison only; the primary generalisation result is the unseen injection task. "
                       "No lookup baseline exists for an unseen agent.")
        ui.show(charts.protocols(metric, text), key="ev_proto")
        ui.note("<b>Random split = comparison, not a result.</b> With a random split the same injection tasks sit in training and test. "
                f"It overstates XGBoost by {f.rnd_gap} PR-AUC and must not be read as how the guard would do on new attacks.", warn=True)
        st.write("")
        st.markdown("**Logistic Regression versus XGBoost** (PR-AUC, raw text)")
        ui.cards([("Unseen injection task", f"{f.lr_inj} <small>vs</small> {f.xgb_inj}", "LR vs XGBoost · main result", "blue"),
                  ("Unseen user task", f"{f.lr_usr} <small>vs</small> {f.xgb_usr}", "XGBoost ahead", ""),
                  ("Unseen attack style", f"{f.lr_att} <small>vs</small> {f.xgb_att}", "level: no advantage for XGBoost", "warn"),
                  ("Unseen agent", f"{f.lr_agt} <small>vs</small> {f.xgb_agt}", "level · agent feature removed", "warn")])
        st.write("")
        st.markdown("**Model-only baseline** (knowing only which agent is running)")
        ui.cards([("Lookup baseline", f.base_inj, "unseen injection task", ""), ("Best content model", f.xgb_inj, "XGBoost refit, same protocol", "blue"),
                  ("Saved Part 3 XGBoost", f.xgb_saved, f"the model behind the risk scores · ROC-AUC {f.xgb_saved_roc}", ""),
                  ("Refit vs saved", f.refit_rho, "Spearman rank agreement of the two XGBoost score sets", "")])
        st.write("")
        st.markdown("**Full results table**")
        t = stats.stage2_table().rename(columns={"protocol": "Protocol", "text": "Text", "model": "Model", "source": "Source", "n folds": "Folds"})
        st.dataframe(t[["Protocol", "Text", "Model", "Source", "PR-AUC", "ROC-AUC", "PR-AUC worst fold", "PR-AUC best fold", "Folds"]].round(3),
                     hide_index=True, width="stretch", height=420)
        ui.source("part6/table_stage2_results.csv")
        ui.original_figure("part6/fig_random_vs_grouped_splits.png")
        ui.explain("protocols")
        ui.explain("stage2_perf")

    with tabs[1]:
        if not paths.available("part6/cache/oof_lr_raw_inj.npy"):
            st.info("The out-of-fold score files of the Part 6 refits (part6/cache) are not present, so only the saved XGBoost curve can be drawn. "
                    "The saved figure below shows all three models.")
        c1, c2 = st.columns(2)
        with c1:
            ui.chart_title("Precision-recall, unseen injection tasks (pooled out-of-fold scores)", "Catching attacks versus raising false alarms",
                           "Higher and further right is better. Every model's curve is far above the 'guessing' line, and XGBoost sits on top.",
                           "x = recall, y = precision. Horizontal line = base rate.")
            ui.show(charts.pr_curves(), key="ev_pr")
        with c2:
            ui.chart_title("ROC, unseen injection tasks (pooled out-of-fold scores)", "The same comparison drawn another way",
                           "Curves closer to the top-left corner are better. The diagonal is what guessing looks like.",
                           "x = false positive rate, y = true positive rate.")
            ui.show(charts.roc_curves(), key="ev_roc")
        st.caption("XGBoost is the saved Part 3 model (the reference for the replay). Logistic Regression and the lookup are Part 6 refits.")
        ui.source("oof_xgb_fold_inj.npy", "part6/cache/oof_lr_raw_inj.npy", "part6/cache/oof_base_raw_inj.npy", "stage2_exposure.parquet")
        ui.original_figure("part6/fig_pr_roc_unseen_injection_task.png")
        ui.explain("pr_roc")

    with tabs[2]:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            ui.chart_title("Calibration of P(obey) (10 equal-size bins)", "Can the percentages be taken at face value?",
                           f"Points on the diagonal mean the model's percentages match reality. The grey raw scores sit below the line (overconfident). "
                           f"The blue corrected scores follow it closely.",
                           "Saved XGBoost score before and after isotonic regression fitted on the other four folds. Exposure rows.")
            ui.show(charts.calibration(), key="ev_cal")
        with c2:
            ui.cards([("Calibration error before", f.ece_raw, "expected calibration error (ECE)", "warn")])
            st.write("")
            ui.cards([("Calibration error after", f.ece_cal, "after isotonic regression, out-of-fold", "good")])
            st.write("")
            st.dataframe(data.table("table_calibration").round(4), hide_index=True, width="stretch")
        ui.source("part6/table_calibration.csv", "oof_xgb_fold_inj.npy", "risk_scores.parquet")
        ui.original_figure("part6/fig_calibration.png")
        ui.explain("calibration")

    with tabs[3]:
        tm = data.table("table_threshold_metrics")
        c1, c2 = st.columns(2)
        with c1:
            ui.chart_title("Stage 1: p_inj ≥ 0.5 vs injection present (all tool outputs)", "Step 1: spotting planted instructions",
                           f"Top-left and bottom-right are correct decisions. Step 1 is right almost every time: {f.s1_fp} false alarms and {f.s1_fn} misses "
                           f"out of {f.outputs} texts.",
                           "Saved out-of-fold p_inj from risk_scores.parquet. Row-normalised.")
            r = tm.iloc[0]
            ui.show(charts.confusion(int(r.TP), int(r.FP), int(r.FN), int(r.TN), ("harmless", "injection")), key="ev_cm1")
        with c2:
            ui.chart_title(f"Guard: risk ≥ {f.t_high} vs agent obeyed (exposure rows)", "The full guard: stopping attacks that would work",
                           f"The guard stopped {f.g_tp} attacks that would have worked and missed {f.g_fn}. It also stopped {f.g_fp} that the agent would "
                           f"have ignored anyway.",
                           "Saved risk against `obeyed`. Row-normalised.")
            r = tm.iloc[1]
            ui.show(charts.confusion(int(r.TP), int(r.FP), int(r.FN), int(r.TN), ("agent resisted", "agent obeyed"), ("not stopped", "stopped")), key="ev_cm2")
        st.dataframe(tm.round(3), hide_index=True, width="stretch")
        ui.source("part6/table_threshold_metrics.csv")
        ui.explain("confusion")
        st.markdown("**What the SUSPICIOUS level buys**")
        lv = data.table("table_decision_levels").rename(columns={"decision": "Decision", "n": "Tool outputs", "obeyed": "Obeyed in the saved run",
                                                                "obey_rate": "Obey rate", "share of all obeyed attacks": "Share of all obeyed attacks"})
        st.dataframe(lv.round(3), hide_index=True, width="stretch")
        ui.note(f"A plain detector that blocks whenever Stage 1 fires would stop {f.b_blocked} of these outputs to catch {f.b_rec} of obeyed attacks. "
                f"The guard stops {f.n_high} and catches {f.high_share}. The price: {f.susp_obeyed} obeyed attacks were only logged.")
        ui.source("part6/table_decision_levels.csv")
        ui.original_figure("part6/fig_confusion_matrices.png")
        ui.explain("decision_levels")
        with st.expander("Try other thresholds on the saved scores (what-if)", icon=":material/tune:"):
            widgets.threshold_lab("ev_thr")

    with tabs[4]:
        ui.chart_title("Over-defence check: how often harmless text is flagged", "False alarms on harmless text",
                       f"Lower is better. Grey: the first step alone raises an alarm. Blue: the full guard would actually stop the agent. "
                       f"Rates are low overall, but noticeably higher in attack-free runs.",
                       "Share of harmless texts with p_inj ≥ 0.5 and with risk ≥ 0.20. NotInject risk uses a fixed obey rate per row label.")
        ui.show(charts.over_defence(), key="ev_od")
        st.dataframe(data.table("table_over_defence").round(4), hide_index=True, width="stretch")
        ui.note(f"False alarms are not evenly spread. In clean runs {f.od_clean_s1} of harmless outputs are flagged by Stage 1 and {f.od_clean_guard} are blocked, "
                "because the harmless content of the slots where injections are normally planted appears nowhere else in the data. "
                "This is where the low false-alarm rate is least reliable.", warn=True)
        ui.source("part6/table_over_defence.csv", "notinject_scores.parquet")
        ui.original_figure("part6/fig_over_defence.png")
        ui.explain("over_defence")

    with tabs[5]:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ui.chart_title("The same attack, undefended models: attack success", "How easily each AI model was fooled",
                           f"The same planted instruction worked on some models {f.asr_max} of the time and on others {f.asr_min}. "
                           "This spread is why the project predicts obedience per agent.",
                           "Main attack, configurations without a built-in defence.")
            ui.show(charts.attack_per_agent(), key="ev_asr")
            ui.source("part6/table_attack_success_per_agent.csv")
            ui.explain("attack_per_agent")
        with c2:
            st.markdown("**Label balance per test fold** (exposure rows)")
            lb = data.table("table_label_balance")
            proto = st.selectbox("Protocol", list(lb.protocol.unique()), key="ev_lb")
            st.dataframe(lb[lb.protocol == proto].drop(columns="protocol").round(3), hide_index=True, width="stretch", height=420)
            st.caption(ui.pick("Each fold is one slice of the data used as the test set in turn. The obey rate differs a lot between slices, "
                               "which is one reason scores vary from fold to fold.",
                               "Grouped folds are not stratified; obey rate ranges from 0.03 to 0.42 across attack-style folds."))
            ui.source("part6/table_label_balance.csv")
