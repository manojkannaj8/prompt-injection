"""C. Explainability: the saved global explanation artefacts of Part 6. No per-input SHAP is computed here."""
import streamlit as st

import components as ui
from content.facts import facts
from core import data
from views import charts


def _limits(f):
    st.subheader("What the explanations revealed about the model's weaknesses")
    ui.steps([
        f"<b>The model uses repeated attack-wrapper wording.</b> Terms from the benchmark's fixed attack sentence carry {f.wrap_raw} of the word-level "
        f"influence, and {f.wrap_top20} of the 20 most important words are wrapper terms.",
        f"<b>The [WRAP] placeholder remains a shortcut.</b> When the wrapper is masked, the placeholder itself carries {f.wrap_masked} of the word-level "
        "influence. The masked test hides the wording of the wrapper, not its presence.",
        f"<b>Agent identity contributes substantially.</b> {f.sh_agent} of the obedience score comes from simply knowing which agent is running.",
        f"<b>The IBAN, email and URL novelty flags contribute very little.</b> Each has an average influence of at most {f.novel_max}, against {f.cue_shap} for the "
        "count of instruction cue words.",
        "<b>Explanations are not proof of causation.</b> They show which inputs moved the model's score. They do not show why an agent obeyed.",
    ])


def render():
    f = facts()
    ui.header("Explainability", "What the models rely on",
              f"A model's score is only as trustworthy as the clues it uses. This page looks inside the second model with a method called "
              f"{ui.term('SHAP')} and inside the simpler model by reading its weights. The honest summary: the explanations are faithful to the "
              f"model, and they expose real weaknesses.",
              "Out-of-fold TreeSHAP of the refitted Stage 2 XGBoost (raw and template-masked), Logistic Regression coefficients, the shortcut "
              "check and the deletion-based faithfulness test. All artefacts are the saved Part 6 outputs.")
    st.markdown(ui.provenance("refit") + " " + ui.chip("GLOBAL EXPLANATIONS · NO PER-INPUT SHAP FOR NEW TEXT"), unsafe_allow_html=True)
    ui.note(f"The explained XGBoost is the <b>Part 6 refit</b>. The original Part 3 model was never saved, only its scores. The refit ranks the rows "
            f"almost identically (Spearman {f.refit_rho}). Per-row SHAP values were stored only as the top reasons of each saved alert, which you can "
            "open in the <i>Alerts</i> page.")
    st.write("")
    _limits(f)

    tabs = st.tabs(["Feature importance (SHAP)", "Feature families and agents", "Raw vs template-masked", "Logistic Regression weights", "Faithfulness", "Beeswarm"])

    with tabs[0]:
        v = st.radio("Text version", ["raw", "masked"], horizontal=True, key="ex_ver", format_func=lambda x: "Raw text" if x == "raw" else "Template-masked text")
        n = st.slider("How many features to show", 10, 40, 18, 1, key="ex_n")
        ui.chart_title(f"What the Stage 2 XGBoost relies on: out-of-fold TreeSHAP, unseen injection tasks ({'raw' if v == 'raw' else 'template-masked'} text)",
                       "The clues the obedience model leans on most",
                       "Longer bars mean more influence on the prediction, averaged over all saved examples. Orange bars are pieces of the "
                       "benchmark's fixed attack sentence: a sign the model partly memorised that sentence.",
                       "Mean |SHAP| in log-odds per unit. Embedding dimensions and one-hot blocks are summed to one unit each. Orange = wrapper term "
                       "(raw) or the [WRAP] placeholder (masked).")
        ui.show(charts.shap_top(v, n), key="ex_shap")
        ui.source(f"part6/shap_global_stage2_xgb_{v}.csv")
        ui.original_figure("part6/fig_shap_summary_raw_vs_masked.png")
        ui.explain("shap_global")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            ui.chart_title("Where the obedience score comes from, by feature family", "Which kinds of clue do the work",
                           f"Words do the most ({f.sh_words}). Simply knowing the agent is second ({f.sh_agent}). The hand-built counts barely register ({f.sh_flags}).",
                           "Share of mean |SHAP| per family, net effect per row, raw vs template-masked.")
            ui.show(charts.family(), key="ex_fam")
            ui.source("part6/table_shap_by_family.csv")
            ui.explain("shap_family")
        with c2:
            ui.chart_title("Agent block: mean SHAP per configuration vs measured obey rate", "Does the model know which agents are gullible?",
                           f"Each dot is one AI agent. Dots rise from left to right, so the model pushes the score up for agents that really do obey "
                           f"more often (rank correlation {f.agent_rho}).",
                           f"Mean SHAP of the agent one-hot block against mean `obeyed`; Spearman {f.agent_rho}.")
            ui.show(charts.agent_effect(), key="ex_agent")
            ui.source("part6/table_agent_shap.csv")
            ui.explain("agent_effect")
        st.markdown("**The roles of each kind of input**")
        st.dataframe([
            {"Input": "Text features (TF-IDF words and word pairs)", "Share (raw)": f.sh_words, "What it contributes": "Action words such as “email”, “send”, “delete”; but also wrapper wording and the model's own name."},
            {"Input": "Agent identity", "Share (raw)": f.sh_agent, "What it contributes": "The agent's usual tendency to obey. Meaningful, but says nothing about the text."},
            {"Input": "Sentence embedding of the tool output", "Share (raw)": f.sh_out, "What it contributes": "Overall meaning of the tool output. Faithful as a block, not explainable feature by feature."},
            {"Input": "Sentence embedding of the user task", "Share (raw)": f.sh_task, "What it contributes": "Overall meaning of the user's request."},
            {"Input": "Hand-built flags", "Share (raw)": f.sh_flags, "What it contributes": "Almost all from the count of instruction cue words. The new-IBAN / new-email / new-link flags do close to nothing."},
            {"Input": "Tool name, task/output similarity", "Share (raw)": "under 1%", "What it contributes": "Negligible."},
        ], hide_index=True, width="stretch")

    with tabs[2]:
        sc = data.table("table_shortcut_check").set_index("text")
        ui.chart_title("Shortcut check: raw vs template-masked Stage 2 XGBoost", "Did the model just memorise the attack sentence?",
                       f"Hiding the fixed attack sentence cost almost nothing in score ({f.xgb_inj} to {f.xgb_inj_masked}). That sounds good, but the "
                       f"model then leaned on the placeholder that replaced it. So this test cannot prove the model understands attacks.",
                       "Same folds and settings for both versions. Wrapper terms decided from the data (masking removes ≥ 50% of the documents containing the term).")
        ui.cards([("PR-AUC, raw text", f.xgb_inj, "unseen injection tasks", "blue"), ("PR-AUC, masked text", f.xgb_inj_masked, f"cost of masking: {f.mask_cost}", "blue"),
                  ("Wrapper share of word SHAP, raw", f.wrap_raw, f"{f.wrap_top20} of the top 20 words", "warn"),
                  ("[WRAP] share of word SHAP, masked", f.wrap_masked, "the placeholder became a clue", "warn")])
        st.write("")
        t = sc.T.copy()
        t.index.name = "Measure"
        st.dataframe(t.round(3).reset_index().astype(str), hide_index=True, width="stretch")
        ui.source("part6/table_shortcut_check.csv", "part6/table_stage2_results.csv")
        s1 = data.table("table_stage1_raw_vs_masked")
        st.markdown("**Stage 1 on raw and masked text** (out-of-fold, unseen user tasks)")
        st.dataframe(s1.round(4), hide_index=True, width="stretch")
        ui.source("part6/table_stage1_raw_vs_masked.csv")
        ui.note(f"A second leak: the wrapper names the model it addresses (“to you, GPT-4”). Words such as “gpt” and “claude” carry about {f.name_share} of "
                "the word-level influence in both versions, telling the model which agent is running even when the agent feature is removed.", warn=True)
        ui.explain("shortcut")

    with tabs[3]:
        opts = {"Stage 1: injection present? (raw text)": ("table_lr_coefficients_stage1_raw", False),
                "Stage 1, template-masked text": ("table_lr_coefficients_stage1_masked", False),
                "Stage 2: will the agent obey? (raw text, agent terms left out)": ("table_lr_coefficients_stage2_raw", True),
                "Stage 2, template-masked text (agent terms left out)": ("table_lr_coefficients_stage2_masked", True)}
        which = st.selectbox("Model", list(opts), key="ex_lr")
        name, drop = opts[which]
        ui.chart_title(f"Logistic Regression: strongest coefficients · {which}", "The simple model's strongest word weights",
                       "Orange words raise the score when they appear, blue words lower it. Look at what kinds of words these are: several are "
                       "everyday words or fragments specific to this benchmark.",
                       "Twelve largest positive and eight largest negative coefficients (log-odds per unit TF-IDF). Final models fitted on all rows, for reading only.")
        ui.show(charts.coefficients(name, drop_agents=drop), key="ex_coef")
        ui.source(f"part6/{name}.csv")
        ui.original_figure("part6/fig_lr_coefficients.png")
        ui.explain("lr_coefs")

    with tabs[4]:
        ui.chart_title("Faithfulness: removing the words the explanation points to", "Do the explanations tell the truth about the model?",
                       f"If the orange line is far above the grey line, the explanation picked words the model truly depends on. Removing the top 5 "
                       f"words lowers the score by {f.fx_top}; removing 5 random words lowers it by {f.fx_rnd}.",
                       "Mean drop in the Stage 2 logit on HIGH RISK alerts, out-of-fold models. XGBoost: feature switch-off (n = 1,500). LR: words deleted from the text (n = 600).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**XGBoost Stage 2** (word features switched off)")
            ui.show(charts.faithfulness("XGBoost"), key="ex_fx")
        with c2:
            st.markdown("**Logistic Regression Stage 2** (words deleted from the text)")
            ui.show(charts.faithfulness("Logistic"), key="ex_fl")
        fa = data.table("table_faithfulness")
        st.dataframe(fa.round(3), hide_index=True, width="stretch")
        ui.source("part6/table_faithfulness.csv")
        ui.original_figure("part6/fig_faithfulness.png")
        ui.explain("faithfulness")

    with tabs[5]:
        ui.chart_title("Stage 2 XGBoost, raw text: SHAP per tool output (5,000 sampled exposure rows)", "How each top clue pushed individual predictions",
                       "One dot per example. Dots to the right raised the predicted chance of obeying, dots to the left lowered it. The agent row is "
                       "the widest: it matters most and in both directions.",
                       "Beeswarm of the 14 top units. Colour = feature value where one exists; grey rows are summed blocks.")
        ui.png("part6/fig_shap_beeswarm_raw.png", "Saved notebook figure. Per-row SHAP values were not stored, so this plot cannot be redrawn interactively.")
        ui.explain("shap_beeswarm")
