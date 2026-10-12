"""On-demand explanations of every dashboard component.

Each entry answers the same six questions, once for a beginner (B) and once for a technical reader (T):
  1 what am I looking at  2 why it matters  3 how it works  4 how to interpret it
  5 what this project's result says  6 what cannot be concluded
"""
from functools import lru_cache

from .facts import facts


@lru_cache(maxsize=None)
def _all():
    f = facts()
    E = {}

    def add(key, title, B, T):
        assert len(B) == 6 and len(T) == 6, key
        E[key] = {"title": title, "B": B, "T": T}

    # ------------------------------------------------------------------ overview
    add("data_volume", "How much data the project used", (
        f"Counts of what was analysed: {f.runs} recorded test runs of AI agents, and the {f.outputs} pieces of text those agents received from their tools.",
        "Every other number on the dashboard is a share of one of these counts, so you need to know which count a percentage belongs to.",
        "The runs come from a public benchmark called AgentDojo. Each run is one agent trying one task. Each time the agent used a tool (read an email, listed files), the text that came back was saved as one row.",
        "Bigger counts mean more evidence. 'Exposure rows' are the important subset: the first time an agent met a planted instruction in an attacked run.",
        f"{f.attacked_runs} runs were attacked and {f.clean_runs} were clean. The agents met a planted instruction for the first time {f.exposure} times, and followed it {f.obeyed} times ({f.obey_pct}).",
        "These are saved benchmark runs, not live traffic. The clean runs are few, so anything measured on them is less precise."),
        (f"Row counts of runs_index.parquet ({f.runs} runs, {f.configs} agent configurations, {f.suites} suites), risk_scores.parquet ({f.outputs} tool outputs) and stage2_exposure.parquet ({f.exposure} exposure rows).",
         "Each metric has its own denominator: Stage 1 uses all tool outputs, Stage 2 only exposure rows, replay uses runs.",
         "Tool outputs were extracted per run; an exposure row is the first injected output of an attacked run. `obeyed` is valid only there.",
         f"Base rate on exposure rows is {f.obey3}; that is the PR-AUC of a random scorer for Stage 2.",
         f"{f.attacked_runs} attacked / {f.clean_runs} clean runs; {f.injected} injected and {f.harmless} harmless outputs; {f.configs} configurations of which {f.agents} have exposure rows.",
         "The raw run files and the Parts 1-4 code are not in the folder, so these counts rest on the saved tables. The feature table is five rows shorter than the older samples.parquet; the reason is not recorded."))

    add("decisions", "HIGH RISK, SUSPICIOUS and SAFE counts", (
        "How many pieces of tool text the guard would put in each of its three boxes: HIGH RISK (stop), SUSPICIOUS (let through but write it down) and SAFE (let through).",
        "A guard that stops everything is useless and a guard that stops nothing is pointless. The split shows how selective this one is.",
        f"Each text gets a risk score between 0 and 1. A score of {f.t_high} or more is HIGH RISK. Below that, if an injection is still likely, it is SUSPICIOUS. Everything else is SAFE.",
        "A large SUSPICIOUS box is expected here: most of these texts do contain a planted instruction, but the agent reading them usually ignores it.",
        f"Of the {f.exposure} first encounters with a planted instruction, {f.n_high} were HIGH RISK, {f.n_susp} SUSPICIOUS and {f.n_safe} SAFE. Half of the HIGH RISK ones ({f.high_rate}) were really obeyed; only {f.susp_rate} of the SUSPICIOUS ones were.",
        "HIGH RISK is a threshold rule, not proof of an attack. Half of the HIGH RISK texts were not obeyed, and some obeyed attacks were only logged."),
        (f"Decision counts under the documented rule: HIGH RISK if risk ≥ {f.t_high}; else SUSPICIOUS if P(injection) ≥ {f.p_log}; else SAFE.",
         "Shows how the two-stage risk separates 'injection present' from 'injection likely to work'.",
         "risk = p_inj × p_obey from risk_scores.parquet. p_inj is the out-of-fold Stage 1 score; p_obey is the calibrated out-of-fold XGBoost score on exposure rows and the agent's obey rate elsewhere.",
         f"Two denominators are offered: exposure rows ({f.exposure}), where `obeyed` is valid, and all outputs up to the exposure point ({f.gate_n}), which is what the guard actually sees.",
         f"Exposure rows: {f.n_high} / {f.n_susp} / {f.n_safe}. Obey rate per level {f.high_rate} / {f.susp_rate} / {f.safe_rate}; HIGH RISK holds {f.high_share} of all obeyed attacks.",
         f"The 0.20 threshold was recovered by reproducing replay_all_configs.csv, not stored by Parts 1-5. The SUSPICIOUS rule was first written as code in Part 6."))

    add("stage1_perf", "Stage 1: spotting a planted instruction", (
        "How well the first model answers a yes/no question: does this piece of tool text contain a planted instruction?",
        "If the first step misses an injection, the rest of the system never looks at it.",
        "The model learned which words and word pairs tend to appear in injected text and which appear in normal tool data, then gives each new text a score from 0 to 1.",
        "ROC-AUC runs from 0.5 (coin flip) to 1.0 (perfect ranking). Recall is the share of real injections caught; the false-alarm rate is the share of harmless texts flagged.",
        f"On tasks it had never seen, the detector reached ROC-AUC {f.s1_roc}, caught {f.s1_recall} of injections and raised a false alarm on {f.s1_fa} of harmless texts.",
        "This near-perfect score is partly because the benchmark is easy: injections are English sentences sitting inside structured data. Do not expect the same on free-flowing text."),
        ("Out-of-fold Stage 1 Logistic Regression (TF-IDF uni+bigrams, tool one-hot) against `has_injection` on all tool outputs.",
         "Stage 1 gates Stage 2 through the product; its false negatives are unrecoverable.",
         "Grouped 5-fold by user task (Part 6 refit) and by injection task (saved scores used for risk).",
         f"Two score sets exist. Part 6 refit, unseen user task: ROC-AUC {f.s1_roc}, recall {f.s1_recall} and false alarms {f.s1_fa} at 0.5. Saved scores in risk_scores.parquet: precision {f.s1_prec}, recall {f.s1_rec}.",
         f"Masking the attack template leaves ROC-AUC at {f.s1_roc_masked}. Strongest coefficients are ordinary words ('todo', 'the', 'should'), so it mostly detects prose inside structured data.",
         "Saturated benchmark with one dominant wrapper. The detector has not been compared with an off-the-shelf detector on the same samples."))

    add("stage2_perf", "Stage 2: predicting whether the agent will obey", (
        "How well the second model predicts something harder: given that a planted instruction is present, will this particular AI agent actually follow it?",
        "Most planted instructions are ignored. Stopping the agent only when it is likely to obey keeps it useful.",
        "The model looks at the words of the text, a numeric summary of its meaning, a few counts, and which agent is reading it, then outputs a chance of obeying.",
        f"PR-AUC is a score from 0 to 1 for how well the model puts the truly dangerous cases at the top of its list. Pure guessing would score {f.obey3} here, because {f.obey_pct} of cases were obeyed.",
        f"Knowing only which agent is running scores {f.base_inj}. Adding the content raises that to {f.lr_inj} (Logistic Regression) and {f.xgb_inj} (XGBoost). The content clearly helps.",
        "0.66 is far from 1.0: the model is useful but often wrong. The score also moves a lot between test groups, so small gaps between models are not meaningful."),
        ("Stage 2 PR-AUC on exposure rows, grouped 5-fold with whole injection tasks held out.",
         "Answers RQ1: what content adds beyond agent identity.",
         "Model-only lookup = agent's obey rate in the training folds. LR on TF-IDF + tool + flags + agent. XGBoost adds MiniLM embeddings (300 trees, depth 6, lr 0.1).",
         f"Chance level is the base rate {f.obey3}. Compare models within one protocol only.",
         f"Lookup {f.base_inj}, LR {f.lr_inj}, XGBoost refit {f.xgb_inj} (saved Part 3 scores: {f.xgb_saved}, ROC-AUC {f.xgb_saved_roc}). Refit vs saved Spearman {f.refit_rho}.",
         f"XGBoost ranges {f.xgb_worst} to {f.xgb_best} across folds. One fixed hyperparameter setting, no search. The fine-tuned encoder is not built, so the model comparison is incomplete."))

    add("attack_before_after", "Attack success before and after the guard", (
        "The share of attacks that worked when nothing was guarding the agent, next to the share that would still have worked with this guard switched on.",
        "This is the security benefit: how many successful attacks the guard would have stopped.",
        "The saved runs are 'replayed'. For each attacked run we check whether the guard's risk score reached the threshold before the harmful step. If so, that attack counts as stopped.",
        "Lower is better. The gap between the two numbers is what the guard removes.",
        f"Across {f.replay_agents} agents, attack success falls from {f.asr_before} to {f.asr_after}. For GPT-4o alone it falls from {f.g4_asr_before} to {f.g4_asr_after}.",
        "This is a replay of recorded runs, not a live test. A blocked agent might have retried or failed differently, and attackers who knew about the guard could write different text."),
        (f"Attack success rate (ASR) on `important_instructions` runs: `security == True`, and with the guard additionally `gate < {f.t_high}`.",
         "Primary security outcome of the replay analysis (RQ3).",
         "gate = max risk over tool outputs up to and including the exposure step. Pooled figure is weighted by attacked runs per agent.",
         "Compare per agent as well as pooled: pooled rates are dominated by agents with more runs.",
         f"Pooled {f.asr_before} → {f.asr_after} at 0.20 (replay_all_configs.csv). Leave-one-agent-out guard at {f.loo_t}: {f.loo_after}. GPT-4o {f.g4_asr_before} → {f.g4_asr_after}.",
         "Counterfactual estimate on saved trajectories; no adaptive attacks; scores are out-of-fold but the threshold in this table was fixed on the same runs."))

    add("task_before_after", "Normal task success before and after the guard", (
        "How often the agent finished an ordinary, attack-free task without the guard, and how often it still would with the guard on.",
        "A guard that blocks honest work is a cost. This number is the price paid for the security.",
        "In clean runs there is no attack, so any block is a mistake. A task counts as done only if the agent finished it and the guard never stepped in.",
        "The closer the two numbers, the less the guard gets in the way.",
        f"For GPT-4o, task success goes from {f.g4_task_before} to {f.g4_task_after}. For most agents nothing changes, but {f.lose5} agents lose more than 5 points: {f.worst1} ({f.worst1_pts}) and {f.worst2} ({f.worst2_pts}).",
        f"Each agent has only about {f.g4_clean} clean runs, so one or two runs move the percentage noticeably. Small differences here cannot be trusted."),
        (f"Utility on clean runs: `utility == True`, and with the guard `gate < {f.t_high}`.",
         "The usefulness side of the security/usefulness trade-off.",
         "Same replay as attack success, on the non-attacked runs of each configuration.",
         "Report with its interval: n ≈ 124 to 132 clean runs per agent gives a Wilson 95% interval of roughly ±8 points.",
         f"GPT-4o {f.g4_task_before} → {f.g4_task_after}; median change 0.0 points over {f.replay_agents} agents; {f.lose5} agents lose more than 5 points ({f.worst1} −{f.worst1_pts}, {f.worst2} −{f.worst2_pts}).",
         "Agents with a high base obey rate cross 0.20 on moderate Stage 1 scores, so harmless outputs get blocked. Per-agent thresholds are listed as future work, not built."))

    # ------------------------------------------------------------------ risk analysis
    add("live_result", "Your live result", (
        "The dashboard's verdict on the text you entered: two probabilities, the risk score made from them, and one of three decisions.",
        "It shows the whole chain from text to decision on an example you chose yourself.",
        f"Stage 1 reads your text and estimates the chance it contains a planted instruction. That is multiplied by how often the chosen agent obeyed planted instructions in the saved runs. A product of {f.t_high} or more is HIGH RISK.",
        "A high injection probability with a low risk score means: this looks planted, but this agent rarely falls for it.",
        "The injection probability is a real model prediction for your text. The obedience probability is the agent's historical average, not a judgement of your text.",
        "The detector only knows benchmark-style tool outputs. On ordinary prose it can be badly wrong. A decision here is a classification by a research prototype, not proof of anything."),
        ("Live P(injection) from the refitted final Stage 1 detector; P(obey) = the agent's measured obey rate; risk and decision by the documented rule.",
         "Demonstrates genuine inference on arbitrary input without overstating what is available.",
         "TF-IDF (1-2 grams, min_df 5, sublinear) + tool one-hot → liblinear LR with balanced class weights, refitted exactly as Day 2 cell 2 and verified against the saved coefficient table and NotInject scores.",
         "For saved AgentDojo text the live score is in-sample (the final detector saw it); the saved out-of-fold score is the honest one.",
         "The content-based Stage 2 is not run on new text: the MiniLM embedding step and the Parts 1-4 flag code are not in the folder. The fallback is the same one risk_scores.parquet uses on non-exposure rows.",
         "No calibration study exists for Stage 1 on out-of-distribution text. An agent whose obey rate is below 0.20 can never reach HIGH RISK in live mode."))

    add("stage1_words", "Which words moved the injection score", (
        "The words in your text that pushed the injection score up, and those that pulled it down.",
        "It lets you see why the model said what it said, instead of just trusting a number.",
        "The model has a learned weight for each word it knows. A word's push is its weight times how prominent the word is in your text.",
        "Longer bars mean a stronger push. Up-pushing words look like instructions to the model; down-pushing words look like ordinary tool data.",
        "Notice that everyday words such as 'the', 'you' and 'should' often rank high. The model has learned that normal sentences are unusual inside tool data.",
        "These are associations the model learned from one benchmark. They are not reasons in the human sense and they do not prove the text is malicious."),
        ("Per-term contribution coefficient × TF-IDF value, in log-odds, for the live Stage 1 Logistic Regression.",
         "Exact additive decomposition of a linear model's logit; the same quantity Part 6 stores as `why_injection`.",
         "Contributions + tool coefficient + intercept sum to the logit shown; sigmoid of that is P(injection).",
         "Sign gives direction, magnitude gives strength. Bigrams and their unigrams are counted separately.",
         "Top coefficients are 'todo', 'the', 'should', 'delete', 'instead', 'previous': prose and imperative cues rather than semantics of harm.",
         "Linear attribution on correlated n-grams splits credit arbitrarily between them. This is not SHAP and not a causal statement."))

    add("saved_record", "A saved result and its reasons", (
        "One real piece of tool text from the saved benchmark runs, with the scores and reasons the research notebooks stored for it.",
        "It shows the complete system, including the second model, on a real example where we also know what the agent actually did.",
        "Nothing is recalculated. The scores were produced earlier by models that had not seen this example's attack task during training.",
        "Read the outcome line: it says whether the attack really succeeded in that run, so you can judge whether the decision was right.",
        "Numbers in brackets after each reason show how hard that reason pushed the obedience score up (+) or down (−).",
        "The reasons come from a refitted copy of the second model, not the exact model behind the saved score. The outcome is for the whole run, not this one text."),
        ("A row of risk_scores.parquet joined to part6/alerts_with_reasons.parquet.",
         "Out-of-fold scores with ground truth, so correctness of the decision is checkable.",
         "p_inj: Stage 1 out-of-fold (fold_inj). p_obey: saved XGBoost score through isotonic regression fitted on the other four folds. Reasons: TreeSHAP top 3 up / top 2 down of the Part 6 refit; Stage 1 reasons are coefficient × value.",
         "SHAP values are in log-odds relative to the fold model's base value.",
         f"The explained model is a refit (Spearman {f.refit_rho} with the saved scores), so reasons and score agree closely but not exactly.",
         "Stage 2 reasons exist only for exposure rows. `obeyed` is a per-run label: the attacker's goal was reached somewhere in the run."))

    # ------------------------------------------------------------------ explainability
    add("shap_global", "What the obedience model relies on most", (
        "A ranking of the inputs the second model leans on most when it predicts whether an agent will obey.",
        "If the model leans on the wrong things, good test scores can be misleading.",
        "A method called SHAP splits each prediction into contributions from each input. This chart averages the size of those contributions over all examples.",
        "Longer bar = more influence overall. It does not say whether the influence raises or lowers the score. Orange bars are words from the attack's fixed wording.",
        f"The single most influential input is which agent is running. Among words, {f.wrap_top20} of the top 20 come from the attack's fixed template, which is a warning sign.",
        "Influence is not causation. A word can be influential only because it always appears together with something else."),
        ("Mean |SHAP| per unit for the Stage 2 XGBoost, out-of-fold TreeSHAP over all exposure rows, raw and template-masked text.",
         "Global attribution; exposes shortcut features.",
         "Each row is explained by the fold model that did not see its injection task. Embedding dimensions and one-hot blocks are summed into one unit each (exact, by additivity).",
         "Units: log-odds of obeying. Wrapper terms are identified from the data: masking removes ≥ 50% of the documents containing the term.",
         f"agent > emb_out > emb_task > n_cue_words ({f.cue_shap}) > word features. Wrapper terms carry {f.wrap_raw} of word-level SHAP; model-name words about {f.name_share}.",
         "Explains the refit, not the saved model. TreeSHAP with path-dependent perturbation credits correlated features unevenly."))

    add("shap_beeswarm", "How each top input pushes individual predictions", (
        "A picture with one dot per example for each important input, showing whether that input pushed the obedience score up or down for that example.",
        "The ranking chart shows how much an input matters; this one shows in which direction.",
        "Dots to the right raised the predicted chance of obeying; dots to the left lowered it. Colour shows whether the input's own value was high or low.",
        "A wide spread means the input matters a lot and differently from case to case.",
        "The agent row is the widest: for careful agents it pushes far left, for easily fooled agents to the right.",
        "It shows 5,000 sampled examples as a saved image. Grey rows are groups of inputs with no single value to colour by."),
        ("SHAP beeswarm of the 14 top units, raw-text XGBoost, 5,000 sampled exposure rows (random state 0).",
         "Shows sign and dispersion of attributions, which mean |SHAP| hides.",
         "x = SHAP value in log-odds; colour = feature value where one exists (TF-IDF weight, flag count).",
         "Blocks (agent, embeddings) are grey because they aggregate many columns.",
         "Agent block spans roughly −4 to +2 log-odds; n_cue_words is monotone.",
         "Per-row SHAP values were not saved, so this is available only as the notebook's PNG."))

    add("shap_family", "Where the obedience score comes from, by type of input", (
        "The second model's influences grouped into five kinds: the words, which agent it is, the meaning of the tool text, the meaning of the user's request, and a few simple counts.",
        "It answers a fair question: is the model reading the content, or mostly just remembering which agents are gullible?",
        "The contributions of all inputs of one kind are added together, then shown as a share of the total.",
        "A bigger share means that kind of input does more of the work.",
        f"Words {f.sh_words}, agent identity {f.sh_agent}, meaning of the tool text {f.sh_out}, meaning of the user's task {f.sh_task}, simple counts {f.sh_flags}.",
        f"About a quarter of the score comes from simply knowing the agent. And {f.sh_emb} comes from 'meaning' summaries that cannot be explained in words."),
        ("Share of mean |SHAP| per feature family (net effect per row), raw vs template-masked.",
         "Answers RQ5: do explanations point at meaningful signals.",
         "SHAP values summed within family per row, absolute value averaged over rows, normalised.",
         "Families: TF-IDF words; agent one-hot; emb_out and emb_task (384-d MiniLM each); flags; similarity; tool.",
         f"Raw: words {f.sh_words}, agent {f.sh_agent}, emb_out {f.sh_out}, emb_task {f.sh_task}, flags {f.sh_flags}. Masked: words {f.sh_words_m}, flags {f.sh_flags_m} (cue-word count picks up the slack).",
         f"Novelty flags (new IBAN / email / URL) each have mean |SHAP| ≤ {f.novel_max}: the planned human-readable reasons do not materialise."))

    add("agent_effect", "How much the agent's identity moves the score", (
        "For each AI agent, how far the model shifts its prediction just because of which agent it is, next to how often that agent really obeyed.",
        "It checks whether the model's opinion of each agent matches reality.",
        "The model's average push for an agent is compared with that agent's real obey rate in the saved runs.",
        "Points rising from lower left to upper right mean the model has learned the agents' real tendencies.",
        f"The match is strong (rank correlation {f.agent_rho}). The agent pushed up hardest is {f.agent_high}, which obeyed {f.agent_high_rate} of the time.",
        "This means the model would have little to go on for a brand-new agent. The 'unseen agent' test measures exactly that and scores lower."),
        ("Mean SHAP of the agent one-hot block per configuration against the measured obey rate.",
         "Validates that the largest attribution block reflects ground truth rather than noise.",
         "SH['raw'].fam[:, agent] averaged per config; Spearman against obeyed mean.",
         "Monotone relation expected; outliers are agents whose content signal differs from their base rate.",
         f"Spearman {f.agent_rho} for XGBoost (0.94 for the LR agent coefficients, per the Part 6 notebook output).",
         "The wrapper text names the model ('to you, GPT-4'), so agent identity also leaks through words; leave-one-agent-out is therefore not a fully clean test."))

    add("shortcut", "The template shortcut check", (
        "A test of whether the model learned what an attack means, or only memorised the fixed sentence most attacks in this benchmark start with.",
        "A model that only recognises one sentence would be useless against an attacker who words things differently.",
        "The fixed attack sentence was replaced by a placeholder, [WRAP], and the model was trained and tested again.",
        "If the score barely drops, it looks as if the model did not need the wording. But you also have to check what it leans on instead.",
        f"The score dropped only slightly ({f.xgb_inj} to {f.xgb_inj_masked}). However, the placeholder itself then became one of the model's favourite clues, carrying {f.wrap_masked} of the word influence.",
        "So the test hid the wording of the template but not its presence. It cannot show that the model understands attacks. A stricter test is future work."),
        ("Raw vs template-masked Stage 2: PR-AUC and the share of word-level SHAP on wrapper terms.",
         "Answers RQ4: reliance on the fixed attack wrapper.",
         "`tool_output_masked` replaces the wrapper with a [WRAP] token; same folds and settings for both versions.",
         "Low PR-AUC cost alone is not evidence of robustness; inspect attributions of the masked model.",
         f"Raw: wrapper terms {f.wrap_raw} of word SHAP, {f.wrap_top20} of top 20 words. Masked: PR-AUC −{f.mask_cost}, 'wrap wrap' and 'wrap' rank 5th and 6th overall and carry {f.wrap_masked}. Unseen attack style: {f.xgb_att} → {f.xgb_att_masked}.",
         "The placeholder is a shortcut. A valid test deletes the wrapper without a marker or paraphrases it; neither has been run."))

    add("lr_coefs", "Logistic Regression: the strongest word weights", (
        "The words the simpler model treats as the strongest signs for and against, read directly from the model.",
        "This model is simple enough that its reasoning can be read off without any extra tool.",
        "It gives every word a weight. Positive weights raise the score when the word appears, negative weights lower it.",
        "Look at what kinds of words are at the top. Sensible warning words are reassuring; oddly specific words suggest memorisation.",
        "For spotting injections, the top words are 'todo', 'the', 'should', 'delete'. For predicting obedience, several top entries are fragments of the attack template or specific dates and account numbers from the benchmark.",
        "Weights are for this benchmark's vocabulary. A big weight on a benchmark-specific account number tells you about the benchmark, not about attacks in general."),
        ("Largest positive and negative coefficients of the final Logistic Regressions: Stage 1 (raw, masked) and Stage 2 (raw, masked).",
         "Transparent baseline to cross-check the SHAP story.",
         "Coefficients in log-odds per unit of TF-IDF value (L2, C=1). Stage 2 also has tool, flag and agent terms.",
         "Coefficient size depends on feature scale and frequency; compare within a model.",
         "Stage 1 raw: todo +9.7, the +6.0, should +5.1. Stage 1 masked: wrap +11.7, wrap wrap +11.0. Stage 2 raw top: 'gpt before', '2022 04', 'subject information': template fragments and benchmark artefacts.",
         "Collinear n-grams share weight arbitrarily. These are in-sample fits of the final models, used for reading, not for evaluation."))

    add("faithfulness", "Do the explanations tell the truth about the model?", (
        "A test of whether the words an explanation points to really are the ones driving the model.",
        "An explanation that sounds plausible but does not match the model is worse than none.",
        "Take the words the explanation ranks highest, remove them, and see how much the score falls. Then remove the same number of random words and compare.",
        "If removing the 'important' words lowers the score far more than removing random words, the explanation is faithful.",
        f"Removing the top 5 words lowers the obedience score by {f.fx_top} (on the model's internal scale), against {f.fx_rnd} for 5 random words. The explanations do point at what the model uses.",
        "Faithful means 'true to the model', not 'true about the world'. A faithful explanation of a model with a shortcut faithfully shows the shortcut."),
        ("Deletion test on HIGH RISK alerts: mean drop in the Stage 2 logit after removing the k top-attributed words vs k random present words.",
         "Standard faithfulness check for attribution methods.",
         "XGBoost: word features switched off, embeddings left unchanged, 1,500 alerts. LR: words deleted from the text and re-vectorised, 600 alerts. Out-of-fold models, k ∈ {1,3,5,10,20}.",
         "Report the gap to random and the share of alerts where top > random.",
         f"k=5: XGBoost {f.fx_top} vs {f.fx_rnd}; LR {f.fl_top} vs {f.fl_rnd} log-odds. Top beats random in more than 99% of alerts.",
         "For XGBoost the embeddings still encode the removed words, so the drop is a lower bound on a true text deletion. Not a human-grounded evaluation."))

    # ------------------------------------------------------------------ alerts
    add("alerts_table", "The alerts table", (
        "Every alert the guard would have raised on the saved runs, one per row, with its scores and the reasons behind it.",
        "This is what a security analyst would actually read. It makes the system's output inspectable case by case.",
        "A row appears whenever a tool text was judged HIGH RISK or SUSPICIOUS. Filters narrow the list; selecting a row opens the full details.",
        "The 'outcome' column tells you what really happened in that run, so you can find correct blocks, unneeded blocks, missed attacks and false alarms.",
        f"There are {f.alerts} alerts in total. Try filtering to 'harmless output (false alarm)' to see the cases where the guard was wrong about harmless text.",
        "The text shown is a short snippet. Outcomes describe the whole run. All of it is historical: nothing here is a live alert."),
        (f"part6/alerts_with_reasons.parquet: {f.alerts} in-gate tool outputs with decision ≠ SAFE.",
         "Per-alert audit trail: scores, decision, kind, outcome, Stage 1 and Stage 2 reasons.",
         "Server-side filtering and pagination; only the visible page is sent to the browser. Alert ID = row number in the saved file.",
         "`kind` separates exposure rows from harmless outputs in clean runs; `outcome` derives from `obeyed`.",
         f"{f.gate_high} HIGH RISK and {f.gate_susp} SUSPICIOUS alerts; harmless-output alerts carry no Stage 2 reasons.",
         "Reasons are from the refitted XGBoost. Snippets are cut to 230 characters around the first attack marker."))

    # ------------------------------------------------------------------ evaluation
    add("protocols", "Scores under different testing rules", (
        "The same models scored five different ways. Each way hides a different kind of thing from the model during training and then tests on it.",
        "A model can look brilliant if it is tested on things it has effectively already seen. The honest question is how it does on new things.",
        "'Random split' mixes everything, so near-copies of test examples sit in training. The other four hold back a whole group: a whole attack goal, a whole user task, a whole attack style, or a whole agent.",
        "Trust the 'unseen' bars. The random-split bars are shown only to demonstrate how much an unfair test flatters the model.",
        f"XGBoost scores {f.xgb_rnd} on the random split but {f.xgb_inj} when the attack goal is new: the random split overstates it by {f.rnd_gap}. On a new attack style, the two models are level ({f.lr_att} and {f.xgb_att}).",
        "Even the fair tests use one benchmark with repetitive attacks. They say nothing certain about attacks written specifically to get past this guard."),
        ("Stage 2 PR-AUC (and ROC-AUC) by evaluation protocol and model; whiskers show worst and best fold.",
         "Quantifies leakage and tests generalisation along four axes.",
         "Stratified random 5-fold vs grouped folds: fold_inj, fold_user, fold_attack, and leave-one-agent-out (28 folds, agent one-hot dropped).",
         "Primary result = unseen injection task. Random split is a comparison only.",
         f"Random {f.lr_rnd}/{f.xgb_rnd}; unseen injection task {f.lr_inj}/{f.xgb_inj}; unseen user task {f.lr_usr}/{f.xgb_usr}; unseen attack style {f.lr_att}/{f.xgb_att}; unseen agent {f.lr_agt}/{f.xgb_agt} (LR/XGBoost).",
         "Fold variance is large (unseen agent: 0.05 to 0.91), so differences of a few hundredths are within noise. No lookup baseline exists for an unseen agent."))

    add("pr_roc", "Precision-recall and ROC curves", (
        "Two curves per model that show every possible trade-off between catching dangerous cases and raising false alarms.",
        "A single threshold gives a single trade-off. The curves show whether a model is better across the board.",
        "Imagine sliding the alarm level from strict to lenient. At each setting, mark how many dangerous cases are caught and how many alarms are wrong. Joining the marks draws the curve.",
        "On the precision-recall chart, higher and further right is better. On the ROC chart, closer to the top-left corner is better. The area under each curve is the AUC number.",
        f"XGBoost's curve sits above Logistic Regression's, and both sit far above the agent-only baseline. Areas: {f.base_inj}, {f.lr_inj} and {f.xgb_saved}.",
        "ROC-AUC looks flattering when dangerous cases are rare, which is why this project reports PR-AUC first."),
        ("PR and ROC curves of the three Stage 2 scorers on unseen injection tasks, pooled out-of-fold scores.",
         "Threshold-free comparison; PR is the informative one at a 19% base rate.",
         "Computed from the saved out-of-fold score arrays; XGBoost is the saved Part 3 model, LR and lookup are Part 6 refits.",
         f"PR baseline = {f.obey3}; ROC baseline = diagonal.",
         f"PR-AUC {f.base_inj} / {f.lr_inj} / {f.xgb_saved}; ROC-AUC of the saved XGBoost {f.xgb_saved_roc}.",
         "Pooling folds mixes score scales across fold models; per-fold PR-AUC varies widely."))

    add("calibration", "Can the percentages be taken at face value?", (
        "A check of whether a predicted '30% chance of obeying' really corresponds to about 30 in 100 cases being obeyed.",
        "The risk score multiplies probabilities. That only makes sense if the probabilities mean what they say.",
        "Examples are sorted into ten groups by predicted chance. For each group, the average prediction is compared with what actually happened.",
        "Points on the diagonal line are perfect. Points below it mean the model was overconfident.",
        f"The raw model scores were too high. After a correction step called calibration, the average gap fell from {f.ece_raw} to {f.ece_cal}.",
        "This holds on average for this benchmark's data. It does not make any single prediction certain, and it would need re-checking on other data."),
        ("Reliability diagram (10 equal-size bins) and ECE of P(obey) on exposure rows, before and after isotonic regression.",
         "risk = p_inj × p_obey is only interpretable if p_obey is calibrated.",
         "Isotonic regression fitted on the other four folds and applied to the held-out fold (verified in Part 6: max difference 0).",
         "ECE uses 10 equal-width bins; lower is better.",
         f"ECE {f.ece_raw} → {f.ece_cal}; Brier 0.106 → 0.098; mean score 0.253 → 0.190 against a base rate of {f.obey3}.",
         "Calibrated in-distribution only. Stage 1 is not separately calibrated, and the product assumes the two scores can be multiplied."))

    add("confusion", "Right and wrong decisions, counted", (
        "A two-by-two count of the guard's decisions against what really happened: correct alarms, false alarms, misses, and correct passes.",
        "Percentages hide the two different ways of being wrong. This table shows both.",
        "Each case goes in one of four boxes depending on whether the guard flagged it and whether it was truly dangerous.",
        "A false positive is a harmless case that was flagged (annoying). A false negative is a dangerous case that was missed (harmful).",
        f"At the {f.t_high} threshold the guard correctly stopped {f.g_tp} attacks that would have worked, stopped {f.g_fp} that would have failed anyway, and missed {f.g_fn}.",
        "Half of the blocks were not needed. Whether that is acceptable depends on how costly a block is compared with a successful attack."),
        (f"Confusion counts at the decision thresholds: Stage 1 (p_inj ≥ 0.5 vs has_injection) and the guard (risk ≥ {f.t_high} vs obeyed).",
         "Makes precision/recall concrete and comparable with a Stage 1-only blocker.",
         "Stage 1 on all tool outputs; guard on exposure rows where `obeyed` is defined.",
         "Guard precision = share of blocks that prevented a real compromise.",
         f"Guard: TP {f.g_tp}, FP {f.g_fp}, FN {f.g_fn}, TN {f.g_tn}; precision {f.g_prec}, recall {f.g_rec}. Stage 1 alone as a blocker: {f.b_blocked} blocked, recall {f.b_rec}, precision {f.b_prec}.",
         "Accuracy is reported only as a secondary column; it is misleading at this class balance."))

    add("over_defence", "False alarms on harmless text", (
        "How often the system raises an alarm on text that contains no attack at all.",
        "A guard that cries wolf gets switched off. Low false alarms are what make it usable.",
        "The models were run on harmless tool outputs from the benchmark and on a separate set of harmless questions that deliberately contain scary-sounding words like 'ignore'.",
        "Lower is better. Compare the two columns: the first model alone flags more than the full guard blocks.",
        f"Over all harmless tool outputs, the first model flags {f.od_all_s1} and the guard blocks {f.od_all_guard}. On the {f.ni_n} tricky harmless questions, the guard blocks between {f.ni_avg} and {f.ni_easy}, depending on the agent.",
        f"The rate is not even. In attack-free runs it rises to {f.od_clean_s1} flagged and {f.od_clean_guard} blocked, because that harmless content is rare in the training data."),
        ("Over-defence rates: share of harmless texts with p_inj ≥ 0.5 and with risk ≥ 0.20, on AgentDojo harmless outputs and NotInject.",
         "Answers the second half of RQ4.",
         "NotInject risk uses p_inj × a fixed obey rate (average agent 0.19, most easily fooled 0.56).",
         "Distinct-text and clean-run rows control for repeated outputs.",
         f"All rows {f.od_all_s1} / {f.od_all_guard} (n={f.od_all_n}); clean runs {f.od_clean_s1} / {f.od_clean_guard} (n={f.od_clean_n}); NotInject {f.ni_s1} flagged, {f.ni_avg} to {f.ni_easy} blocked.",
         "The saved 'at most 0.9%' NotInject figure was at threshold 0.13. Clean-run slots are where the low rate is least reliable."))

    add("decision_levels", "What the SUSPICIOUS level buys", (
        "A comparison of two ways to run a guard: stop everything that looks planted, or stop only what is also likely to work.",
        "It is the main argument for having two models instead of one.",
        "Every first encounter with a planted instruction is put in its decision box, then we count how many in each box were really obeyed.",
        "Good: most of the obeyed attacks sit in HIGH RISK, and few sit in the boxes that were let through.",
        f"A simple detector would stop {f.b_blocked} texts to catch {f.b_rec} of working attacks, and {f.b_wasted} of those stops would be unnecessary. The two-stage guard stops {f.n_high}, about a third as many, and catches {f.high_share}.",
        f"The cost is real: {f.susp_obeyed} attacks that did work were only logged, not stopped."),
        ("Exposure rows by decision level with obey counts (table_decision_levels.csv) against a Stage 1-only blocker.",
         "Quantifies the selectivity gained from Stage 2.",
         "Decisions from saved risk scores; `obeyed` from the saved runs.",
         "Compare blocked volume and recall jointly.",
         f"HIGH RISK {f.n_high} ({f.high_rate} obeyed, {f.high_share} of all obeyed); SUSPICIOUS {f.n_susp} ({f.susp_rate}); SAFE {f.n_safe} ({f.safe_rate}).",
         f"{f.susp_obeyed} obeyed attacks fall in SUSPICIOUS and {f.safe_obeyed} in SAFE. Costs of a block and of a missed attack are not modelled."))

    add("attack_per_agent", "The same attack against different AI models", (
        "How often the benchmark's main attack succeeded against each AI model when nothing was defending it.",
        "It is the reason this project predicts obedience per agent: models differ enormously.",
        "The same planted instruction was put in front of each model in hundreds of runs, and the successes were counted.",
        "Longer bar = more easily fooled.",
        f"Across {f.asr_n} undefended models, the success rate ranges from {f.asr_min} to {f.asr_max}.",
        "This is one attack wording on one benchmark. A model that resists it may still fall for a different one."),
        ("Attack success of `important_instructions` on the undefended configurations.",
         "Motivation for a per-agent Stage 2.",
         "Mean of `security` per config, excluding configs with a built-in defence suffix.",
         "n = 629 or 949 attacked runs per model.",
         f"{f.asr_n} models, {f.asr_min} to {f.asr_max}.",
         "Benchmark-specific; several models are older releases."))

    # ------------------------------------------------------------------ security vs usefulness
    add("tradeoff_gpt4o", "Security versus usefulness on GPT-4o", (
        "A map of defences for one AI model. Each point is one way of protecting it: how many attacks still get through, and how many normal tasks still get done.",
        "A defence is only worth having if it stops attacks without stopping the agent from being useful.",
        "Left means fewer successful attacks. Up means more normal tasks completed. The line shows this project's guard as its alarm level is moved from strict to lenient.",
        "The best place to be is the top-left corner: few attacks, lots of work done.",
        f"With no defence, {f.g4_asr_before} of attacks succeed. The benchmark's built-in detector cuts that to {f.det_asr} but task success collapses to {f.det_task}. This guard reaches {f.g4_asr_after} while keeping task success at {f.g4_task_after}.",
        f"There are only {f.g4_clean} attack-free runs behind the task-success figure, so it is uncertain by several points. And all of this is a replay of saved runs, not a live deployment."),
        ("Attack success (x) vs clean-run utility (y) for GPT-4o: saved defence configurations, and the guard's replay across thresholds.",
         "The headline trade-off figure (RQ3).",
         "Saved setups are AgentDojo runs as recorded. Guard points are replayed: a run is blocked when gate ≥ t.",
         "Pareto view: up-left dominates. The sweep line is tuned on the same runs it is drawn on.",
         f"No defence {f.g4_asr_before}/{f.g4_task_before}; built-in detector {f.det_asr}/{f.det_task}; tool filter {f.tf_asr}/{f.tf_task}; guard at 0.20 {f.g4_asr_after}/{f.g4_task_after}. Saved held-out-threshold result: 5.7% / 71.5% (95% CI 63.0 to 78.8).",
         "The held-out threshold code is not in the folder. Wilson interval on 124 clean runs is about ±8 points; 71.5% and 72.6% are indistinguishable."))

    add("tradeoff_agents", "Before and after, for every agent", (
        "For each AI agent, one line from its attack success without the guard to its attack success with the guard.",
        "It shows whether the guard helps everywhere or only for a few agents.",
        "Grey dot: no guard. Blue dot: with the guard. The longer the line between them, the more attacks the guard removed.",
        "Agents that were easily fooled have the longest lines. Agents that already resisted have short ones because there was little to fix.",
        f"Pooled over {f.replay_agents} agents, attack success goes from {f.asr_before} to {f.asr_after}. Select an agent below the chart to see its numbers and its cost in normal tasks.",
        "The same threshold is used for every agent, which is why two of them lose noticeable task success."),
        (f"Per-configuration ASR without and with the guard at risk ≥ {f.t_high} (replay_all_configs.csv).",
         "Shows heterogeneity hidden by the pooled figure.",
         "Dumbbell per config; task change shown alongside.",
         "Sort order = undefended ASR.",
         f"Pooled {f.asr_before} → {f.asr_after}; median task change 0.0; {f.lose5} agents lose more than 5 points.",
         "Scores are out-of-fold by injection task, but each agent's own runs were in Stage 2 training; see the leave-one-agent-out result for the stricter test."))

    add("loo", "Protecting an agent the guard has never seen", (
        "The same before-and-after comparison, but each agent is protected by a guard that was trained without any data from that agent.",
        "A new AI model appears every few months. A guard that only works for models it trained on would always be out of date.",
        "One agent at a time is removed from training. The guard is then built from the rest and tested on the one that was left out.",
        "If results stay close to the normal ones, the guard generalises to new agents.",
        f"Attack success still falls from {f.asr_before} to {f.loo_after}, and no agent loses more than {f.loo_worst} points of task success.",
        "The attack text often names the model it is addressing, which gives the guard a hint about the unseen agent. So this test is easier than it looks."),
        ("Leave-one-agent-out replay (Day 2): Stage 2 LR without the agent feature, threshold chosen on the other agents' clean runs.",
         "Tests transfer to an unseen agent configuration.",
         f"28 fits; p_obey for non-exposure rows = mean obey rate of the other agents; threshold = lowest t costing ≤ 1 point of task success on the others ({f.loo_t} for every agent).",
         "Compare with the 0.20 replay per agent.",
         f"Pooled ASR {f.asr_before} → {f.loo_after}; worst task change −{f.loo_worst}; pooled Stage 2 PR-AUC {f.lr_agt}.",
         f"Model-name words carry about {f.name_share} of word SHAP, so agent identity leaks through the text."))

    add("threshold_sweep", "Moving the threshold", (
        "What would happen to attacks and to normal tasks if the HIGH RISK line were drawn somewhere other than 0.20.",
        "The threshold is a choice. This shows what the choice costs in each direction.",
        "The saved runs are replayed at every threshold from 0 to 1. Nothing is retrained; only the cut-off moves.",
        "A lower threshold blocks more: fewer attacks succeed, but more normal tasks get interrupted. A higher threshold does the opposite.",
        "For most agents there is a range around 0.20 where attacks are low and tasks are almost untouched. That is why 0.20 was used.",
        "This slider is a what-if. The project's configured threshold stays 0.20 and the saved results are not changed."),
        ("Replay of attack success and utility as functions of the risk threshold t ∈ [0, 1] for one configuration.",
         "Sensitivity of the operating point; reproduces the blue sweep in fig_security_vs_usefulness.png.",
         "Recomputed from risk_scores.parquet and runs_index.parquet with the notebooks' replay function; equals replay_all_configs.csv at t = 0.20 to 1e-16.",
         "The curve is tuned on the runs it is evaluated on; use the held-out result for an unbiased point.",
         "0.20 is the value that reproduces the saved replay for all 28 agents.",
         "Step-shaped because few runs sit near any given threshold. Not a recommendation to change the configured threshold."))
    return E


def get(key):
    return _all()[key]


def keys():
    return list(_all())
