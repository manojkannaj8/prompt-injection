"""Every measured number quoted in the explanations, read from the saved files and formatted once."""
from functools import lru_cache
from types import SimpleNamespace

from core import data, rules, stats


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


@lru_cache(maxsize=None)
def facts():
    from scipy.stats import spearmanr
    s = stats.overview()
    f = SimpleNamespace()
    # volumes
    f.runs, f.configs, f.suites = f"{s['n_runs']:,}", str(s["n_configs"]), str(s["n_suites"])
    f.attacked_runs, f.clean_runs = f"{s['n_attacked_runs']:,}", f"{s['n_clean_runs']:,}"
    f.attack_styles = str(s["n_attack_styles"])
    f.outputs, f.injected, f.harmless = f"{s['n_tool_outputs']:,}", f"{s['n_injected_outputs']:,}", f"{s['n_harmless_outputs']:,}"
    f.exposure, f.obeyed = f"{s['n_exposure']:,}", f"{s['n_obeyed']:,}"
    f.obey_pct, f.obey3 = pct(s["obey_rate"]), f"{s['obey_rate']:.3f}"
    f.agents = str(s["n_agents_stage2"])
    f.alerts, f.gate_n = f"{s['n_alerts']:,}", f"{s['n_gate']:,}"
    f.t_high, f.p_log = f"{rules.T_HIGH:.2f}", f"{rules.P_INJ_LOG}"
    # decisions
    L = s["levels"]
    for k, key in [("high", rules.HIGH), ("susp", rules.SUSPICIOUS), ("safe", rules.SAFE)]:
        setattr(f, f"n_{k}", f"{L[key]['n']:,}")
        setattr(f, f"{k}_obeyed", f"{L[key]['obeyed']:,}")
        setattr(f, f"{k}_rate", pct(L[key]["obey_rate"]))
        setattr(f, f"{k}_share", pct(L[key]["share_of_obeyed"]))
        setattr(f, f"gate_{k}", f"{s['decisions_gate'][key]:,}")
    # Stage 1
    s1 = data.table("table_stage1_raw_vs_masked").set_index("text")
    f.s1_roc = f"{s1.loc['raw', 'ROC-AUC unseen user task']:.4f}"
    f.s1_roc_masked = f"{s1.loc['masked', 'ROC-AUC unseen user task']:.4f}"
    f.s1_recall, f.s1_fa = pct(s1.loc["raw", "recall at p>=0.5"], 2), pct(s1.loc["raw", "false alarms at p>=0.5"])
    tm = data.table("table_threshold_metrics")
    t1, tg, tb = tm.iloc[0], tm.iloc[1], tm.iloc[2]
    f.s1_prec, f.s1_rec = pct(t1.precision), pct(t1.recall)
    f.s1_tp, f.s1_fp, f.s1_fn, f.s1_tn = f"{t1.TP:,}", f"{t1.FP:,}", f"{t1.FN:,}", f"{t1.TN:,}"
    f.g_prec, f.g_rec, f.g_fpr = pct(tg.precision), pct(tg.recall), pct(tg["false positive rate"])
    f.g_tp, f.g_fp, f.g_fn, f.g_tn = f"{tg.TP:,}", f"{tg.FP:,}", f"{tg.FN:,}", f"{tg.TN:,}"
    f.b_blocked, f.b_rec, f.b_prec = f"{tb.TP + tb.FP:,}", pct(tb.recall), pct(tb.precision)
    f.b_wasted = pct(tb.FP / (tb.TP + tb.FP), 0)
    # Stage 2
    P = lambda m, p, t="raw", c="PR-AUC": stats.pr_auc(m, p, t, c)  # noqa: E731
    inj, usr, att, agt, rnd = ("Unseen injection task", "Unseen user task", "Unseen attack style",
                               "Unseen agent (leave one out)", "Random split (leaky)")
    f.base_inj, f.lr_inj, f.xgb_inj = f"{P('Model-only lookup', inj):.3f}", f"{P('Logistic Regression', inj):.3f}", f"{P('XGBoost', inj):.3f}"
    f.xgb_saved = f"{P('XGBoost (saved Part 3 scores)', inj):.3f}"
    f.xgb_saved_roc = f"{P('XGBoost (saved Part 3 scores)', inj, c='ROC-AUC'):.3f}"
    f.xgb_inj_roc = f"{P('XGBoost', inj, c='ROC-AUC'):.3f}"
    f.xgb_worst, f.xgb_best = f"{P('XGBoost', inj, c='PR-AUC worst fold'):.3f}", f"{P('XGBoost', inj, c='PR-AUC best fold'):.3f}"
    f.base_rnd, f.lr_rnd, f.xgb_rnd = f"{P('Model-only lookup', rnd):.3f}", f"{P('Logistic Regression', rnd):.3f}", f"{P('XGBoost', rnd):.3f}"
    f.lr_usr, f.xgb_usr = f"{P('Logistic Regression', usr):.3f}", f"{P('XGBoost', usr):.3f}"
    f.lr_att, f.xgb_att = f"{P('Logistic Regression', att):.3f}", f"{P('XGBoost', att):.3f}"
    f.lr_agt, f.xgb_agt = f"{P('Logistic Regression', agt):.3f}", f"{P('XGBoost', agt):.3f}"
    f.rnd_gap = f"{P('XGBoost', rnd) - P('XGBoost', inj):.2f}"
    f.xgb_inj_masked, f.xgb_att_masked = f"{P('XGBoost', inj, 'masked'):.3f}", f"{P('XGBoost', att, 'masked'):.3f}"
    f.mask_cost = f"{P('XGBoost', inj) - P('XGBoost', inj, 'masked'):.3f}"
    f.refit_rho = f"{data.summary()['refit_vs_saved_spearman']:.3f}"
    # replay
    f.asr_before, f.asr_after = pct(s["asr_pooled_before"]), pct(s["asr_pooled_after"])
    f.task_before, f.task_after = pct(s["task_pooled_before"]), pct(s["task_pooled_after"])
    g = s["gpt4o"]
    f.g4_asr_before, f.g4_asr_after = pct(g["asr_before"]), pct(g["asr_after"])
    f.g4_task_before, f.g4_task_after = pct(g["task_before"]), pct(g["task_after"])
    f.g4_clean, f.g4_attacked = str(g["n_clean"]), str(g["n_attacked"])
    f.replay_agents, f.lose5 = str(s["n_replay_agents"]), str(s["n_agents_lose_5"])
    f.task_median = f"{s['task_change_median']:.1f}"
    f.loo_after, f.loo_worst, f.loo_t = pct(s["loo_asr_after"]), f"{abs(s['loo_worst_task_change']):.1f}", f"{s['loo_threshold']:.2f}"
    rp = data.replay_all().set_index("config")
    worst = rp["task change (pts)"].nsmallest(2)
    f.worst1, f.worst1_pts = worst.index[0], f"{abs(worst.iloc[0]):.1f}"
    f.worst2, f.worst2_pts = worst.index[1], f"{abs(worst.iloc[1]):.1f}"
    det, tf_ = rp.loc["gpt-4o-2024-05-13-transformers_pi_detector"], rp.loc["gpt-4o-2024-05-13-tool_filter"]
    f.det_asr, f.det_task = pct(det["ASR no guard"]), pct(det["task no guard"])
    f.tf_asr, f.tf_task = pct(tf_["ASR no guard"]), pct(tf_["task no guard"])
    asr = data.table("table_attack_success_per_agent")
    f.asr_min, f.asr_max, f.asr_n = pct(asr.asr.min()), pct(asr.asr.max()), str(len(asr))
    # calibration, over-defence
    cal = data.table("table_calibration")
    f.ece_raw, f.ece_cal = f"{cal.ECE.iloc[0]:.3f}", f"{cal.ECE.iloc[1]:.3f}"
    od = data.table("table_over_defence")
    c1, c2 = od.columns[2], od.columns[3]
    f.od_all_s1, f.od_all_guard, f.od_all_n = pct(od[c1][0], 2), pct(od[c2][0], 2), f"{od.n[0]:,}"
    f.od_clean_s1, f.od_clean_guard, f.od_clean_n = pct(od[c1][2], 2), pct(od[c2][2], 2), f"{od.n[2]:,}"
    f.ni_s1, f.ni_avg, f.ni_easy, f.ni_n = pct(od[c1][3], 2), pct(od[c2][3], 2), pct(od[c2][4], 2), str(od.n[3])
    # explanations
    fam = data.table("table_shap_by_family").set_index("Unnamed: 0")
    sh = lambda k, c="raw share": pct(fam.loc[k, c], 0)  # noqa: E731
    f.sh_words, f.sh_agent, f.sh_out, f.sh_task, f.sh_flags = sh("words"), sh("agent"), sh("tool output meaning"), sh("user task meaning"), sh("flags")
    f.sh_emb = pct(fam.loc["tool output meaning", "raw share"] + fam.loc["user task meaning", "raw share"], 0)
    f.sh_words_m, f.sh_flags_m = sh("words", "masked share"), sh("flags", "masked share")
    sc = data.table("table_shortcut_check").set_index("text")
    f.wrap_raw, f.wrap_masked = pct(sc.loc["raw", "wrapper share of word SHAP"], 0), pct(sc.loc["masked", "wrapper share of word SHAP"], 0)
    f.name_share = pct(sc.loc["raw", "model-name share of word SHAP"], 0)
    f.wrap_top20 = str(int(sc.loc["raw", "wrapper terms in top 20 words"]))
    G = data.table("shap_global_stage2_xgb_raw").set_index("unit")
    f.cue_shap = f"{G.mean_abs_shap['n_cue_words']:.2f}"
    f.novel_max = f"{max(G.mean_abs_shap[k] for k in ['n_iban_new', 'n_email_new', 'n_url_new']):.4f}"
    fa = data.table("table_faithfulness")
    x5 = fa[(fa["words removed"] == 5)]
    fx, fl = x5[x5.model.str.startswith("XGBoost")].iloc[0], x5[x5.model.str.startswith("Logistic")].iloc[0]
    f.fx_top, f.fx_rnd = f"{fx['drop, top-attributed']:.2f}", f"{fx['drop, random']:.2f}"
    f.fl_top, f.fl_rnd = f"{fl['drop, top-attributed']:.2f}", f"{fl['drop, random']:.2f}"
    ag = data.table("table_agent_shap")
    f.agent_rho = f"{spearmanr(ag.mean_agent_shap, ag.obey_rate)[0]:.2f}"
    f.agent_low, f.agent_high = ag.config.iloc[0], ag.config.iloc[-1]
    f.agent_high_rate = pct(ag.obey_rate.iloc[-1])
    return f
