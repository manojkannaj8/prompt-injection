"""Where the research artefacts live. The UI only ever reads them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the project folder (holds README.md, part6/, ui/)
P6 = ROOT / "part6"
UI = ROOT / "ui"
CACHE = UI / "cache"                                # the only folder the UI writes to

# name shown in the UI -> path. Names are relative on purpose: absolute paths are never displayed.
FILES = {
    "runs_index.parquet": ROOT / "runs_index.parquet",
    "samples_features.parquet": ROOT / "samples_features.parquet",
    "stage2_exposure.parquet": ROOT / "stage2_exposure.parquet",
    "risk_scores.parquet": ROOT / "risk_scores.parquet",
    "notinject_scores.parquet": ROOT / "notinject_scores.parquet",
    "replay_all_configs.csv": ROOT / "replay_all_configs.csv",
    "replay_leave_one_agent_out.csv": ROOT / "replay_leave_one_agent_out.csv",
    "oof_xgb_fold_inj.npy": ROOT / "oof_xgb_fold_inj.npy",
    "fig_security_vs_usefulness.png": ROOT / "fig_security_vs_usefulness.png",
    "part6/part6_summary.json": P6 / "part6_summary.json",
    "part6/alerts_with_reasons.parquet": P6 / "alerts_with_reasons.parquet",
    "part6/alerts_examples.csv": P6 / "alerts_examples.csv",
}
TABLES = ["table_stage2_results", "table_shap_by_family", "table_shortcut_check", "table_agent_shap",
          "table_stage1_raw_vs_masked", "table_decisions_by_output_type", "table_decision_levels",
          "table_faithfulness", "table_calibration", "table_threshold_metrics", "table_over_defence",
          "table_attack_success_per_agent", "table_label_balance", "shap_global_stage2_xgb_raw",
          "shap_global_stage2_xgb_masked", "table_lr_coefficients_stage1_raw", "table_lr_coefficients_stage1_masked",
          "table_lr_coefficients_stage2_raw", "table_lr_coefficients_stage2_masked"]
for _t in TABLES:
    FILES[f"part6/{_t}.csv"] = P6 / f"{_t}.csv"
FIGURES = ["fig_shap_summary_raw_vs_masked", "fig_shap_beeswarm_raw", "fig_shap_by_family", "fig_lr_coefficients",
           "fig_faithfulness", "fig_pr_roc_unseen_injection_task", "fig_random_vs_grouped_splits",
           "fig_attack_success_per_agent", "fig_leave_one_agent_out", "fig_calibration", "fig_confusion_matrices",
           "fig_over_defence"]
for _f in FIGURES:
    FILES[f"part6/{_f}.png"] = P6 / f"{_f}.png"
for _f in ["fig_architecture", "fig_pipeline", "fig_decision_rule"]:
    FILES[f"report_figures/{_f}.png"] = ROOT / "report_figures" / f"{_f}.png"
# optional: out-of-fold scores of the Part 6 refits (the README says this folder may be deleted)
OPTIONAL = {f"part6/cache/oof_{m}_raw_inj.npy": P6 / "cache" / f"oof_{m}_raw_inj.npy" for m in ["base", "lr", "xgb"]}


def path(name):
    return FILES.get(name) or OPTIONAL[name]


def available(name):
    return path(name).exists()


def missing():
    """Display names of expected artefacts that are not on disk."""
    return sorted(n for n, p in FILES.items() if not p.exists())
