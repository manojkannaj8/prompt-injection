# Prompt-injection defence dashboard

A local dashboard for the project's saved results. Built on 12 October 2026 on top of Parts 1-6.
It reads the research files and never changes them. The only folder it writes to is `ui/cache`.

## Start it

From the project folder (the one that holds `README.md` and `part6/`):

```
python run_dashboard.py
```

It opens http://localhost:8501. On Windows you can also double-click `run_dashboard.bat`.
Options: `--port 8600`, `--no-browser`. Stop it with Ctrl+C.

Alternative, from inside `ui/`: `streamlit run app.py`.

## Set-up

Built and tested on Python 3.13 (Windows 11). Other Python versions were not tested.

```
pip install -r ui/requirements.txt
```

The versions are pinned to the ones the dashboard was tested with (pandas is a narrow range, see Deploy). scikit-learn in particular must stay at 1.8.0 for the
shipped `ui/cache` detector to be used; with another version the dashboard refits it on first use.
`xgboost` and `shap` are not needed for the dashboard. They are needed only to re-run the Part 6 notebook.

## Deploy (Streamlit Community Cloud)

The code and the 49 saved result files it reads are published at https://github.com/manojkannaj8/prompt-injection.

1. https://share.streamlit.io → sign in with the GitHub account that owns the repository.
2. Create app → Deploy a public app from GitHub → repository `manojkannaj8/prompt-injection`, branch `main`, main file path `ui/app.py`.
3. Deploy. Python 3.13 or 3.14 both work.

Two things a host needs, both learned the hard way on the first deployment (12 October 2026):

- **Every pinned package must have a ready-made build for the host's Python.** The host picked Python 3.14 and
  `pandas 2.3.2` has none, so it tried to compile pandas and sat on "Your app is in the oven". pandas is now a
  range (`>=2.3.2,<2.4`) that resolves to a release with a 3.14 build.
- **`ui/.streamlit/config.toml` must not set `server.address`.** Streamlit loads the config next to the main script
  last, so it applies on the host too. Bound to localhost, a hosted app cannot be reached and never finishes starting.
  Local runs are bound to localhost by `run_dashboard.py` instead. There is only this one config file.
- **The config must not turn the file watcher off either.** The host uses it to pick up pushed changes; with it off, a push
  did not show until the app was rebooted. It is also switched off only by `run_dashboard.py`. Because that launcher
  runs without a watcher, restart it after editing code locally.

Vercel cannot host this app. Streamlit needs a long-running server with a WebSocket connection; Vercel runs short
serverless functions, and the Python packages alone exceed its function size limit.

What is published is controlled by the whitelist in `.gitignore`, which mirrors `ui/core/paths.py`. The notebooks,
embeddings, refitted XGBoost models, the written report, README.md, PROGRESS.md and RESULTS.md are not published.
The git object store is kept outside Google Drive (`C:/Users/HP/.gitdirs/prompt-injection.git`); the `.git` file in the
project folder points to it.

The first time the Risk analysis page (or Lesson 3) is opened on a new machine or a different scikit-learn version,
the Stage 1 detector is refitted from `samples_features.parquet`. Measured here: 33 to 64 seconds. After that it loads
from `ui/cache` in under a second. Deleting `ui/cache` is safe; it is rebuilt.

If this is the very first time Streamlit runs on the machine, it may ask for an email address in the terminal.
Press Enter to skip. `run_dashboard.py` starts Streamlit in headless mode, which avoids the question.

## What is live and what is historical

| Shown as | Meaning |
|---|---|
| LIVE MODEL PREDICTION | Stage 1 P(injection) computed now for text you typed. |
| HISTORICAL · SAVED RESULT | Loaded from the saved files. Nothing recomputed. |
| SAVED PART 3-5 RESULT / PART 6 REFIT | Which notebook a model result came from. |
| WHAT-IF · RECOMPUTED FROM SAVED SCORES | A threshold other than 0.20 applied to saved scores. The configured threshold does not change. |
| ILLUSTRATION · NOT A PREDICTION | Made-up numbers in the risk-score explainer. |

Live scoring covers **Stage 1 only**. The trained Stage 1 model was never saved, so the dashboard refits it with the
recipe of `ML project Day2.ipynb` cell 2 and checks the refit before use:

- every coefficient against `part6/table_lr_coefficients_stage1_raw.csv`: largest difference 4.4e-16
- the 339 saved NotInject scores in `notinject_scores.parquet`: largest difference 1.9e-06

Stage 2 cannot score new text. It needs MiniLM sentence embeddings and the flag-extraction code of Parts 1-4, and
neither is in this folder (`sentence-transformers` is not installed and no MiniLM model is cached). For new text the
dashboard uses the selected agent's measured obey rate as P(obey). That is the fallback the pipeline itself uses on
non-exposure rows of `risk_scores.parquet` and in the Day 2 NotInject check. A consequence, stated in the UI: an
agent whose obey rate is below 0.20 can never reach HIGH RISK in live mode.

There is no per-input SHAP for new text. SHAP reasons are shown only for saved alerts.

## Architecture

```
run_dashboard.py          launcher
ui/
  app.py                  entry point: navigation, sidebar, explanation-level switch
  components.py           look, cards, chart styling, explanation panel, risk breakdown, error wrapper
  core/                   read-only adapter to the research files. No Streamlit imports.
    paths.py              the list of saved files (relative names only)
    data.py               loaders; MissingArtefact when a file is absent
    rules.py              T_HIGH = 0.20, P_INJ_LOG = 0.5, risk(), decide()   (unchanged from the notebooks)
    stage1.py             refit + cache + verification + scoring of the Stage 1 detector, input validation
    replay.py             the notebooks' replay function, threshold sweep, Wilson interval
    stats.py              overview numbers, curves, calibration, threshold metrics
  content/                the words: facts (numbers read from files), explanations, lessons, glossary
  views/                  one module per page, plus shared charts, record detail and teaching widgets
  tests/                  unittest suites and the read-only check
  cache/                  Stage 1 detector (UI-owned, safe to delete)
  .streamlit/config.toml  the only config: dark theme, no tracebacks in the browser (no server.address, see Deploy)
```

Three layers, one direction: `views` → `content` / `components` → `core` → saved files.
Model inference lives in `core/stage1.py` and decision logic in `core/rules.py`; neither knows about rendering.
A change to a page cannot alter a result, because pages have no code path that writes to the research files.

Stack: Streamlit and Plotly. Both were already installed, there was no existing UI, and everything stays in Python
and the project's parquet/CSV formats. No external API calls are made; all charts render offline.

## Pages and their data sources

| Page | What it shows | Source files |
|---|---|---|
| Overview | Runs, agents, tool outputs, exposures; decision counts (two denominators); Stage 1 and Stage 2 performance; attack and task success before and after the guard | runs_index.parquet, risk_scores.parquet, stage2_exposure.parquet, replay_all_configs.csv, part6/table_stage1_raw_vs_masked.csv, table_threshold_metrics.csv, table_stage2_results.csv, table_decision_levels.csv |
| Risk analysis | Live Stage 1 scoring with word contributions; saved-example explorer (ten worked examples, or any run by number); risk-score explainer | samples_features.parquet (refit and text), risk_scores.parquet, part6/alerts_with_reasons.parquet, part6/alerts_examples.csv, notinject_scores.parquet |
| Alerts explorer | 27,040 saved alerts; filters for decision, agent, outcome, suite, risk range, text; sort; paging; detail view with reasons | part6/alerts_with_reasons.parquet, samples_features.parquet (full text) |
| Explainability | SHAP feature importance (raw and masked), feature families, agent effect, shortcut check, LR coefficients, faithfulness, beeswarm; the five discovered limitations | part6/shap_global_stage2_xgb_*.csv, table_shap_by_family.csv, table_agent_shap.csv, table_shortcut_check.csv, table_lr_coefficients_*.csv, table_faithfulness.csv, fig_shap_beeswarm_raw.png |
| Model evaluation | LR vs XGBoost, lookup baseline, five protocols with fold range, saved vs refit, PR and ROC curves, calibration, confusion matrices, decision levels, over-defence, label balance | part6/table_stage2_results.csv, oof_xgb_fold_inj.npy, part6/cache/oof_*_raw_inj.npy (optional), table_calibration.csv, table_threshold_metrics.csv, table_decision_levels.csv, table_over_defence.csv, table_label_balance.csv, table_attack_success_per_agent.csv |
| Security vs usefulness | GPT-4o trade-off redrawn interactively, every agent before and after, per-agent inspector with threshold what-if, leave-one-agent-out | replay_all_configs.csv, replay_leave_one_agent_out.csv, risk_scores.parquet, runs_index.parquet, fig_security_vs_usefulness.png |
| Methodology & limits | Six-part pipeline, both stages, risk formula, decision rule, project status table, limitations | report_figures/*.png, PROGRESS.md (status mirrored by hand) |
| Understand the system | Seven lessons (52 sections), risk-score explainer, glossary of 45 terms | all of the above through `content/facts.py` |
| Guided demo | Eight steps with Next / Previous and a progress bar; steps 3 to 5 follow one saved alert | part6/alerts_examples.csv and the files above |

Every chart has a plain-English title and reading in Beginner mode and the original technical title and method in
Technical mode. 28 components have a "What does this mean?" panel that answers six questions at the selected level.
The original notebook figures are available under each chart.

## Test report

Run from `ui/`:

```
python tests/run_all.py
```

Result on 12 October 2026: **45 tests, all passed. 92 research files fingerprinted before and after; 0 created, changed or deleted.**

What was verified:

- **Counts against source data.** 36,679 runs, 29 configurations, 136,554 tool outputs, 27,016 exposure rows, 33,119 attacked and
  3,560 clean runs, read independently from the parquet files and compared with the dashboard and `part6_summary.json`.
  Decision counts match `table_decision_levels.csv` (8,624 / 17,997 / 395) and `table_decisions_by_output_type.csv`
  (8,725 / 18,315 / 44,669). The two denominators are kept apart and tested.
- **Metrics against results files.** All 20 refit PR-AUC values equal `part6_summary.json`. Headline figures equal RESULTS.md.
  Threshold metrics at 0.20 reproduce `table_threshold_metrics.csv`. PR-AUC recomputed from the saved score arrays matches
  the table to 9 decimals. ECE matches `table_calibration.csv`. Replay at 0.20 reproduces `replay_all_configs.csv` (difference 1e-16).
- **Risk and decision boundaries.** risk ≥ 0.20 is inclusive, P(injection) ≥ 0.5 is inclusive, HIGH RISK is checked first,
  invalid probabilities are rejected, scalar and array rules agree on a 41 × 41 grid, and the rule reproduces the saved
  decision of all 27,040 alerts.
- **Live Stage 1.** Refit equals the saved coefficients and NotInject scores; the score equals the sum of its word
  contributions, tool term and intercept; empty, over-long, non-text and unknown-tool input is rejected with a message;
  out-of-vocabulary text is flagged. P(obey) equals the value the pipeline uses.
- **Alerts.** Each filter, literal (non-regex) search, paging, reset to page 1 on filter change, empty state.
  150 sampled alerts point at the right full text. One page (25 rows) is sent to the browser, not the file.
- **Missing files.** Missing optional refit scores leave only the saved curve; a missing required file gives a named
  message on the page; a missing figure gives a note. No crash in any case.
- **Pages.** All nine pages render at both explanation levels with no exception; all seven lessons at both levels; the
  guided demo steps forward through all eight steps and back; the level switch changes the text and keeps the warnings.
- **No paths or unfinished claims.** No absolute path appears in any page or content; the `path` column of the run index is
  dropped; encoder, Integrated Gradients, external-detector comparison and the dissertation chapter are all marked not built.
- **Cold start.** With `ui/cache` deleted the detector was rebuilt in 33 seconds and passed its checks.
- **In a browser.** Overview, Risk analysis (example load, analyse), Alerts (row select, detail), Explainability, Evaluation,
  Security vs usefulness, Learn and Guided demo were opened and looked at in Chromium at 1440 × 900. The only network requests
  were to localhost.

What could not be tested:

- A clean virtual environment install of `requirements.txt`. Installed versions were checked against the file (all eight
  satisfy it), but nothing was downloaded.
- Browsers other than Chromium, and phone-size screens. The layout uses Streamlit's responsive columns; it was not looked at below 1440 px.
- Other operating systems, and the Colab environment.
- A machine without a prebuilt `ui/cache` and with a different scikit-learn version at the same time (each was covered separately).
- Several simultaneous users.
- The last three guided-demo steps and the Methodology page were checked by headless test and page text, not by eye.

## Limitations and unfinished ML components

- **Fine-tuned encoder: not started.** No encoder outputs exist.
- **Integrated Gradients: not started.** It needs the encoder.
- **Direct external-detector comparison: not done.** No off-the-shelf detector was scored on this project's samples.
- **Dissertation results chapter and Section N diagrams: not done.**
- Live scoring is Stage 1 only; P(obey) for new text is an agent average, not a prediction for that text.
- Stage 1 was trained on benchmark tool outputs and mostly detects prose inside structured data. On free text it can be wrong.
- All results are replays of saved AgentDojo runs. They are not a live or production security guarantee.
- The explained XGBoost is the Part 6 refit, not the model behind the saved scores (Spearman 0.976 between them).
- Per-row SHAP values were not saved, so the beeswarm is a static image and SHAP reasons exist only for saved alerts.
- The saved result "71.5% (95% CI 63.0 to 78.8)" for the held-out threshold is quoted from RESULTS.md; its code is not in the folder.
- The status table on the Methodology page mirrors PROGRESS.md by hand. Update `views/method.py` when the status changes.
- Text masking (emails, account numbers, links) is display-only and on by default; the benchmark text is synthetic.

## How to demonstrate it (about 8 minutes)

1. **Guided demo** (Learn → Guided demo). Leave the level on Beginner and press Next through the eight steps. This is the
   whole story in order: the problem, the design, one real example, its risk calculation, its alert, the evaluation,
   the trade-off, and the limits. If time is short, this page alone is enough.
2. **Risk analysis → Live analysis.** Press "Load example into the box", choose agent `gpt-4-0125-preview`, press Analyse:
   HIGH RISK. Switch the agent to `claude-3-5-sonnet-20241022` and analyse again: SUSPICIOUS, same text. That is the
   two-stage idea in ten seconds. Then load the harmless example: SAFE. Say out loud that only the first probability is live.
   Optional: load the NotInject example with `gpt-4-0125-preview`. It is a harmless question and it comes out HIGH RISK
   (P(injection) 0.797, the same value the Day 2 notebook saved). It is a real false alarm and a good way to show the limits.
3. **Alerts explorer.** Filter "What really happened" to "harmless output (false alarm)" and open one. It shows the system
   being wrong, which examiners like to see.
4. **Model evaluation → Models and test protocols.** Point at the leftmost group (random split) and the second group
   (unseen injection task): 0.858 against 0.664. This is the methodological point of the project.
5. **Explainability.** Read the five numbered weaknesses at the top, then show the orange bars in the SHAP chart.
6. **Security vs usefulness → GPT-4o.** Top-left is best. The orange point is the guard; the orange bar is the uncertainty.
7. **Methodology & limits → Project status.** End on what is not built.

For a technical audience, switch the sidebar to Technical before step 4.
For someone new to ML who wants to explore alone, point them to **Understand the system** and Lesson 1.
