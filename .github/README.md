# Prompt-injection defence dashboard

A dashboard for a research project on **ML-based prompt-injection detection and agent obedience risk prediction**.

AI agents read emails, files and web pages. An attacker can hide instructions in that text. This project estimates,
for each piece of text an agent receives from a tool, how likely it is to contain a planted instruction (Stage 1) and
how likely this particular agent is to obey it (Stage 2). The product of the two is a risk score, and a fixed rule turns
it into HIGH RISK, SUSPICIOUS or SAFE.

> **Research prototype.** Every result is a replay of saved [AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark
> runs. Nothing here is a measurement of a live, deployed defence. A fine-tuned encoder, Integrated Gradients and a direct
> comparison with an off-the-shelf detector were planned and are **not built**.

## What is in this repository

- `ui/` the dashboard (Streamlit + Plotly), its tests, and its own README with the architecture, data sources and test report.
- The saved result files the dashboard reads: scores, alerts with reasons, replay tables, result tables and figures (`part6/`, `report_figures/`, and the files in the root).
- `run_dashboard.py` a local launcher.

The notebooks that produced the results, the embeddings, the refitted XGBoost models and the written report are not in this repository.

## Run it locally

```
pip install -r ui/requirements.txt
python run_dashboard.py
```

It opens http://localhost:8501. Built and tested on Python 3.13.

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with the GitHub account that owns this repository.
2. **Create app** → *Deploy a public app from GitHub* → repository `manojkannaj8/prompt-injection`, branch `main`, main file path `ui/app.py`.
3. **Deploy**. Python 3.13 or 3.14 both work (3.13 is the version the dashboard was tested on locally).

The dependencies are read from `ui/requirements.txt` and the theme from `ui/.streamlit/config.toml`.
This app cannot be hosted on Vercel: Streamlit needs a long-running server with a WebSocket connection, and Vercel runs short serverless functions.

## Pages

Overview · Risk analysis (live Stage 1 scoring, saved examples) · Alerts explorer · Explainability · Model evaluation ·
Security vs usefulness · Methodology and limits · Understand the system (seven lessons and a glossary for readers new to ML) · Guided demo.

Live scoring of new text covers Stage 1 only. For new text the obedience probability is the selected agent's measured
obey rate in the saved runs, not a prediction for that text. The dashboard says so wherever it applies.

## Tests

```
cd ui
python tests/run_all.py
```

45 tests check the displayed numbers against the saved files, the decision boundaries, the live detector against saved
scores, the alert filters, and that no data file is modified by the dashboard.

## Data

The tool-output text comes from saved runs of the AgentDojo benchmark and is synthetic: the names, email addresses and
account numbers in it are fictional. `notinject_scores.parquet` holds prompts from the NotInject dataset
(`leolee99/NotInject`) with this project's Stage 1 scores. Check the licences of both sources before reusing the data.
