"""Chart builders. Every chart is drawn from a saved result file; none contains typed-in data."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import components as ui
from core import data, replay, rules, stats

PROTOCOL_LABEL = {"Random split (leaky)": "Random split<br>(leaky, comparison only)", "Unseen injection task": "Unseen injection task<br>(main result)",
                  "Unseen user task": "Unseen user task", "Unseen attack style": "Unseen attack style",
                  "Unseen agent (leave one out)": "Unseen agent<br>(leave one out)"}
FAMILY_LABEL = {"words": "Words and word pairs", "agent": "Which agent is running", "tool output meaning": "Meaning of the tool output (embedding)",
                "user task meaning": "Meaning of the user task (embedding)", "flags": "Flags (counts)", "task/output similarity": "Task/output similarity",
                "tool": "Which tool returned it"}
FLAG_NICE = {"n_cue_words": "number of instruction cue words", "out_len": "length of the tool output", "step": "step number",
             "n_email": "email addresses in the output", "n_email_new": "email addresses not in the user task",
             "n_iban": "bank accounts in the output", "n_iban_new": "bank accounts not in the user task",
             "n_url": "web links in the output", "n_url_new": "web links not in the user task"}


def nice_unit(u):
    """Readable name of a SHAP unit (same wording as Part 6 cell 4)."""
    fixed = {"emb_out": "tool output meaning (embedding)", "emb_task": "user task meaning (embedding)", "sim": "task/output similarity",
             "agent": "which agent is running", "tool": "which tool returned it", **FLAG_NICE}
    if u in fixed:
        return fixed[u]
    return f"word “{u[5:]}”" if u.startswith("word:") else u


def decisions(counts, total_label):
    order = rules.ORDER
    vals = [counts[d] for d in order]
    tot = sum(vals)
    fig = go.Figure(ui.bar([f"{ui.DECISION_ICON[d]} {d}" for d in order], vals, color=[ui.DECISION_COLOR[d] for d in order],
                           text=[f"{v:,}<br>{v / tot:.1%}" for v in vals], hover="%{x}: %{y:,}<extra></extra>"))
    return ui.style(fig, height=340, legend=False, yaxis=dict(title=f"tool outputs ({total_label})", range=[0, max(vals) * 1.22]))


def obey_by_level(levels):
    order = rules.ORDER
    vals = [levels[d]["obey_rate"] * 100 for d in order]
    fig = go.Figure(ui.bar([f"{ui.DECISION_ICON[d]} {d}" for d in order], vals, color=[ui.DECISION_COLOR[d] for d in order],
                           text=[f"{v:.1f}%" for v in vals], hover="%{x}: %{y:.1f}% were obeyed<extra></extra>"))
    return ui.style(fig, height=340, legend=False, yaxis=dict(title="share really obeyed in the saved run (%)", range=[0, max(vals) * 1.25]))


def stage2_main():
    """Stage 2 PR-AUC on unseen injection tasks: lookup, LR, XGBoost (refits) and the saved XGBoost."""
    t = stats.stage2_table()
    t = t[(t.protocol == "Unseen injection task") & (t.text == "raw")]
    order = ["Model-only lookup", "Logistic Regression", "XGBoost", "XGBoost (saved Part 3 scores)"]
    t = t.set_index("model").reindex(order).dropna(subset=["PR-AUC"])
    labels = {"Model-only lookup": "Agent identity only<br>(lookup)", "Logistic Regression": "Logistic<br>Regression",
              "XGBoost": "XGBoost<br>(Part 6 refit)", "XGBoost (saved Part 3 scores)": "XGBoost<br>(saved Part 3)"}
    fig = go.Figure(ui.bar([labels[m] for m in t.index], t["PR-AUC"], color=[ui.MODEL_COLOR[m] for m in t.index],
                           text=[f"{v:.3f}" for v in t["PR-AUC"]], hover="%{x}<br>PR-AUC %{y:.3f}<extra></extra>"))
    base = data.overall_obey_rate()
    fig.add_hline(y=base, line=dict(color=ui.MUTED, width=1.5), annotation_text=f"guessing = {base:.3f}", annotation_position="top left",
                  annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    return ui.style(fig, height=340, legend=False, yaxis=dict(title="PR-AUC (0 to 1, higher is better)", range=[0, 1]))


def protocols(metric="PR-AUC", text="raw"):
    t = stats.stage2_table()
    t = t[(t.text == text) & (t.source == "Part 6 refit")]
    order = [p for p in PROTOCOL_LABEL if p in set(t.protocol)]
    fig = go.Figure()
    for m in ["Model-only lookup", "Logistic Regression", "XGBoost"]:
        d = t[t.model == m].set_index("protocol").reindex(order)
        if d[metric].notna().sum() == 0:
            continue
        err = None
        if metric == "PR-AUC":
            err = dict(type="data", symmetric=False, array=(d["PR-AUC best fold"] - d["PR-AUC"]).values,
                       arrayminus=(d["PR-AUC"] - d["PR-AUC worst fold"]).values, color=ui.INK2, thickness=1, width=3)
        fig.add_trace(go.Bar(x=[PROTOCOL_LABEL[p] for p in order], y=d[metric], name=m, error_y=err,
                             marker=dict(color=ui.MODEL_COLOR[m], line=dict(color=ui.SURFACE, width=2), cornerradius=4),
                             customdata=np.stack([d["PR-AUC worst fold"], d["PR-AUC best fold"]], axis=-1),
                             hovertemplate=f"{m}<br>{metric} %{{y:.3f}}<br>fold range %{{customdata[0]:.3f}} to %{{customdata[1]:.3f}}<extra></extra>"))
    if metric == "PR-AUC":
        base = data.overall_obey_rate()
        fig.add_hline(y=base, line=dict(color=ui.MUTED, width=1.5), annotation_text=f"guessing = {base:.3f}",
                      annotation_position="top left", annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    else:
        fig.add_hline(y=0.5, line=dict(color=ui.MUTED, width=1.5), annotation_text="guessing = 0.5", annotation_position="top left",
                      annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    return ui.style(fig, height=420, barmode="group", yaxis=dict(title=f"Stage 2 {metric} (whiskers: worst and best fold)" if metric == "PR-AUC" else f"Stage 2 {metric}", range=[0, 1.02]))


def guard_effect(kind):
    """kind = 'attack' or 'task': no guard vs guard at 0.20, pooled over the replay agents and for GPT-4o."""
    s = stats.overview()
    g = s["gpt4o"]
    if kind == "attack":
        before, after, ytitle = [s["asr_pooled_before"], g["asr_before"]], [s["asr_pooled_after"], g["asr_after"]], "attack success (%)"
    else:
        before, after, ytitle = [s["task_pooled_before"], g["task_before"]], [s["task_pooled_after"], g["task_after"]], "task success, no attack (%)"
    x = [f"All {s['n_replay_agents']} agents<br>(pooled)", "GPT-4o"]
    fig = go.Figure()
    for name, v, c in [("No guard", before, ui.GRAY), (f"With the guard (risk ≥ {rules.T_HIGH:.2f})", after, ui.BLUE)]:
        fig.add_trace(go.Bar(x=x, y=[100 * a for a in v], name=name, text=[f"{100 * a:.1f}%" for a in v], textposition="outside", cliponaxis=False,
                             textfont=dict(color=ui.INK2, size=11.5), marker=dict(color=c, line=dict(color=ui.SURFACE, width=2), cornerradius=4),
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
    top = max(100 * max(before + after) * 1.25, 10)
    return ui.style(fig, height=340, barmode="group", yaxis=dict(title=ytitle, range=[0, min(top, 105)]))


def family():
    t = data.table("table_shap_by_family").rename(columns={"Unnamed: 0": "family"}).sort_values("raw share")
    fig = go.Figure()
    for col, name, c in [("masked share", "Template-masked text", ui.AQUA), ("raw share", "Raw text", ui.BLUE)]:
        fig.add_trace(go.Bar(y=[FAMILY_LABEL.get(x, x) for x in t.family], x=t[col] * 100, name=name, orientation="h",
                             text=[f"{v:.0%}" if v >= 0.005 else "<1%" for v in t[col]], textposition="outside", cliponaxis=False,
                             textfont=dict(color=ui.INK2, size=11), marker=dict(color=c, line=dict(color=ui.SURFACE, width=2), cornerradius=4),
                             hovertemplate=f"{name}<br>%{{y}}: %{{x:.1f}}% of mean |SHAP|<extra></extra>"))
    return ui.style(fig, height=400, barmode="group", xaxis=dict(title="share of the obedience score's total influence (%)", range=[0, 45]),
                    legend_traceorder="reversed")


def shap_top(version="raw", n=18):
    G = data.table(f"shap_global_stage2_xgb_{version}").head(n).iloc[::-1]
    fig = go.Figure()
    for flag, name, c in [(False, "Feature", ui.BLUE), (True, "Term from the fixed attack wrapper" if version == "raw" else "The [WRAP] placeholder", ui.ORANGE)]:
        d = G[G.wrapper_term == flag]
        fig.add_trace(go.Bar(y=[nice_unit(u) for u in d.unit], x=d.mean_abs_shap, name=name, orientation="h",
                             marker=dict(color=c, cornerradius=4), customdata=np.stack([d.mean_shap, d.doc_freq], axis=-1),
                             hovertemplate="%{y}<br>mean |SHAP| %{x:.3f} log-odds<br>mean signed SHAP %{customdata[0]:+.3f}<br>present in %{customdata[1]:,} outputs<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[nice_unit(u) for u in G.unit])
    return ui.style(fig, height=max(360, 24 * n + 80), xaxis=dict(title="mean |SHAP|, log-odds of obeying"))


def agent_effect():
    a = data.table("table_agent_shap")
    fig = go.Figure(go.Scatter(x=a.obey_rate * 100, y=a.mean_agent_shap, mode="markers", text=a.config, customdata=a.n,
                               marker=dict(size=11, color=ui.BLUE, line=dict(color=ui.SURFACE, width=2)),
                               hovertemplate="%{text}<br>real obey rate %{x:.1f}%<br>mean agent SHAP %{y:+.2f}<br>%{customdata:,} exposure rows<extra></extra>"))
    for i in [0, len(a) - 1]:
        fig.add_annotation(x=a.obey_rate[i] * 100, y=a.mean_agent_shap[i], text=a.config[i], showarrow=False, yshift=14 if i else -14,
                           font=dict(color=ui.INK2, size=11))
    fig.add_hline(y=0, line=dict(color=ui.AXIS, width=1))
    return ui.style(fig, height=400, legend=False, xaxis=dict(title="how often the agent really obeyed (%)"),
                    yaxis=dict(title="model's average push for this agent (SHAP, log-odds)"))


def coefficients(table_name, drop_agents=False, n_pos=12, n_neg=8):
    t = data.table(table_name)
    if drop_agents and "family" in t.columns:
        t = t[t.family != "agent"]
    t = t.sort_values("coef")
    w = pd.concat([t.head(n_neg), t.tail(n_pos)])
    fig = go.Figure(go.Bar(y=[f.replace("word:", "") for f in w.feature], x=w.coef, orientation="h",
                           marker=dict(color=[ui.ORANGE if c > 0 else ui.BLUE for c in w.coef], cornerradius=4),
                           hovertemplate="%{y}<br>coefficient %{x:+.2f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.AXIS, width=1))
    return ui.style(fig, height=520, legend=False, xaxis=dict(title="coefficient (log-odds): orange raises the score, blue lowers it"))


def faithfulness(model_prefix):
    fa = data.table("table_faithfulness")
    d = fa[fa.model.str.startswith(model_prefix)]
    fig = go.Figure()
    for col, name, c in [("drop, top-attributed", "Words the explanation ranks highest", ui.ORANGE), ("drop, random", "Random words from the same output", ui.GRAY)]:
        fig.add_trace(go.Scatter(x=d["words removed"], y=d[col], name=name, mode="lines+markers", line=dict(color=c, width=2),
                                 marker=dict(size=9, color=c, line=dict(color=ui.SURFACE, width=2)),
                                 hovertemplate=f"{name}<br>%{{x}} words removed: score drops %{{y:.2f}} log-odds<extra></extra>"))
    return ui.style(fig, height=340, xaxis=dict(title="words removed", tickvals=list(d["words removed"])),
                    yaxis=dict(title="mean drop in obey score (log-odds)", rangemode="tozero"))


def pr_curves():
    fig = go.Figure()
    for c in stats.curves():
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=f"{c['model']} · PR-AUC {c['pr_auc']:.3f}", mode="lines",
                                 line=dict(color=ui.MODEL_COLOR[c["model"]], width=2),
                                 hovertemplate=f"{c['model']}<br>recall %{{x:.2f}}, precision %{{y:.2f}}<extra></extra>"))
    base = data.overall_obey_rate()
    fig.add_hline(y=base, line=dict(color=ui.MUTED, width=1.5), annotation_text=f"guessing = {base:.3f}", annotation_position="bottom right",
                  annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    return ui.style(fig, height=400, xaxis=dict(title="recall (share of obeyed attacks caught)", range=[0, 1]),
                    yaxis=dict(title="precision", range=[0, 1.02]))


def roc_curves():
    fig = go.Figure()
    for c in stats.curves():
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], name=f"{c['model']} · ROC-AUC {c['roc_auc']:.3f}", mode="lines",
                                 line=dict(color=ui.MODEL_COLOR[c["model"]], width=2),
                                 hovertemplate=f"{c['model']}<br>false positive rate %{{x:.2f}}, true positive rate %{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, width=1.5), name="guessing", hoverinfo="skip"))
    return ui.style(fig, height=400, xaxis=dict(title="false positive rate", range=[0, 1]), yaxis=dict(title="true positive rate", range=[0, 1.02]))


def calibration():
    fig = go.Figure()
    for c, col in zip(stats.calibration(), [ui.GRAY, ui.BLUE]):
        p = c["points"]
        fig.add_trace(go.Scatter(x=p.predicted, y=p.observed, name=f"{c['name']} · ECE {c['ece']:.3f}", mode="lines+markers", customdata=p.n,
                                 line=dict(color=col, width=2), marker=dict(size=9, color=col, line=dict(color=ui.SURFACE, width=2)),
                                 hovertemplate=f"{c['name']}<br>predicted %{{x:.2f}}, actually obeyed %{{y:.2f}}<br>%{{customdata:,}} rows<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, width=1.5), name="perfect", hoverinfo="skip"))
    return ui.style(fig, height=420, xaxis=dict(title="predicted chance of obeying", range=[0, 1]),
                    yaxis=dict(title="share that actually obeyed", range=[0, 1]))


def confusion(tp, fp, fn, tn, row_labels, col_labels=("not flagged", "flagged")):
    """Row-normalised 2x2 matrix as a one-hue heatmap with counts written in."""
    M = np.array([[tn, fp], [fn, tp]], dtype=float)
    share = M / M.sum(1, keepdims=True)
    txt = [[f"{int(M[i, j]):,}<br>{share[i, j]:.1%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=share, x=list(col_labels), y=list(row_labels), text=txt, texttemplate="%{text}", zmin=0, zmax=1, xgap=3, ygap=3,
                               colorscale=[[0, "#16263c"], [1, ui.BLUE]], showscale=False, textfont=dict(color=ui.INK, size=13),
                               hovertemplate="really %{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return ui.style(fig, height=280, legend=False)


def over_defence():
    od = data.table("table_over_defence")
    c1, c2 = od.columns[2], od.columns[3]
    short = ["AgentDojo harmless outputs<br>(all rows)", "AgentDojo harmless outputs<br>(distinct texts, worst case)", "AgentDojo clean runs only",
             "NotInject, average agent", "NotInject, most easily<br>fooled agent"]
    fig = go.Figure()
    for col, name, c in [(c1, "Stage 1 flags it (P(injection) ≥ 0.5)", ui.GRAY), (c2, f"Guard blocks it (risk ≥ {rules.T_HIGH:.2f})", ui.BLUE)]:
        fig.add_trace(go.Bar(x=short[:len(od)], y=od[col] * 100, name=name, text=[f"{v:.2%}" for v in od[col]], textposition="outside", cliponaxis=False,
                             textfont=dict(color=ui.INK2, size=11), marker=dict(color=c, line=dict(color=ui.SURFACE, width=2), cornerradius=4),
                             customdata=od.n, hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}% of %{{customdata:,}} harmless texts<extra></extra>"))
    return ui.style(fig, height=400, barmode="group", yaxis=dict(title="harmless texts flagged (%)", range=[0, max(od[c1].max(), od[c2].max()) * 125]))


def attack_per_agent():
    a = data.table("table_attack_success_per_agent").sort_values("asr")
    fig = go.Figure(go.Bar(y=a.config, x=a.asr * 100, orientation="h", text=[f"{v:.1f}" for v in a.asr * 100], textposition="outside", cliponaxis=False,
                           textfont=dict(color=ui.INK2, size=11), marker=dict(color=ui.BLUE, cornerradius=4), customdata=a.n,
                           hovertemplate="%{y}<br>attack success %{x:.1f}% of %{customdata} attacked runs<extra></extra>"))
    return ui.style(fig, height=560, legend=False, xaxis=dict(title="attack success (%), main attack, no defence", range=[0, a.asr.max() * 112]))


def dumbbell(df, before, after, after_name, highlight=None):
    d = df.sort_values(before).reset_index(drop=True)
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_shape(type="line", x0=r[after] * 100, x1=r[before] * 100, y0=r.config, y1=r.config,
                      line=dict(color=ui.BLUE if r.config == highlight else ui.AXIS, width=4 if r.config == highlight else 3), layer="below")
    fig.add_trace(go.Scatter(x=d[before] * 100, y=d.config, mode="markers", name="No guard", marker=dict(size=11, color=ui.GRAY, line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="%{y}<br>no guard: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=d[after] * 100, y=d.config, mode="markers", name=after_name, marker=dict(size=11, color=ui.BLUE, line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="%{y}<br>" + after_name + ": %{x:.1f}%<extra></extra>"))
    return ui.style(fig, height=max(420, 23 * len(d) + 90), xaxis=dict(title="attack success (%)", rangemode="tozero"),
                    yaxis=dict(tickfont=dict(size=11, color=ui.INK2)))


def task_change(df, highlight=None):
    d = df.sort_values("ASR no guard").reset_index(drop=True)
    fig = go.Figure(go.Bar(y=d.config, x=d["task change (pts)"], orientation="h",
                           marker=dict(color=[ui.BLUE if c == highlight else ui.ORANGE for c in d.config], cornerradius=4),
                           hovertemplate="%{y}<br>task success change: %{x:+.1f} points<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.AXIS, width=1))
    return ui.style(fig, height=max(420, 23 * len(d) + 90), legend=False,
                    xaxis=dict(title="change in task success (points)", range=[min(-12, d["task change (pts)"].min() - 1), 2]), yaxis=dict(showticklabels=False))


GPT4O_SETUPS = {"gpt-4o-2024-05-13": "No defence", "gpt-4o-2024-05-13-repeat_user_prompt": "Repeat user prompt",
                "gpt-4o-2024-05-13-spotlighting_with_delimiting": "Spotlighting", "gpt-4o-2024-05-13-tool_filter": "Tool filter",
                "gpt-4o-2024-05-13-transformers_pi_detector": "Built-in detector"}


def tradeoff_gpt4o():
    """Native redraw of fig_security_vs_usefulness.png: saved GPT-4o setups, the guard's threshold sweep, and the guard at 0.20."""
    base = "gpt-4o-2024-05-13"
    sw = replay.sweep(base)
    sw = sw[(sw.threshold >= 0.01) & (sw.threshold <= 0.60)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw.attack_success * 100, y=sw.task_success * 100, mode="lines", name="Our guard, threshold sweep (tuned on these runs)",
                             line=dict(color=ui.BLUE, width=2), customdata=sw.threshold,
                             hovertemplate="guard at threshold %{customdata:.2f}<br>attack success %{x:.1f}%, task success %{y:.1f}%<extra></extra>"))
    rp = data.replay_all().set_index("config")
    xs, ys, labs = [], [], []
    for cfg, lab in GPT4O_SETUPS.items():
        if cfg in rp.index:
            xs.append(rp.loc[cfg, "ASR no guard"] * 100); ys.append(rp.loc[cfg, "task no guard"] * 100); labs.append(lab)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labs, textposition="top center", textfont=dict(color=ui.INK2, size=11.5),
                             name="Saved GPT-4o setups, as run", marker=dict(size=12, color=ui.GRAY, line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="%{text} (saved run)<br>attack success %{x:.1f}%, task success %{y:.1f}%<extra></extra>"))
    a, t, _, n = replay.replay(base)
    lo, hi = replay.task_ci(base)
    fig.add_trace(go.Scatter(x=[a * 100], y=[t * 100], mode="markers+text", text=[f"Our guard at {rules.T_HIGH:.2f}"], textposition="bottom right",
                             textfont=dict(color=ui.INK, size=12), name=f"Our guard at risk ≥ {rules.T_HIGH:.2f} (95% interval on task success)",
                             marker=dict(size=14, color=ui.ORANGE, line=dict(color=ui.SURFACE, width=2)),
                             error_y=dict(type="data", symmetric=False, array=[(hi - t) * 100], arrayminus=[(t - lo) * 100], color=ui.ORANGE, thickness=1.5, width=5),
                             hovertemplate=f"our guard at {rules.T_HIGH:.2f} (replay)<br>attack success %{{x:.1f}}%<br>task success %{{y:.1f}}% (95% interval {lo:.1%} to {hi:.1%}, n={n})<extra></extra>"))
    fig.add_annotation(x=1, y=89, text="better: fewer attacks, more tasks done ↖", showarrow=False, xanchor="left", font=dict(color=ui.MUTED, size=11))
    return ui.style(fig, height=470, xaxis=dict(title="attack success (%), lower is better", range=[0, 55]),
                    yaxis=dict(title="task success, no attack (%)", range=[25, 92]))


def sweep(cfg, t_mark=None):
    sw = replay.sweep(cfg)
    fig = go.Figure()
    for col, name, c in [("attack_success", "Attack success (lower is better)", ui.ORANGE), ("task_success", "Task success, no attack (higher is better)", ui.BLUE)]:
        fig.add_trace(go.Scatter(x=sw.threshold, y=sw[col] * 100, name=name, mode="lines", line=dict(color=c, width=2, shape="hv"),
                                 hovertemplate=f"{name}<br>threshold %{{x:.2f}}: %{{y:.1f}}%<extra></extra>"))
    fig.add_vline(x=rules.T_HIGH, line=dict(color=ui.INK2, width=1.5), annotation_text=f"configured threshold {rules.T_HIGH:.2f}",
                  annotation_position="top right", annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    if t_mark is not None and abs(t_mark - rules.T_HIGH) > 1e-9:
        fig.add_vline(x=t_mark, line=dict(color=ui.WARNING, width=1.5), annotation_text=f"what-if {t_mark:.2f}", annotation_position="bottom right",
                      annotation_font=dict(color=ui.INK2, size=11), annotation_bgcolor="rgba(13,17,23,0.82)")
    return ui.style(fig, height=380, hovermode="x unified", xaxis=dict(title="risk threshold for HIGH RISK", range=[0, 1]),
                    yaxis=dict(title="share of runs (%)", range=[0, 100]))


def word_contributions(up, down):
    """Stage 1 word contributions for one text: coefficient x TF-IDF value, log-odds."""
    d = pd.concat([down.sort_values("contribution"), up.sort_values("contribution")])
    fig = go.Figure(go.Bar(y=[f"“{t}”" for t in d.term], x=d.contribution, orientation="h",
                           marker=dict(color=[ui.ORANGE if c > 0 else ui.BLUE for c in d.contribution], cornerradius=4),
                           customdata=np.stack([d.coefficient, d.tfidf], axis=-1),
                           hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<br>= weight %{customdata[0]:+.2f} × prominence %{customdata[1]:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.AXIS, width=1))
    return ui.style(fig, height=max(260, 26 * len(d) + 90), legend=False,
                    xaxis=dict(title="push on the injection score (log-odds): orange raises it, blue lowers it"))


_REASON = re.compile(r"^(.*)\s\(([+-]\d+(?:\.\d+)?)\)$")


def parse_reasons(text):
    """'contains "x" (+0.12); agent is y (-1.20)' -> [(label, value), ...]. Empty when the text has no scored items."""
    out = []
    for part in (text or "").split("; "):
        m = _REASON.match(part.strip())
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def reasons(items, xtitle):
    items = sorted(items, key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(y=[k for k, _ in items], x=[v for _, v in items], orientation="h",
                           marker=dict(color=[ui.ORANGE if v > 0 else ui.BLUE for _, v in items], cornerradius=4),
                           text=[f"{v:+.2f}" for _, v in items], textposition="outside", cliponaxis=False, textfont=dict(color=ui.INK2, size=11),
                           hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.AXIS, width=1))
    span = max(abs(v) for _, v in items) * 1.35
    return ui.style(fig, height=max(190, 34 * len(items) + 80), legend=False, xaxis=dict(title=xtitle, range=[-span if any(v < 0 for _, v in items) else 0, span]))
