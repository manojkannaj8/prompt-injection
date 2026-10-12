"""Detail view of one saved tool output. Everything shown is loaded from the saved files; nothing is recomputed."""
import streamlit as st

import components as ui
from core import data
from views import charts


def from_alert(a):
    """Record dict from a row of the alerts table."""
    return {k: a[k] for k in ["run_id", "step", "config", "suite", "tool_name", "attack", "kind", "outcome", "p_inj", "p_obey", "risk",
                              "decision", "why_injection", "why_obey", "why_resist", "row", "alert_id"]}


def from_risk_row(r):
    """Record dict from a row of risk_scores, with the saved reasons attached when an alert exists for it."""
    rec = {k: r[k] for k in ["run_id", "step", "config", "suite", "tool_name", "attack", "kind", "p_inj", "p_obey", "risk", "decision", "row"]}
    if r["is_exposure"] == 1:
        rec["outcome"] = "attack succeeded in the saved run" if r["obeyed"] == 1 else "attack failed in the saved run"
    elif r["has_injection"] == 1:
        rec["outcome"] = "later injected output (the obey label belongs to the run's first injected output)"
    else:
        rec["outcome"] = "harmless output"
    A = data.alerts()
    hit = A[(A.run_id == r["run_id"]) & (A.step == r["step"])]
    if len(hit):
        h = hit.iloc[0]
        rec.update(alert_id=h.alert_id, why_injection=h.why_injection, why_obey=h.why_obey, why_resist=h.why_resist)
    return rec


def render(rec, key="rec"):
    is_exposure = str(rec["kind"]).startswith("exposure")
    st.markdown(" ".join([ui.provenance("historical"), ui.decision_chip(rec["decision"]), ui.chip(rec["kind"]), ui.chip(rec["outcome"])]),
                unsafe_allow_html=True)
    st.write("")
    tool_text, task_text = data.text_for_row(rec["row"])
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Context**")
        attack = rec.get("attack")
        ctx = {"Alert ID": rec.get("alert_id", "no alert (decision was SAFE)"), "Agent configuration": rec["config"], "Task suite": rec["suite"],
               "Tool that returned the text": rec["tool_name"], "Step in the run": int(rec["step"]),
               "Attack style": attack if isinstance(attack, str) and attack else "none (clean run)", "Run": int(rec["run_id"])}
        rows = "".join(f'<tr><td style="padding:6px 10px 6px 0;color:var(--muted);vertical-align:top;white-space:nowrap">{ui.esc(k)}</td>'
                       f'<td style="padding:6px 0;color:var(--ink);word-break:break-word">{ui.esc(v)}</td></tr>' for k, v in ctx.items())
        st.markdown(f'<div class="pi-card"><table style="width:100%;border-collapse:collapse;font-size:.88rem">{rows}</table></div>', unsafe_allow_html=True)
    with right:
        st.markdown("**The user's original task**")
        ui.show_text(task_text or "(not recorded)", limit=700)
        st.markdown("**Tool output the agent received**")
        ui.show_text(tool_text, limit=1500)
        st.caption("Synthetic benchmark text from AgentDojo. Names, addresses and account numbers are fictional. "
                   "It is displayed as inert text and never executed.")

    st.markdown("#### How the saved decision was reached")
    note = ("Stage 2 XGBoost, calibrated, scored by a model that had not seen this attack goal (saved)." if is_exposure
            else "The agent's usual obey rate. No Stage 2 score exists for this output.")
    ui.risk_breakdown(float(rec["p_inj"]), float(rec["p_obey"]), "historical", p_obey_note=note, key=key)

    if "why_injection" in rec:
        st.markdown("#### Reasons stored with this alert")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Why it looks like an injection** (Stage 1 words)")
            items = charts.parse_reasons(rec["why_injection"].replace('"', ""))
            if items:
                ui.show(charts.reasons(items, "push on the injection score (log-odds)"), key=f"{key}_s1")
            else:
                st.caption(rec["why_injection"] or "No single word stands out.")
        with c2:
            st.markdown("**Why this agent may obey, and what lowers the risk** (Stage 2 SHAP)")
            items = charts.parse_reasons(rec["why_obey"]) + charts.parse_reasons(rec.get("why_resist", ""))
            if items:
                ui.show(charts.reasons(items, "push on the obedience score (SHAP, log-odds)"), key=f"{key}_s2")
            else:
                st.caption(rec["why_obey"])
        st.caption("Stage 2 reasons come from the XGBoost refitted in Part 6, which closely tracks but is not identical to the model behind the saved score.")
    else:
        st.caption("No reasons were stored for this output because its decision was SAFE (only HIGH RISK and SUSPICIOUS outputs become alerts).")
    ui.explain("saved_record")
