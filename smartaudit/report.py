"""Scoring + audit reports (per call and across many calls)."""
from __future__ import annotations
import pandas as pd
from typing import Dict, List
from .schemas import AuditSheet, AuditRun, Call


def score_call(sheet: AuditSheet, answers: Dict[str, str]) -> dict:
    sec, crit_fail = {}, []
    tot_w = tot_ok = 0
    for q in sheet.questions:
        a = answers.get(q.qid)
        if a not in ("YES", "NO"):
            continue
        s = sec.setdefault(q.section, [0, 0]); s[0] += q.weight * (a == "YES"); s[1] += q.weight
        tot_w += q.weight; tot_ok += q.weight * (a == "YES")
        if q.critical and a == "NO":
            crit_fail.append(q.qid)
    total = round(100 * tot_ok / tot_w, 1) if tot_w else 0.0
    return dict(total=total, sections={k: round(100 * v[0] / v[1], 1) for k, v in sec.items()},
                critical_fails=crit_fail, passed=(total >= sheet.pass_mark and not crit_fail))


def grade(total: float, crit: bool) -> str:
    if crit: return "Auto-fail"
    return "Excellent" if total >= 90 else "Good" if total >= 80 else "Needs coaching" if total >= 65 else "Poor"


def batch_report(runs: List[AuditRun], calls: Dict[str, Call], sheet: AuditSheet) -> dict:
    rows, qrows = [], []
    qmap = sheet.by_qid()
    for r in runs:
        c = calls[r.call_id]; sc = score_call(sheet, r.answers())
        rows.append(dict(call_id=r.call_id, agent=c.agent_name, scenario=c.scenario, score=sc["total"], passed=sc["passed"],
                         grade=grade(sc["total"], bool(sc["critical_fails"])), critical_fails=", ".join(sc["critical_fails"]),
                         **{f"sec:{k}": v for k, v in sc["sections"].items()}, cost=r.usage.cost_usd))
        for x in r.results:
            qrows.append(dict(call_id=r.call_id, agent=c.agent_name, qid=x.qid, section=qmap[x.qid].section, answer=x.answer))
    calls_df, q_df = pd.DataFrame(rows), pd.DataFrame(qrows)
    app = q_df[q_df.answer.isin(["YES", "NO"])]
    qfail = (app.assign(fail=app.answer == "NO").groupby("qid").agg(applicable=("fail", "size"), fails=("fail", "sum"))
             .assign(fail_rate=lambda d: (100 * d.fails / d.applicable).round(1)).reset_index())
    qfail["question"] = qfail.qid.map(lambda i: qmap[i].text); qfail["critical"] = qfail.qid.map(lambda i: qmap[i].critical)
    qfail = qfail.sort_values("fail_rate", ascending=False)
    sec_cols = [c for c in calls_df.columns if c.startswith("sec:")]
    agent_df = calls_df.groupby("agent").agg(calls=("call_id", "size"), avg_score=("score", "mean"),
                                             pass_rate=("passed", "mean"), **{c: (c, "mean") for c in sec_cols}).round(1).reset_index()
    agent_df["pass_rate"] = (agent_df["pass_rate"] * 100).round(1)
    agent_df.columns = [c.replace("sec:", "") for c in agent_df.columns]
    return dict(calls=calls_df, questions=qfail, agents=agent_df.sort_values("avg_score", ascending=False),
                kpis=dict(n=len(calls_df), avg=round(calls_df.score.mean(), 1), pass_rate=round(100 * calls_df.passed.mean(), 1),
                          crit_calls=int((calls_df.critical_fails != "").sum()), cost=round(calls_df.cost.sum(), 4)))


def call_feedback(sheet: AuditSheet, run: AuditRun) -> str:
    qm = sheet.by_qid(); sc = score_call(sheet, run.answers())
    fails = [qm[r.qid] for r in run.results if r.answer == "NO"]
    wins = [qm[r.qid] for r in run.results if r.answer == "YES"]
    lines = [f"Score {sc['total']}% - {grade(sc['total'], bool(sc['critical_fails']))}."]
    if sc["critical_fails"]:
        lines.append("CRITICAL: " + "; ".join(f"{q} ({qm[q].text})" for q in sc["critical_fails"]))
    if fails:
        lines.append("Coach on: " + "; ".join(f"{q.qid} {q.text}" for q in fails if not q.critical) or "-")
    lines.append(f"Strengths: {len(wins)} of {len(wins) + len(fails)} applicable checks passed.")
    return "\n".join(lines)


def to_markdown(rep: dict) -> str:
    k = rep["kpis"]
    md = [f"# Audit report\n", f"- Calls: {k['n']}\n- Average score: {k['avg']}%\n- Pass rate: {k['pass_rate']}%\n- Calls with a critical failure: {k['crit_calls']}\n",
          "## Agents\n", rep["agents"].to_markdown(index=False) if hasattr(rep["agents"], "to_markdown") else rep["agents"].to_string(index=False),
          "\n## Most-failed questions\n", rep["questions"].head(8)[["qid", "question", "fail_rate", "critical"]].to_string(index=False)]
    return "\n".join(md)


def export_excel(path, rep: dict, runs: List[AuditRun], sheet: AuditSheet):
    qm = sheet.by_qid()
    detail = pd.DataFrame([dict(call_id=r.call_id, qid=x.qid, section=qm[x.qid].section, question=qm[x.qid].text, answer=x.answer,
                                confidence=x.confidence, evidence_turns=",".join(map(str, x.evidence_turns)), quote=x.quote, rationale=x.rationale)
                           for r in runs for x in r.results])
    with pd.ExcelWriter(path) as w:
        rep["calls"].to_excel(w, sheet_name="Calls", index=False); rep["agents"].to_excel(w, sheet_name="Agents", index=False)
        rep["questions"].to_excel(w, sheet_name="Questions", index=False); detail.to_excel(w, sheet_name="Detail", index=False)
    return path
