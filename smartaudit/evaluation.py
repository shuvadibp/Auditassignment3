"""Model evaluation - everything from the 'Verdict' chapter: P/R/F1 on the rare class, kappa, calibration,
evidence validity, cost, latency, difficulty slices, consistency, cascade simulation."""
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, List
from .data_loader import load_gold
from .report import score_call
from .schemas import AuditRun, AuditSheet

LABELS = ["YES", "NO", "NA"]


def answers_df(runs: List[AuditRun], sheet: AuditSheet) -> pd.DataFrame:
    gold, qm, rows = load_gold(), sheet.by_qid(), []
    for r in runs:
        for x in r.results:
            g = gold[r.call_id][x.qid]
            hit = None
            if g["evidence_turns"] and x.evidence_turns:
                hit = bool(set(g["evidence_turns"]) & set(x.evidence_turns))
            rows.append(dict(engine=r.engine_key, call_id=r.call_id, qid=x.qid, section=qm[x.qid].section, critical=qm[x.qid].critical,
                             weight=qm[x.qid].weight, difficulty=g["difficulty"], variant=g["variant"], gold=g["answer"], pred=x.answer,
                             conf=x.confidence, ev_valid=x.evidence_valid, ev_hit=hit, rationale=x.rationale, source=x.source))
    return pd.DataFrame(rows)


def kappa(y, p, labels=LABELS) -> float:
    y, p = np.array(y), np.array(p)
    if len(y) == 0: return float("nan")
    po = (y == p).mean()
    pe = sum((y == l).mean() * (p == l).mean() for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def prf(df: pd.DataFrame, pos="NO"):
    tp = ((df.pred == pos) & (df.gold == pos)).sum(); fp = ((df.pred == pos) & (df.gold != pos)).sum(); fn = ((df.pred != pos) & (df.gold == pos)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def question_metrics(df: pd.DataFrame) -> dict:
    p, r, f1 = prf(df, "NO"); _, _, f1_yes = prf(df, "YES")
    crit = df[df.critical & (df.gold == "NO")]
    return dict(accuracy=(df.pred == df.gold).mean(), fail_precision=p, fail_recall=r, fail_f1=f1, macro_f1=(f1 + f1_yes) / 2,
                kappa=kappa(df.gold, df.pred), critical_recall=(crit.pred == "NO").mean() if len(crit) else float("nan"),
                na_accuracy=(df[df.gold == "NA"].pred == "NA").mean() if (df.gold == "NA").any() else float("nan"),
                subtle_acc=(df[df.difficulty == "subtle"].pred == df[df.difficulty == "subtle"].gold).mean(),
                error_rate=(df.pred == "ERROR").mean(),
                evidence_valid=df.ev_valid.dropna().astype(bool).mean() if df.ev_valid.notna().any() else float("nan"))


def ops_metrics(runs: List[AuditRun]) -> dict:
    lat = [x for r in runs for x in r.usage.latencies]
    n = max(1, len(runs))
    return dict(cost_per_call=sum(r.usage.cost_usd for r in runs) / n, cost_per_1k_calls=1000 * sum(r.usage.cost_usd for r in runs) / n,
                sec_per_call=sum(r.usage.latency_s for r in runs) / n, p95_latency_s=float(np.percentile(lat, 95)) if lat else 0.0,
                tokens_per_call=sum(r.usage.input_tokens + r.usage.output_tokens for r in runs) / n,
                api_calls_per_call=sum(r.usage.n_api_calls for r in runs) / n, json_failures=sum(r.parse_failures for r in runs))


def call_level(runs: List[AuditRun], sheet: AuditSheet) -> dict:
    gold, errs, agree, crit_tp, crit_fn = load_gold(), [], [], 0, 0
    for r in runs:
        ga = {q: v["answer"] for q, v in gold[r.call_id].items()}
        gs, ps = score_call(sheet, ga), score_call(sheet, r.answers())
        errs.append(abs(gs["total"] - ps["total"])); agree.append(gs["passed"] == ps["passed"])
    return dict(score_mae=float(np.mean(errs)), pass_fail_agreement=float(np.mean(agree)))


def compare(runs_by_engine: Dict[str, List[AuditRun]], sheet: AuditSheet) -> pd.DataFrame:
    rows = []
    for k, runs in runs_by_engine.items():
        df = answers_df(runs, sheet)
        rows.append(dict(engine=k, **question_metrics(df), **call_level(runs, sheet), **ops_metrics(runs)))
    return pd.DataFrame(rows).sort_values("fail_f1", ascending=False).reset_index(drop=True)


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df.gold, df.pred).reindex(index=LABELS, columns=LABELS + ["ERROR"], fill_value=0)


def by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(col):
        p, r, f1 = prf(g, "NO")
        out.append({col: k, "n": len(g), "accuracy": (g.pred == g.gold).mean(), "fail_recall": r, "fail_f1": f1})
    return pd.DataFrame(out)


def calibration(df: pd.DataFrame, bins=(0, .5, .7, .85, .95, 1.01)):
    d = df[df.conf.notna() & df.pred.isin(LABELS)].copy()
    if d.empty: return pd.DataFrame(), float("nan")
    d["bin"] = pd.cut(d.conf, list(bins), right=False)
    t = d.groupby("bin", observed=True).apply(lambda g: pd.Series(dict(n=len(g), avg_conf=g.conf.mean(), accuracy=(g.pred == g.gold).mean())), include_groups=False).reset_index()
    ece = float((t.n * (t.avg_conf - t.accuracy).abs()).sum() / t.n.sum())
    return t, ece


def consistency(runs_a: List[AuditRun], runs_b: List[AuditRun]) -> float:
    b = {r.call_id: r.answers() for r in runs_b}; same = tot = 0
    for r in runs_a:
        for q, a in r.answers().items():
            if q in b.get(r.call_id, {}): tot += 1; same += (a == b[r.call_id][q])
    return same / tot if tot else float("nan")


def simulate_cascade(small: List[AuditRun], large: List[AuditRun], sheet: AuditSheet, thresholds=None, escalate_critical_no=False):
    """Post-hoc cascade from cached runs: zero extra API cost. Question-level cost = run cost / questions."""
    gold = load_gold(); crit = {q.qid for q in sheet.questions if q.critical}
    L = {(r.call_id, x.qid): x for r in large for x in r.results}
    thresholds = thresholds if thresholds is not None else [0, .5, .6, .7, .8, .9, .95, 1.01]
    out = []
    for th in thresholds:
        rows, cost, esc, n = [], 0.0, 0, 0
        for r in small:
            for x in r.results:
                n += 1; use = x; cost += x.cost_usd
                if (x.confidence or 0) < th or x.answer == "ERROR" or (escalate_critical_no and x.qid in crit and x.answer == "NO"):
                    if (r.call_id, x.qid) in L: use = L[(r.call_id, x.qid)]; cost += use.cost_usd; esc += 1
                g = gold[r.call_id][x.qid]
                rows.append(dict(call_id=r.call_id, qid=x.qid, gold=g["answer"], pred=use.answer, critical=x.qid in crit))
        d = pd.DataFrame(rows); p, rc, f1 = prf(d, "NO")
        cd = d[d.critical & (d.gold == "NO")]
        out.append(dict(threshold=th, accuracy=(d.pred == d.gold).mean(), fail_f1=f1, fail_recall=rc,
                        critical_recall=(cd.pred == "NO").mean() if len(cd) else float("nan"), pct_escalated=100 * esc / max(1, n),
                        cost_per_call=cost / max(1, len(small))))
    return pd.DataFrame(out)
