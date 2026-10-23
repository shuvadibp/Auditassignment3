"""SLM-first cascade: small model answers everything; only doubtful / critical answers go to the big model."""
from __future__ import annotations
from .base import AuditEngine
from ..schemas import AuditRun, Usage


class CascadeEngine(AuditEngine):
    def __init__(self, key, small: AuditEngine, large: AuditEngine, threshold=0.8, escalate_critical_no=False):
        self.key, self.small, self.large = key, small, large
        self.threshold, self.esc_crit = threshold, escalate_critical_no
        self.spec = {"tier": "cascade", "label": key}

    def audit(self, call, sheet, options=None, question_ids=None):
        o = {**self._opts(options), "strategy": "per_question"}
        r1 = self.small.audit(call, sheet, o, question_ids)
        crit = {q.qid for q in sheet.questions if q.critical}
        esc = [r.qid for r in r1.results if r.answer == "ERROR" or (r.confidence or 0) < self.threshold
               or (self.esc_crit and r.qid in crit and r.answer == "NO")]
        usage = Usage().add(r1.usage)
        res = {r.qid: r for r in r1.results}
        meta = {"escalated": esc, "n_questions": len(r1.results)}
        if esc:
            r2 = self.large.audit(call, sheet, {**o, "strategy": "per_question"}, esc)
            usage.add(r2.usage)
            for r in r2.results:
                res[r.qid] = r
        return AuditRun(call.call_id, sheet.sheet_id, self.key, o, [res[r.qid] for r in r1.results], usage,
                        r1.parse_failures, r1.error, meta)
