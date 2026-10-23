"""Simulated models (NO API calls) for dry runs/tests, plus the 'human gold' engine.
SIM engines flip gold answers with a set error rate - they are NOT real models; label them as such in class."""
from __future__ import annotations
import random, hashlib
from .base import AuditEngine
from ..schemas import AuditRun, QuestionResult, Usage
from ..data_loader import load_gold


class GoldEngine(AuditEngine):
    def __init__(self, key, spec): self.key, self.spec = key, spec

    def audit(self, call, sheet, options=None, question_ids=None):
        g = load_gold()[call.call_id]
        res = [QuestionResult(q.qid, g[q.qid]["answer"], 1.0, g[q.qid]["evidence_turns"], "", "human gold label", True, 0.0, self.key)
               for q in self._questions(sheet, question_ids)]
        return AuditRun(call.call_id, sheet.sheet_id, self.key, self._opts(options), res, Usage())


class SimEngine(AuditEngine):
    def __init__(self, key, spec): self.key, self.spec = key, spec

    def audit(self, call, sheet, options=None, question_ids=None):
        g, sp = load_gold()[call.call_id], self.spec
        usage, res = Usage(), []
        for q in self._questions(sheet, question_ids):
            seed = int(hashlib.md5(f"{self.key}{call.call_id}{q.qid}".encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            gold, diff = g[q.qid]["answer"], g[q.qid]["difficulty"]
            wrong = rng.random() < sp["err_subtle" if diff == "subtle" else "err_easy"]
            ans = gold
            if wrong:
                ans = {"YES": "NO", "NO": "YES", "NA": rng.choice(["YES", "NO"])}[gold]
            conf = rng.uniform(.5, .92) if wrong else rng.uniform(.72, .99)
            inp, out = 450 + len(call.turns) * 25, 60
            lat = sp.get("latency", 1.0) * rng.uniform(.7, 1.4)
            cost = inp / 1e6 * sp.get("price_in", 0) + out / 1e6 * sp.get("price_out", 0)
            usage.add(Usage(inp, out, cost, lat, 1, [lat]))
            res.append(QuestionResult(q.qid, ans, round(conf, 2), [] if wrong else g[q.qid]["evidence_turns"], "",
                                      "simulated", None if ans == "NA" else (not wrong), cost, self.key))
        return AuditRun(call.call_id, sheet.sheet_id, self.key, self._opts(options), res, usage)
