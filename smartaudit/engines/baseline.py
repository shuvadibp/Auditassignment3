"""Non-LLM engines. 'Always keep a strong, simple baseline' - and the lazy model shows why accuracy lies."""
from __future__ import annotations
import re
from .base import AuditEngine, norm
from ..schemas import AuditRun, QuestionResult, Usage


class LazyEngine(AuditEngine):
    """Answers YES to everything. High accuracy, zero recall on failures - the 'accuracy is a liar' demo."""
    def __init__(self, key, spec): self.key, self.spec = key, spec

    def audit(self, call, sheet, options=None, question_ids=None):
        qs = self._questions(sheet, question_ids)
        return AuditRun(call.call_id, sheet.sheet_id, self.key, self._opts(options),
                        [QuestionResult(q.qid, "YES", 1.0, [], "", "always YES", None, 0.0, self.key) for q in qs], Usage())


def _agent(call): return [(t.n, norm(t.text)) for t in call.turns if t.speaker == "AGENT"]
def _cust(call): return " ".join(norm(t.text) for t in call.turns if t.speaker == "CUSTOMER")
def _any(call, pat):
    return [n for n, t in _agent(call) if re.search(pat, t)]


def rules(call):
    ag, cu = _agent(call), _cust(call)
    atxt = " ".join(t for _, t in ag)
    first, last = (call.customer_name.split()[0].lower(), call.customer_name.split()[-1].lower())
    hold = any("on hold" in t.text.lower() for t in call.turns if t.speaker == "SYSTEM")
    claim = call.scenario in ("motor_claim", "home_claim")
    r = {}
    r["A1"] = ("YES" if ag and "orion shield" in ag[0][1] and call.agent_name.split()[0].lower() in ag[0][1] else "NO", [ag[0][0]] if ag else [])
    emp = _any(call, r"sorry|must be|sounds|imagine")
    r["B1"] = ("YES" if [n for n in emp if n <= ag[2][0]] else "NO", emp[:1])       # naive: any 'sorry' early
    nm = _any(call, rf"\b({first}|{last})\b")
    r["B2"] = ("YES" if nm else "NO", nm[:1])
    rude = _any(call, r"as i already said|not my problem|should have read|really should have")
    r["B3"] = ("NO" if rude else "YES", rude[:1])
    jar = _any(call, r"subrogation|indemnity|underwriting review")
    r["B4"] = ("NO" if jar and "what does" in cu else "YES", jar[:1])
    rep = _any(call, r"again from the start")
    r["B5"] = ("NO" if rep else "YES", rep[:1])
    ids = _any(call, r"date of birth|postcode|policy number|full name")
    r["C1"] = ("NA" if _any(call, "policyholder") else ("YES" if ids else "NO"), ids[:1])
    tp = _any(call, "policyholder")
    r["C2"] = (("NO" if _any(call, r"i can see the policy") else "YES") if tp else "NA", [])
    rec = _any(call, "recorded")
    r["D1"] = ("YES" if rec else "NO", rec[:1])                                        # fooled by 'I've recorded your details'
    gu = _any(call, r"guarantee|promise|definitely|no way")
    r["D2"] = ("NO" if gu else "YES", gu[:1])                                           # fooled by "I can't promise"
    esc = re.search(r"complain|manager", cu)
    r["D4"] = (("YES" if _any(call, "supervisor") else "NO") if esc else "NA", [])
    vul = re.search(r"hard of hearing|passed away|lost my job", cu)
    r["D5"] = (("YES" if _any(call, "priority support") else "NO") if vul else "NA", [])
    bad = _any(call, r"password|three digits|card number")
    r["D6"] = ("NO" if bad else "YES", bad[:1])
    r["E1"] = (("YES" if _any(call, "injured") and _any(call, r"\bwhen\b") else "NO") if claim else "NA", [])
    ref = _any(call, r"(clm|req) \d+|reference")
    r["E2"] = ("YES" if ref else "NO", ref[:1])
    r["E3"] = (("YES" if _any(call, "would you mind") and _any(call, r"thank you for holding|thanks for holding") else "NO") if hold else "NA", [])
    r["E4"] = ("YES", [])                                                                # cannot verify facts: knows nothing about the KB
    last_ag = ag[-3:]
    r["F1"] = ("YES" if any("recap" in t for _, t in last_ag) else "NO", [])
    r["F2"] = ("YES" if any("anything else" in t for _, t in ag[-4:]) else "NO", [])
    r["F3"] = ("YES" if ag and re.search(r"have a good day|thank you for calling", ag[-1][1]) and call.turns[-1].speaker != "AGENT" or (ag and "have a good day" in ag[-1][1]) else "NO", [])
    return r


class KeywordBaselineEngine(AuditEngine):
    """Regex rules. Fast, free, private, explainable - and blind to nuance."""
    def __init__(self, key, spec): self.key, self.spec = key, spec

    def audit(self, call, sheet, options=None, question_ids=None):
        qs, r = self._questions(sheet, question_ids), rules(call)
        res = [QuestionResult(q.qid, r[q.qid][0], 0.6, r[q.qid][1], "", "regex rule", None, 0.0, self.key) for q in qs]
        return AuditRun(call.call_id, sheet.sheet_id, self.key, self._opts(options), res, Usage(latencies=[0.001]))
