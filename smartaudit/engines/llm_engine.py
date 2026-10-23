"""LLM / SLM engine: builds prompts, calls a ModelClient, parses + validates JSON answers."""
from __future__ import annotations
from .base import AuditEngine, validate_evidence
from ..schemas import AuditRun, QuestionResult, Usage, Call
from ..prompts import SYSTEM, SLM_SYSTEM, build_user_prompt, window_turns
from ..jsonutil import extract_json
from ..pii import redact_call


class LLMAuditEngine(AuditEngine):
    def __init__(self, key: str, spec: dict, client):
        self.key, self.spec, self.client = key, spec, client

    def _ask(self, call: Call, qs, o, usage: Usage, windowed_turns=None):
        slm = self.spec.get("tier") == "slm"
        system = SLM_SYSTEM if (slm and o["compact"]) else SYSTEM.get(o["prompt_variant"], SYSTEM["v1"])
        text = call.as_text(windowed_turns)
        user = build_user_prompt(text, qs, include_kb=True, compact=o["compact"])
        parse_fail, data = 0, None
        for attempt in range(2):
            u = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
            resp = self.client.complete(system, u, max_tokens=3000 if len(qs) > 3 else 900, temperature=0.0)
            cost = self.client.cost(resp.input_tokens, resp.output_tokens)
            usage.add(Usage(resp.input_tokens, resp.output_tokens, cost, resp.latency_s, 1, [resp.latency_s]))
            data = extract_json(resp.text)
            items = data.get("answers") if isinstance(data, dict) else data if isinstance(data, list) else None
            if items is not None:
                return items, parse_fail, cost
            parse_fail += 1
        return [], parse_fail, 0.0

    def audit(self, call, sheet, options=None, question_ids=None):
        o = self._opts(options)
        qs_all = self._questions(sheet, question_ids)
        work = redact_call(call) if o["redact"] else call
        usage, results, pf = Usage(), {}, 0
        if o["strategy"] == "full_sheet":
            groups = [qs_all]
        elif o["strategy"] == "per_section":
            groups = [[q for q in qs_all if q.section == s] for s in dict.fromkeys(q.section for q in qs_all)]
        else:
            groups = [[q] for q in qs_all]
        err = ""
        for g in groups:
            turns = window_turns(work, g[0]) if (o["context"] == "regional" and len(g) == 1) else None
            try:
                items, f, cost = self._ask(work, g, o, usage, turns)
            except Exception as e:          # network/auth errors shouldn't kill the whole batch
                items, f, cost, err = [], 1, 0.0, f"{type(e).__name__}: {e}"[:300]
            pf += f
            valid = {q.qid for q in g}
            for it in items:
                if not isinstance(it, dict) or str(it.get("qid", "")).strip() not in valid:
                    continue
                qid = str(it["qid"]).strip()
                ans = str(it.get("answer", "")).strip().upper()
                ans = ans if ans in ("YES", "NO", "NA") else ("NA" if ans in ("N/A", "NOT APPLICABLE") else "ERROR")
                try: conf = float(it.get("confidence"))
                except Exception: conf = None
                raw_ev = it.get("evidence_turns") if isinstance(it.get("evidence_turns"), list) else []
                ev = [int(str(x).lstrip("Tt")) for x in raw_ev if str(x).lstrip("Tt").isdigit()]
                quote = str(it.get("quote", ""))[:300]
                results[qid] = QuestionResult(qid, ans, conf, ev, quote, str(it.get("rationale", ""))[:400],
                                              validate_evidence(work, ev, quote) if ans in ("YES", "NO") else None,
                                              cost / len(g), self.key)
        for q in qs_all:
            results.setdefault(q.qid, QuestionResult(q.qid, "ERROR", None, [], "", "no/invalid answer returned", None, 0.0, self.key))
        return AuditRun(call.call_id, sheet.sheet_id, self.key, o, [results[q.qid] for q in qs_all], usage, pf, err)
