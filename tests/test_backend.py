"""Run:  python tests/test_backend.py   (no pytest needed, no API keys needed)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartaudit.jsonutil import extract_json
from smartaudit.pii import redact_call
from smartaudit.data_loader import load_calls, load_sheet, load_gold
from smartaudit.clients.base import ModelClient, LLMResponse
from smartaudit.engines.llm_engine import LLMAuditEngine
from smartaudit.engines.cascade import CascadeEngine
from smartaudit.engines.simulated import SimEngine
from smartaudit.runner import run_audits
from smartaudit import evaluation as ev, report as rp

CALLS, SHEET, GOLD = load_calls(), load_sheet(), load_gold()


def test_json():
    assert extract_json('Sure! ```json\n{"a": [1,2,]}\n```')["a"] == [1, 2]
    assert extract_json('{"answers":[{"qid":"A1","answer":"YES"')["answers"][0]["qid"] == "A1"   # truncated
    assert extract_json("no json here") is None


def test_pii():
    c = next(iter(CALLS.values())); r = redact_call(c)
    txt = r.as_text()
    assert "OS-" not in txt and c.customer_name.split()[-1] not in txt


def test_gold_complete():
    assert len(GOLD) == len(CALLS)
    for cid, g in GOLD.items():
        assert set(g) == {q.qid for q in SHEET.questions}


class OracleClient(ModelClient):
    """Fake 'LLM' that answers from gold and sometimes returns broken JSON - tests parsing/retry/evidence."""
    def __init__(self, break_first=False):
        super().__init__(dict(price_in=1, price_out=1)); self.n = 0; self.break_first = break_first

    def complete(self, system, user, **kw):
        self.n += 1
        cid = next(c for c in CALLS if CALLS[c].as_text() in user)
        qids = user.split("Return the JSON for question ids: ")[1].split("\n")[0].rstrip(".").split(", ")
        if self.break_first and self.n % 2 == 1: return LLMResponse("Sorry I cannot", 10, 5, .1)
        items = [dict(qid=q, answer=GOLD[cid][q]["answer"], confidence=.9, evidence_turns=GOLD[cid][q]["evidence_turns"][:1],
                      quote="", rationale="oracle") for q in qids]
        return LLMResponse("```json\n" + json.dumps({"answers": items}) + "\n```", 1000, 200, .5)


def _acc(run, cid): return sum(run.answers()[q] == GOLD[cid][q]["answer"] for q in run.answers()) / len(run.answers())


def test_llm_engine_strategies():
    cid = list(CALLS)[3]; call = CALLS[cid]
    for strat in ("full_sheet", "per_section", "per_question"):
        eng = LLMAuditEngine("oracle", dict(tier="llm"), OracleClient())
        r = eng.audit(call, SHEET, dict(strategy=strat))
        assert _acc(r, cid) == 1.0, strat
        assert r.usage.n_api_calls == {"full_sheet": 1, "per_section": len(SHEET.sections()), "per_question": len(SHEET.questions)}[strat]


def test_retry_on_bad_json():
    cid = list(CALLS)[0]
    r = LLMAuditEngine("oracle", dict(tier="llm"), OracleClient(break_first=True)).audit(CALLS[cid], SHEET, dict(strategy="full_sheet"))
    assert r.parse_failures == 1 and _acc(r, cid) == 1.0


def test_cascade_and_eval():
    ids = list(CALLS)[:15]
    R = {k: run_audits(k, ids, use_cache=False) for k in ("baseline-lazy", "baseline-keyword", "sim-slm", "sim-llm")}
    t = ev.compare(R, SHEET).set_index("engine")
    assert t.loc["baseline-lazy", "fail_recall"] == 0 and t.loc["baseline-lazy", "accuracy"] > .5      # accuracy lies
    assert t.loc["sim-llm", "fail_f1"] > t.loc["sim-slm", "fail_f1"]
    sim = ev.simulate_cascade(R["sim-slm"], R["sim-llm"], SHEET)
    assert sim.iloc[0].pct_escalated == 0 and sim.iloc[-1].pct_escalated == 100
    casc = CascadeEngine("c", SimEngine("sim-slm", {"err_easy": .1, "err_subtle": .4}), SimEngine("sim-llm", {"err_easy": 0, "err_subtle": 0}), .8)
    r = casc.audit(CALLS[ids[0]], SHEET); assert "escalated" in r.meta


def test_report():
    ids = list(CALLS)[:10]; runs = run_audits("human-gold", ids, use_cache=False)
    rep = rp.batch_report(runs, CALLS, SHEET); assert rep["kpis"]["n"] == 10 and len(rep["agents"]) >= 1


if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e:
                fails += 1; print("FAIL", n, type(e).__name__, e)
    sys.exit(fails)
