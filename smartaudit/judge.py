"""LLM-as-judge with the two classic checks: faithfulness rating + position-bias test (swap order)."""
from __future__ import annotations
from .jsonutil import extract_json
from .data_loader import load_call

J_SYS = """You are a strict reviewer. Rate whether an auditor's rationale is FAITHFUL to the transcript (every claim supported).
Reply JSON only: {"faithfulness": 1-5, "comment": "short"}."""
P_SYS = """You compare two audit rationales for the same question and pick the better one (accurate, grounded, concise).
Reply JSON only: {"winner": "A" or "B"}."""


def rate_rationale(client, call_id, question_text, answer, rationale) -> dict:
    call = load_call(call_id)
    u = f"<transcript>\n{call.as_text()}\n</transcript>\nQuestion: {question_text}\nAuditor answer: {answer}\nRationale: {rationale}"
    return extract_json(client.complete(J_SYS, u, max_tokens=300).text) or {}


def position_bias(client, call_id, question_text, r1, r2) -> dict:
    """Ask twice with A/B swapped. A consistent judge picks the same *rationale* both times."""
    call = load_call(call_id)
    def ask(a, b):
        u = f"<transcript>\n{call.as_text()}\n</transcript>\nQuestion: {question_text}\nA: {a}\nB: {b}"
        return (extract_json(client.complete(P_SYS, u, max_tokens=100).text) or {}).get("winner")
    w1, w2 = ask(r1, r2), ask(r2, r1)
    first = (w1 == "A", w2 == "A")           # picked first-listed both times -> position bias
    return dict(order1=w1, order2=w2, consistent=(w1 == "A") != (w2 == "A"), always_first=all(first))
