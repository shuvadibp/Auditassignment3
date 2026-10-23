"""Guardrail: redact personal data BEFORE a transcript leaves your network.
Tokens keep the *fact* that something was said (so auditability survives)."""
from __future__ import annotations
import re
from .schemas import Call, Turn

PATTERNS = [
    (r"\bOS-\d{6}\b", "[POLICY_NO]"),
    (r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december),?\s+(?:19|20)\d{2}\b", "[DOB]"),
    (r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b", "[DOB]"),
    (r"\b[A-Z]{2}\d\s?\d[A-Z]{2}\b", "[POSTCODE]"),
    (r"\b[a-z]{2}\d\s?\d[a-z]{2}\b", "[POSTCODE]"),
    (r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b", "[VEHICLE_REG]"),
    (r"\b[\w.]+@[\w.]+\.\w+\b", "[EMAIL]"),
    (r"\b(?:\d[ -]?){13,19}\b", "[CARD_NO]"),
    (r"\b555[- ]?\d{4}\b", "[PHONE]"),
]


def redact_text(text: str, names=()) -> str:
    out = text
    for n in sorted({x for x in names if x}, key=len, reverse=True):
        out = re.sub(re.escape(n), "[CUSTOMER_NAME]", out, flags=re.I)
    for pat, rep in PATTERNS:
        flags = re.I if rep in ("[DOB]", "[POSTCODE]", "[VEHICLE_REG]") else 0
        out = re.sub(pat, rep, out, flags=flags)
    return out


def redact_call(call: Call) -> Call:
    parts = call.customer_name.split()
    names = [call.customer_name] + parts + ([f"{parts[0]} {parts[-1]}"] if parts else [])
    # titles: "Mr Rahman"
    if len(parts) >= 2:
        for title in ("Mr", "Ms", "Mrs"):
            names.append(f"{title} {parts[-1]}")
    turns = [Turn(t.n, t.speaker, redact_text(t.text, names), t.t) for t in call.turns]
    return Call(call.call_id, call.agent_id, call.agent_name, "[CUSTOMER_NAME]",
                call.scenario, call.date, call.duration_sec, turns, list(call.tags))
