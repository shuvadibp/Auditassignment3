"""Helpers for working with SLMs - what makes small models work (and how to show it in class)."""
from __future__ import annotations
from .prompts import window_turns, build_user_prompt, SLM_SYSTEM, SYSTEM
from .schemas import Call, Question
from .pii import redact_call

SLM_RECIPE = {
    "one question per call": "SLMs lose track when asked 20 things at once -> strategy='per_question'",
    "short, windowed context": "send only the opening / closing turns when the question is about them -> context='regional'",
    "compact prompt": "short system prompt, rubric guidance only where needed -> compact=True",
    "greedy decoding + JSON": "temperature 0; validate JSON and retry once (jsonutil.extract_json repairs truncation)",
    "escalate doubt": "low confidence or critical NO -> ask a larger model (CascadeEngine)",
}
SLM_OPTIONS = dict(strategy="per_question", context="regional", compact=True)
LLM_OPTIONS = dict(strategy="full_sheet", context="full", compact=False)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)          # rule of thumb for English


def prompt_size_demo(call: Call, q: Question) -> dict:
    """Show students how windowing + compact prompts shrink what the model must read."""
    full = build_user_prompt(call.as_text(), [q], True, False)
    win = build_user_prompt(call.as_text(window_turns(call, q)), [q], True, True)
    return dict(full_tokens=approx_tokens(SYSTEM["v1"] + full), windowed_tokens=approx_tokens(SLM_SYSTEM + win),
                full_prompt=full, windowed_prompt=win)


def what_the_api_sees(call: Call) -> Call:
    return redact_call(call)
