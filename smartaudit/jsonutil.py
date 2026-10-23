"""Tolerant JSON extraction - small models often add prose, fences or truncate."""
from __future__ import annotations
import json, re


def _balance(s: str) -> str:
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack: stack.pop()
    if in_str: s += '"'
    return s + "".join(reversed(stack))


def extract_json(text: str):
    """Return parsed JSON object or None."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    t = t[start:]
    for cand in (t, t[: t.rfind("}") + 1] if "}" in t else t, _balance(t)):
        for fix in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(fix)
            except Exception:
                continue
    return None
