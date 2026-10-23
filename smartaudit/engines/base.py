from __future__ import annotations
import re
from typing import Optional, List
from ..schemas import Call, AuditSheet, AuditRun, QuestionResult, Usage

DEFAULT_OPTIONS = dict(strategy="per_question", redact=False, prompt_variant="v1", context="full", compact=False)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9$ ]", "", s.lower().replace("-", " ")).strip()


def validate_evidence(call: Call, turns: List[int], quote: str) -> Optional[bool]:
    """Grounding check: did the model cite turns/quotes that really exist? (hallucination guard)"""
    if not turns and not quote:
        return None
    by_n = {t.n: norm(t.text) for t in call.turns}
    if any(n not in by_n for n in turns):
        return False
    if not quote:
        return True
    q = set(norm(quote).split())
    if not q:
        return True
    hay = " ".join(by_n[n] for n in turns) if turns else " ".join(by_n.values())
    return len(q & set(hay.split())) / len(q) >= 0.8


class AuditEngine:
    key: str = "engine"
    spec: dict = {}

    def audit(self, call: Call, sheet: AuditSheet, options: Optional[dict] = None,
              question_ids: Optional[List[str]] = None) -> AuditRun:
        raise NotImplementedError

    def _opts(self, options):
        return {**DEFAULT_OPTIONS, **(options or {})}

    def _questions(self, sheet, question_ids):
        return [q for q in sheet.questions if question_ids is None or q.qid in question_ids]
