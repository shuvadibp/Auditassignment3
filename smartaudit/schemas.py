from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class Turn:
    n: int
    speaker: str          # AGENT | CUSTOMER | SYSTEM
    text: str
    t: float = 0.0


@dataclass
class Call:
    call_id: str
    agent_id: str
    agent_name: str
    customer_name: str
    scenario: str
    date: str
    duration_sec: int
    turns: List[Turn]
    tags: List[str] = field(default_factory=list)

    def as_text(self, turns: Optional[List[Turn]] = None) -> str:
        return "\n".join(f"[T{t.n}] {t.speaker}: {t.text}" for t in (turns or self.turns))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Call":
        d = dict(d)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        return cls(**d)


@dataclass
class Question:
    qid: str
    section: str
    text: str
    weight: int = 1
    critical: bool = False
    guidance: str = ""
    region: str = "anywhere"       # anywhere | opening | closing  (used by SLM windowing)
    nli_hypothesis: Optional[str] = None


@dataclass
class AuditSheet:
    sheet_id: str
    title: str
    version: str
    pass_mark: float
    questions: List[Question]

    def sections(self) -> List[str]:
        out = []
        for q in self.questions:
            if q.section not in out:
                out.append(q.section)
        return out

    def by_qid(self) -> Dict[str, Question]:
        return {q.qid: q for q in self.questions}

    def subset(self, qids) -> "AuditSheet":
        qs = [q for q in self.questions if q.qid in set(qids)]
        return AuditSheet(self.sheet_id, self.title, self.version, self.pass_mark, qs)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditSheet":
        return cls(d["sheet_id"], d["title"], d["version"], d["pass_mark"],
                   [Question(**q) for q in d["questions"]])


@dataclass
class QuestionResult:
    qid: str
    answer: str                         # YES | NO | NA | ERROR
    confidence: Optional[float] = None
    evidence_turns: List[int] = field(default_factory=list)
    quote: str = ""
    rationale: str = ""
    evidence_valid: Optional[bool] = None   # did the cited turns/quote really exist?
    cost_usd: float = 0.0
    source: str = ""                        # which model produced it (useful for cascades)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    n_api_calls: int = 0
    latencies: List[float] = field(default_factory=list)

    def add(self, o: "Usage") -> "Usage":
        self.input_tokens += o.input_tokens
        self.output_tokens += o.output_tokens
        self.cost_usd += o.cost_usd
        self.latency_s += o.latency_s
        self.n_api_calls += o.n_api_calls
        self.latencies += o.latencies
        return self


@dataclass
class AuditRun:
    call_id: str
    sheet_id: str
    engine_key: str
    options: Dict[str, Any]
    results: List[QuestionResult]
    usage: Usage = field(default_factory=Usage)
    parse_failures: int = 0
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def answers(self) -> Dict[str, str]:
        return {r.qid: r.answer for r in self.results}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRun":
        d = dict(d)
        d["results"] = [QuestionResult(**r) for r in d["results"]]
        d["usage"] = Usage(**d["usage"])
        return cls(**d)
