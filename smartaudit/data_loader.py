from __future__ import annotations
import json
from functools import lru_cache
from typing import Dict, List
from .config import DATA_DIR
from .schemas import Call, AuditSheet


@lru_cache(maxsize=1)
def load_calls() -> Dict[str, Call]:
    out = {}
    for f in sorted((DATA_DIR / "transcripts").glob("*.json")):
        c = Call.from_dict(json.loads(f.read_text()))
        out[c.call_id] = c
    return out


def load_call(call_id: str) -> Call:
    return load_calls()[call_id]


def list_sheets() -> List[str]:
    return [f.stem for f in sorted((DATA_DIR / "audit_sheets").glob("*.json"))]


@lru_cache(maxsize=8)
def load_sheet(sheet_id: str = "insurance_helpdesk_v1") -> AuditSheet:
    return AuditSheet.from_dict(json.loads((DATA_DIR / "audit_sheets" / f"{sheet_id}.json").read_text()))


@lru_cache(maxsize=1)
def load_gold() -> dict:
    p = DATA_DIR / "ground_truth.json"
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=1)
def load_kb() -> list:
    return json.loads((DATA_DIR / "knowledge_base.json").read_text())


def kb_text() -> str:
    return "\n".join(f"- {k['id']}: {k['text']}" for k in load_kb())
