from __future__ import annotations
import json, hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional
from .config import CACHE_DIR, PROMPT_VERSION
from .data_loader import load_call, load_sheet
from .engines.base import DEFAULT_OPTIONS
from .registry import get_engine, all_specs
from .schemas import AuditRun, QuestionResult, Usage


def _key(engine_key, call, sheet, options):
    blob = json.dumps([engine_key, call.call_id, sheet.sheet_id, sheet.version, [q.qid for q in sheet.questions],
                       options, PROMPT_VERSION, hashlib.md5(call.as_text().encode()).hexdigest()], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _path(engine_key, h):
    d = CACHE_DIR / "runs" / engine_key.replace(":", "_").replace(">", "_").replace("@", "_").replace("!", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def get_cached(engine_key, call_id, sheet_id, options=None) -> Optional[AuditRun]:
    o = {**DEFAULT_OPTIONS, **(options or {})}
    call, sheet = load_call(call_id), load_sheet(sheet_id)
    p = _path(engine_key, _key(engine_key, call, sheet, o))
    return AuditRun.from_dict(json.loads(p.read_text())) if p.exists() else None


def run_audits(engine_key: str, call_ids: List[str], sheet_id="insurance_helpdesk_v1", options=None,
               use_cache=True, max_workers: Optional[int] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> List[AuditRun]:
    o = {**DEFAULT_OPTIONS, **(options or {})}
    sheet, eng = load_sheet(sheet_id), get_engine(engine_key)
    base = engine_key.split(":")[1].split(">")[0] if engine_key.startswith("cascade:") else engine_key
    workers = max_workers or all_specs().get(base, {}).get("max_workers", 4)
    done = [0]

    def one(cid):
        call = load_call(cid); p = _path(engine_key, _key(engine_key, call, sheet, o))
        if use_cache and p.exists():
            run = AuditRun.from_dict(json.loads(p.read_text()))
        else:
            run = eng.audit(call, sheet, o)
            if not run.error and run.parse_failures == 0 or run.usage.n_api_calls == 0:
                p.write_text(json.dumps(run.to_dict()))
        done[0] += 1
        if progress: progress(done[0], len(call_ids))
        return run
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(one, call_ids))
