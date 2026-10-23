from __future__ import annotations
import json, importlib.util
from functools import lru_cache
from .config import MODELS_JSON, get_secret
from .engines.baseline import KeywordBaselineEngine, LazyEngine
from .engines.simulated import SimEngine, GoldEngine
from .engines.llm_engine import LLMAuditEngine
from .engines.cascade import CascadeEngine


@lru_cache(maxsize=1)
def all_specs() -> dict:
    return {m["key"]: m for m in json.loads(MODELS_JSON.read_text())["models"]}


def _has(mod): return importlib.util.find_spec(mod) is not None


def readiness(spec: dict) -> tuple[bool, str]:
    p = spec["provider"]
    if p in ("baseline", "lazy", "sim", "gold"): return True, "ready (offline)"
    if p == "anthropic":
        if not _has("anthropic"): return False, "pip install anthropic"
        return (True, "ready") if get_secret("ANTHROPIC_API_KEY") else (False, "ANTHROPIC_API_KEY missing")
    if p == "openai":
        if not _has("openai"): return False, "pip install openai"
        return (True, "ready") if get_secret("OPENAI_API_KEY") else (False, "OPENAI_API_KEY missing")
    if p == "hf":
        if not _has("transformers") or not _has("torch"): return False, "pip install transformers torch accelerate"
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        return True, "ready (GPU)" if gpu else "ready (CPU - slow! enable GPU)"
    if p == "ollama":
        from .clients.ollama_client import ollama_up
        return (True, "ready") if ollama_up() else (False, "Ollama server not running")
    return False, "unknown provider"


def list_models(include_hidden=False):
    out = []
    for k, s in all_specs().items():
        if s.get("hidden_in_eval") and not include_hidden: continue
        ok, why = readiness(s)
        out.append({**s, "ready": ok, "status": why})
    return out


_ENGINES: dict = {}


def get_engine(key: str):
    if key in _ENGINES: return _ENGINES[key]
    if key.startswith("cascade:"):                      # cascade:<small>><large>@<thr>[!crit]
        body = key[8:]
        crit = body.endswith("!crit"); body = body.replace("!crit", "")
        pair, thr = body.split("@")
        s, l = pair.split(">")
        eng = CascadeEngine(key, get_engine(s), get_engine(l), float(thr), crit)
    else:
        spec = all_specs()[key]; p = spec["provider"]
        if p == "baseline": eng = KeywordBaselineEngine(key, spec)
        elif p == "lazy": eng = LazyEngine(key, spec)
        elif p == "sim": eng = SimEngine(key, spec)
        elif p == "gold": eng = GoldEngine(key, spec)
        else:
            if p == "anthropic":
                from .clients.anthropic_client import AnthropicClient as C
            elif p == "openai":
                from .clients.openai_client import OpenAIClient as C
            elif p == "hf":
                from .clients.hf_client import HFClient as C
            else:
                from .clients.ollama_client import OllamaClient as C
            eng = LLMAuditEngine(key, spec, C(spec))
        eng.spec = spec
    _ENGINES[key] = eng
    return eng


def get_client(key: str):
    """Raw text client (for Ask-AI and the LLM-judge)."""
    e = get_engine(key)
    if not hasattr(e, "client"): raise ValueError(f"{key} is not a generative model")
    return e.client
