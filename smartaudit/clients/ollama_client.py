"""Optional: local SLMs through Ollama (laptop use). On Kaggle prefer the HF client."""
from __future__ import annotations
import json, time, urllib.request, os
from .base import ModelClient, LLMResponse

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_up(timeout=1.0) -> bool:
    try:
        urllib.request.urlopen(HOST + "/api/tags", timeout=timeout); return True
    except Exception:
        return False


class OllamaClient(ModelClient):
    def complete(self, system, user, max_tokens=700, temperature=0.0, json_mode=True):
        body = {"model": self.spec["model_id"], "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}}
        if json_mode: body["format"] = "json"
        if self.spec.get("extra", {}).get("think") is False: body["think"] = False
        req = urllib.request.Request(HOST + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        return LLMResponse(r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0), time.time() - t0)
