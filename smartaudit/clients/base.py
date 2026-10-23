from __future__ import annotations
import time, random
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ModelClient:
    """Anything that turns (system, user) into text."""
    def __init__(self, spec: dict):
        self.spec = spec

    def complete(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.0,
                 json_mode: bool = True) -> LLMResponse:
        raise NotImplementedError

    def cost(self, inp: int, out: int) -> float:
        return inp / 1e6 * self.spec.get("price_in", 0) + out / 1e6 * self.spec.get("price_out", 0)


def with_retries(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(x in msg for x in ("429", "rate", "overloaded", "timeout", "500", "502", "503", "529"))
            if i == tries - 1 or not retryable:
                raise
            time.sleep(2 ** i + random.random())
