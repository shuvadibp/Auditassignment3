from __future__ import annotations
import time
from .base import ModelClient, LLMResponse, with_retries
from ..config import get_secret


class AnthropicClient(ModelClient):
    def __init__(self, spec):
        super().__init__(spec)
        import anthropic                      # lazy import
        self._mod = anthropic
        self.client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

    def complete(self, system, user, max_tokens=2048, temperature=0.0, json_mode=True):
        kw = dict(model=self.spec["model_id"], max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}], **self.spec.get("extra", {}))
        if self.spec.get("supports_temperature", True):
            kw["temperature"] = temperature
        t0 = time.time()

        def call():
            try:
                return self.client.messages.create(**kw)
            except self._mod.BadRequestError as e:      # some models reject temperature
                if "temperature" in str(e) and "temperature" in kw:
                    kw.pop("temperature")
                    return self.client.messages.create(**kw)
                raise
        r = with_retries(call)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens, time.time() - t0)
