from __future__ import annotations
import time
from .base import ModelClient, LLMResponse, with_retries
from ..config import get_secret


class OpenAIClient(ModelClient):
    def __init__(self, spec):
        super().__init__(spec)
        import openai
        self._mod = openai
        self.client = openai.OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    def complete(self, system, user, max_tokens=2048, temperature=0.0, json_mode=True):
        kw = dict(model=self.spec["model_id"],
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  max_completion_tokens=max_tokens, **self.spec.get("extra", {}))
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self.spec.get("supports_temperature", True):
            kw["temperature"] = temperature
        t0 = time.time()

        def call():
            for _ in range(4):                  # drop params this model rejects, then retry
                try:
                    return self.client.chat.completions.create(**kw)
                except self._mod.BadRequestError as e:
                    m = str(e)
                    if "temperature" in m and "temperature" in kw: kw.pop("temperature")
                    elif "response_format" in m and "response_format" in kw: kw.pop("response_format")
                    elif "max_completion_tokens" in m and "max_completion_tokens" in kw:
                        kw["max_tokens"] = kw.pop("max_completion_tokens")
                    else:
                        raise
            raise RuntimeError("OpenAI request kept failing")
        r = with_retries(call)
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, time.time() - t0)
