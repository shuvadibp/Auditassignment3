"""Run a Small Language Model locally with Hugging Face transformers (works on a Kaggle T4/P100 GPU).
Models are cached in-process so several audits reuse the same weights."""
from __future__ import annotations
import time, threading
from .base import ModelClient, LLMResponse

_CACHE, _LOCK = {}, threading.Lock()


def _load(model_id: str, quant: str | None):
    with _LOCK:
        if model_id in _CACHE:
            return _CACHE[model_id]
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        kw = dict(device_map="auto")
        if quant == "4bit":                       # needs bitsandbytes; shrinks memory ~4x ("quantisation" slide)
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        model.eval()
        _CACHE[model_id] = (tok, model)
        return tok, model


class HFClient(ModelClient):
    def __init__(self, spec):
        super().__init__(spec)
        self.tok, self.model = _load(spec["model_id"], spec.get("quant"))
        self.lock = threading.Lock()              # one GPU generation at a time

    def complete(self, system, user, max_tokens=700, temperature=0.0, json_mode=True):
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with self.lock, torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=min(max_tokens, 700), do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][ids.shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return LLMResponse(text, int(ids.shape[1]), int(len(gen)), time.time() - t0)
