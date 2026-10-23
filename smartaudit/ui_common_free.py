"""Streamlit-free helpers (notebooks / CLI)."""
import random
from .data_loader import load_calls


def sample_ids(n, seed=1):
    calls = load_calls(); ids = list(calls)
    if n >= len(ids): return ids
    tricky = [i for i in ids if set(calls[i].tags) & {"third_party", "vulnerable", "prompt_injection", "noisy_asr"}]
    rest = [i for i in ids if i not in tricky]
    r = random.Random(seed); r.shuffle(tricky); r.shuffle(rest)
    return (tricky[: max(1, n // 3)] + rest)[:n]
