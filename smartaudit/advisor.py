"""SLM vs LLM decision guide. Transparent, editable weights - students are asked to change them (Assignment Part 5)."""
from __future__ import annotations
from dataclasses import dataclass, field

# factor -> (label, help). Each factor is 1..5. Weights: positive => pushes towards SLM, negative => towards LLM.
FACTORS = {
 "privacy":     ("Data must stay in-house (privacy / regulation)", +2.0),
 "volume":      ("Call volume (1 = tiny, 5 = millions per month)", +1.5),
 "latency":     ("Latency / offline need (5 = real time or offline)", +1.0),
 "labels":      ("Labelled examples available (5 = thousands)", +1.0),
 "narrow":      ("Task is narrow & repetitive (5 = very narrow)", +1.5),
 "reasoning":   ("Reasoning depth needed (5 = multi-step judgement)", -2.0),
 "context":     ("Context length / document size (5 = very long)", -1.0),
 "open_ended":  ("Open-ended / varied inputs (5 = highly varied)", -1.5),
 "start_today": ("Need to start today with no training data (5 = yes)", -1.0),
 "criticality": ("Cost of a wrong answer (5 = severe, e.g. compliance)", -0.5),
}

PRESETS = {  # audit sub-tasks -> typical factor values
 "A1 Greeting check (pattern)":            dict(privacy=3, volume=5, latency=3, labels=4, narrow=5, reasoning=1, context=1, open_ended=1, start_today=2, criticality=1),
 "B1 Empathy detection (tone/semantics)":  dict(privacy=3, volume=5, latency=3, labels=3, narrow=4, reasoning=2, context=2, open_ended=3, start_today=3, criticality=2),
 "D1 Recording disclosure (critical)":     dict(privacy=3, volume=5, latency=3, labels=4, narrow=5, reasoning=1, context=1, open_ended=1, start_today=2, criticality=5),
 "E4 KB fact-check (grounded reasoning)":  dict(privacy=3, volume=4, latency=2, labels=2, narrow=2, reasoning=4, context=3, open_ended=3, start_today=4, criticality=4),
 "D5 Vulnerability handling (judgement)":  dict(privacy=3, volume=4, latency=2, labels=2, narrow=2, reasoning=4, context=3, open_ended=4, start_today=4, criticality=4),
 "Executive summary of 10k calls":         dict(privacy=3, volume=1, latency=1, labels=1, narrow=1, reasoning=4, context=5, open_ended=5, start_today=5, criticality=2),
 "Air-gapped bank, tone scoring":          dict(privacy=5, volume=5, latency=4, labels=4, narrow=4, reasoning=2, context=2, open_ended=2, start_today=2, criticality=3),
}


@dataclass
class Recommendation:
    verdict: str
    score: float
    reasons: list = field(default_factory=list)


def recommend(f: dict, weights: dict | None = None) -> Recommendation:
    w = {k: v[1] for k, v in FACTORS.items()}; w.update(weights or {})
    score, reasons = 0.0, []
    for k, val in f.items():
        c = w[k] * (val - 3)              # centred at 3 = neutral
        score += c
        if abs(c) >= 2: reasons.append(f"{'SLM' if c > 0 else 'LLM'} +{abs(c):.1f}: {FACTORS[k][0]} = {val}")
    hard_llm = f.get("reasoning", 0) >= 4 and f.get("criticality", 0) >= 4
    if f.get("privacy", 0) == 5 and f.get("reasoning", 0) >= 4:
        verdict = "SLM (self-hosted) + human review - privacy forces it; expect to invest in fine-tuning"
    elif abs(score) < 4 or (hard_llm and score > -4) or (f.get("criticality", 0) >= 5 and score > 0):
        verdict = "CASCADE: SLM first, escalate low-confidence / critical cases to an LLM"
    else:
        verdict = "SLM (small / fine-tuned model)" if score > 0 else "LLM (frontier / mid-tier API model)"
    return Recommendation(verdict, round(score, 1), sorted(reasons, key=lambda s: -float(s.split("+")[1].split(":")[0])))


def monthly_cost(calls_per_month, tokens_in_per_call, tokens_out_per_call, price_in, price_out,
                 gpu_usd_per_hour=0.0, gpu_hours_per_month=730, slm_sec_per_call=0.0):
    api = calls_per_month * (tokens_in_per_call * price_in + tokens_out_per_call * price_out) / 1e6
    selfhost_fixed = gpu_usd_per_hour * gpu_hours_per_month
    selfhost_busy = calls_per_month * slm_sec_per_call / 3600 * gpu_usd_per_hour
    return dict(api=api, selfhost=max(selfhost_fixed, selfhost_busy))


def break_even_calls(tokens_in, tokens_out, price_in, price_out, gpu_usd_per_hour, gpu_hours=730):
    per_call = (tokens_in * price_in + tokens_out * price_out) / 1e6
    return (gpu_usd_per_hour * gpu_hours / per_call) if per_call else float("inf")
