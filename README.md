# SmartAudit Lab

A teaching code base that mimics a *SmartAudit.AI*-style call-audit platform for an **insurance help desk**:
agent ↔ customer call transcripts are graded against an **audit sheet** (behavioural, compliance, process…)
by interchangeable **baseline / SLM / LLM / cascade** engines, and the engines are **evaluated against ground truth**.

```
transcript → [PII redaction] → engine (keyword | SLM | LLM | cascade) → answers + evidence + confidence
          → evidence check (grounding) → scoring (sections, critical fails) → audit report
gold labels ─────────────────────────────────────────────────────────────→ evaluation (P/R/F1, κ, cost, latency)
```

## Run on Kaggle (recommended path)
1. Zip is `smartaudit-lab.zip`. Kaggle → *Datasets → New Dataset* → upload it. Create a notebook, **Add Input** → your dataset.
2. Notebook settings: **Internet On**; **Accelerator GPU T4 ×2** only if you want SLMs.
3. *Add-ons → Secrets*: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (tick "attached" for the notebook).
4. Open `SmartAudit_Lab_Kaggle.ipynb` (included) and run top to bottom. Cell **3A** starts the Streamlit app and prints a public
   `trycloudflare.com` link; cell **3B** runs the same engine from plain notebook code (best for the assignment).

Without keys/GPU everything still runs: *keyword baseline*, *lazy model* and two **SIMULATED** models work offline so you can rehearse.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # add keys
streamlit run app.py
python tests/test_backend.py && python tests/ui_smoke.py
```

## What's inside
| Path | Purpose |
|---|---|
| `data/` | 60 synthetic calls, 2 audit sheets (20 + 9 questions), knowledge base, **ground truth** (answer, difficulty, variant, evidence turns) |
| `smartaudit/datagen/` | Generator: plants each behaviour on purpose → labels are exact. Includes hard negatives, NA cases, noisy-ASR calls, 2 prompt-injection calls, 3rd-party callers, vulnerable customers |
| `smartaudit/clients/` | `anthropic`, `openai`, `hf` (local SLM on GPU), `ollama` |
| `smartaudit/engines/` | `llm_engine` (prompts, JSON repair, retry, evidence validation), `baseline` (regex + lazy), `simulated`, `cascade` |
| `smartaudit/evaluation.py` | accuracy, fail-class P/R/F1, Cohen κ, critical recall, calibration/ECE, difficulty slices, consistency, cascade replay |
| `smartaudit/advisor.py`, `slm_tools.py` | SLM-vs-LLM decision factors, cost break-even, SLM recipe (windowing, compact prompts) |
| `models.json` | **Edit model IDs & prices here** |
| `app.py` | Streamlit: Transcripts · Audit Sheet · Run Audit · Audit Report · Model Evaluation · SLM-vs-LLM Lab · Ask AI |
| `assignments/` | Student assignment + instructor guide |

## Choosing engines (keys in `models.json`)
`baseline-keyword`, `baseline-lazy`, `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6-astra`,
`hf-qwen2.5-1.5b|3b`, `hf-phi-3.5-mini`, `hf-qwen2.5-7b-4bit`, `ollama-*`, `sim-llm`, `sim-slm`, `human-gold`.
Cascade = a string key: `cascade:hf-qwen2.5-3b>claude-haiku-5-5@0.8` (append `!crit` to always escalate a critical NO).

## Regenerating data
`python scripts/generate_data.py --n 60 --seed 2026` (students can generate their own variant; delete `.cache` afterwards).

## Caveats
- Model IDs/prices were taken from public pages in Oct 2026 - verify in `models.json` before class.
- The regex baseline is written against the generator's phrasing, so it looks better than it would on real calls. Say this in class.
- Results are cached in `.cache/` (Kaggle: `/kaggle/working/.smartaudit_cache`), keyed by model + options + prompt + transcript, so re-runs cost nothing.
  Pre-run the expensive models once and ship the cache to students to save API spend.
