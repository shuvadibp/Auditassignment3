# Instructor guide

## Suggested timing (matches the 180-min hands-on block)
| Min | Activity | App page / notebook cell |
|---|---|---|
| 0-15 | Tour: transcript, sheet, gold labels (instructor mode) | Transcripts, Audit Sheet |
| 15-40 | First LLM audit, evidence, prompt injection | Run Audit |
| 40-75 | Evaluation + accuracy trap | Model Evaluation |
| 75-115 | SLMs on GPU, recipe, windowing | SLM vs LLM Lab → toolbox, notebook §4 |
| 115-140 | Cascade + cost | Cascade simulator, Break-even |
| 140-155 | Advisor discussion | Decision advisor |
| 155-175 | Guardrails: redaction, injection | Run Audit with Redact |

## Prep checklist (do the day before)
1. Run the notebook once on Kaggle with your keys; verify model IDs in `models.json` actually respond (OpenAI/Anthropic IDs change often).
2. Pre-run expensive models on the 60 calls and ship `/kaggle/working/.smartaudit_cache` inside the dataset (set `SMARTAUDIT_CACHE`) - students replay for free.
3. Students without keys: use `sim-llm`/`sim-slm` (labelled SIMULATED - they flip gold answers at set error rates; say so).
4. Never put your API keys in the dataset; use Kaggle Secrets per student, or a shared budget-capped key.

## Teaching points hidden in the data
- **Accuracy trap**: `baseline-lazy` scores ≈0.66 accuracy on 60 calls with 0 failure recall.
- **Strong simple baseline**: the regex engine reaches fail-F1 ≈0.84 offline (measured), but it is tuned to synthetic phrasing and blind to fact-checks (E4).
- **Look-alikes**: "I've recorded your details" (D1), "I can't promise…" (D2), "sorry, I need your details" (B1), volunteered policy number (C1), firm-but-polite refusal (B3).
- **NA handling**: C2, D4, D5, E1, E3 - models often answer YES instead of NA.
- **Omission vs commission**: B2 (name never used) and F2 need the whole call; windowing hurts them.
- **Noisy ASR** (lowercase, no punctuation) hurts SLMs more than LLMs; **prompt injection** calls test instruction hierarchy.
- **Critical failures** matter more than the average → critical recall, cascade with `!crit`.

## Answer-key notes
Do not quote numbers for real models until you have run them; results depend on model versions. Expected *shape*: LLM ≫ SLM on E4, D5, B1-subtle; SLM competitive on A1, F1-F3, D1; calibration of self-reported confidence is poor for small models; cascade at ~0.8 usually recovers most of the LLM quality at a fraction of the cost.

## Extending
- New scenario/question: edit `datagen/generator.py` (SCEN, VARIANTS) and `datagen/sheet_def.py`, regenerate, rerun `tests/test_backend.py`.
- New provider: subclass `ModelClient` in `clients/`, add a branch in `registry.get_engine`, add an entry in `models.json`.
