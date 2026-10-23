"""CLI evaluation (no UI):  python scripts/run_evaluation.py --models baseline-keyword sim-slm claude-haiku-5-5 --n 20"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from smartaudit.config import load_secrets_into_env
from smartaudit.data_loader import load_sheet
from smartaudit.runner import run_audits
from smartaudit.ui_common_free import sample_ids
from smartaudit import evaluation as ev

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--strategy", default="per_question"); ap.add_argument("--context", default="full"); ap.add_argument("--compact", action="store_true")
a = ap.parse_args()
load_secrets_into_env(); sheet = load_sheet(); ids = sample_ids(a.n)
opts = dict(strategy=a.strategy, context=a.context, compact=a.compact)
res = {m: run_audits(m, ids, options=opts) for m in a.models}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ev.compare(res, sheet)[["engine", "accuracy", "fail_precision", "fail_recall", "fail_f1", "kappa", "critical_recall", "cost_per_1k_calls", "sec_per_call"]].round(3))
