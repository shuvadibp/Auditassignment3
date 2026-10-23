"""Regenerate the synthetic dataset:  python scripts/generate_data.py --n 50 --seed 2026"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartaudit.datagen.generator import generate
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
g = generate(a.n, a.seed, Path(__file__).resolve().parent.parent / "data")
print(f"Generated {len(g)} calls")
