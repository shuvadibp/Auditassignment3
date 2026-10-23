"""Headless smoke test of app.py without Streamlit installed (stub). Usage: python tests/ui_smoke.py"""
import sys, os, types, runpy, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))


class D:
    def __call__(self, *a, **k): return D()
    def __getattr__(self, n): return D()
    def __getitem__(self, i): return D()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([])


class P(D):
    """Container proxy: widgets called on a column/tab/sidebar behave like the top-level ones."""
    def __init__(self, m): self._m = m
    def __getattr__(self, n):
        v = self._m.__dict__.get(n)
        return v if v is not None else D()


def make(page_idx, press):
    m = types.ModuleType("streamlit"); ss = {}
    m.session_state = ss
    PAGES = None
    m.radio = lambda label, options, *a, **k: options[page_idx]
    m.columns = lambda n, *a, **k: [P(m) for _ in range(n if isinstance(n, int) else len(n))]
    m.tabs = lambda labels: [P(m) for _ in labels]
    m.selectbox = lambda label, options, index=0, **k: list(options)[index] if len(list(options)) else None
    m.multiselect = lambda label, options, default=None, **k: default or []
    m.checkbox = lambda label, value=False, **k: value
    m.toggle = lambda label, value=False, **k: True            # instructor mode ON to exercise gold paths
    m.slider = lambda label, mn, mx, value=None, **k: mn if value is None else value
    m.number_input = lambda label, mn=0, mx=0, value=0, **k: value
    m.text_input = lambda label, value="", **k: value
    m.button = lambda *a, **k: press
    m.download_button = lambda *a, **k: False
    m.progress = lambda *a, **k: D()
    m.cache_data = lambda f=None, **k: (f if f else (lambda g: g))
    m.sidebar = P(m)
    m.__getattr__ = lambda n: D()
    return m


fails = 0
for press in (False, True):
    for i in range(8):
        sys.modules["streamlit"] = make(i, press)
        for mod in [k for k in sys.modules if k.startswith("smartaudit.ui_common")]: del sys.modules[mod]
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            print(f"page {i} press={press}: ok")
        except Exception:
            fails += 1; print(f"page {i} press={press}: FAIL"); traceback.print_exc()
print("FAILS:", fails); sys.exit(1 if fails else 0)
