"""Streamlit helpers (kept out of the backend so the backend runs anywhere)."""
from __future__ import annotations
import html
import pandas as pd
import streamlit as st
from .data_loader import load_calls, load_sheet, load_gold
from .runner import run_audits
from .registry import list_models

ANS_COLOR = {"YES": "#c8f7c5", "NO": "#f9c6c6", "NA": "#e3e3e3", "ERROR": "#ffe08a"}


def ready_models(tiers=None, include_hidden=False):
    return [m for m in list_models(include_hidden) if m["ready"] and (tiers is None or m["tier"] in tiers)]


def label_map():
    return {m["key"]: m["label"] for m in list_models(True)}


def pick_models(label, default=None, multi=True, key=None, tiers=None):
    ms = ready_models(tiers)
    opts = [m["key"] for m in ms]
    lm = label_map()
    if not opts:
        st.warning("No models ready."); return [] if multi else None
    default = [d for d in (default or []) if d in opts] or opts[:1]
    if multi:
        return st.multiselect(label, opts, default=default, format_func=lambda k: f"{lm[k]}  [{k}]", key=key)
    return st.selectbox(label, opts, index=opts.index(default[0]), format_func=lambda k: f"{lm[k]}  [{k}]", key=key)


def style_answers(df: pd.DataFrame, cols):
    f = lambda v: f"background-color: {ANS_COLOR.get(v, '')}" if v in ANS_COLOR else ""
    sty = df.style
    return (sty.map if hasattr(sty, "map") else sty.applymap)(f, subset=cols)


def render_transcript(call, highlight=(), gold_turns=()):
    hl, gt = set(highlight), set(gold_turns)
    for t in call.turns:
        if t.speaker == "SYSTEM":
            st.caption(f"⏸ {t.text}"); continue
        who = "assistant" if t.speaker == "AGENT" else "user"
        txt = html.escape(t.text)
        if t.n in hl: txt = f"<mark style='background:#ffe08a'>{txt}</mark>"
        if t.n in gt: txt = f"<span style='border-bottom:2px solid #2e7d32'>{txt}</span>"
        with st.chat_message(who, avatar="🎧" if who == "assistant" else "🙂"):
            st.markdown(f"<small>T{t.n} · {t.speaker.title()} · {int(t.t)}s</small><br>{txt}", unsafe_allow_html=True)


def run_with_progress(engine_key, call_ids, sheet_id, options, use_cache=True, label=None):
    """Batches run from the main thread so Streamlit widgets can be updated safely."""
    bar = st.progress(0.0, text=label or f"Running {engine_key}...")
    runs, step = [], 6
    for i in range(0, len(call_ids), step):
        runs += run_audits(engine_key, call_ids[i:i + step], sheet_id, options, use_cache)
        bar.progress(min(1.0, len(runs) / len(call_ids)), text=f"{label or engine_key}: {len(runs)}/{len(call_ids)} calls")
    bar.empty()
    return runs


from .ui_common_free import sample_ids as sample_call_ids
