"""SmartAudit Lab - Streamlit front end.   Run:  streamlit run app.py"""
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartAudit Lab", page_icon="🎧", layout="wide")

from smartaudit.config import load_secrets_into_env
from smartaudit.data_loader import load_calls, load_sheet, list_sheets, load_gold, load_kb
from smartaudit.registry import list_models, get_engine, get_client
from smartaudit.runner import run_audits
from smartaudit import evaluation as ev, report as rp, advisor as adv, slm_tools
from smartaudit.engines.base import DEFAULT_OPTIONS
from smartaudit.ui_common import (pick_models, style_answers, render_transcript, run_with_progress,
                                  sample_call_ids, label_map, ready_models)
from smartaudit.pii import redact_call

load_secrets_into_env()
CALLS, GOLD = load_calls(), load_gold()
PAGES = ["🏠 Home", "📞 Transcripts", "📋 Audit Sheet", "▶️ Run Audit", "📊 Audit Report",
         "🧪 Model Evaluation", "⚖️ SLM vs LLM Lab", "💬 Ask AI"]
st.session_state.setdefault("runs", {})

with st.sidebar:
    st.title("🎧 SmartAudit Lab")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    sheet_id = st.selectbox("Audit sheet", list_sheets())
    instructor = st.toggle("Instructor mode (show ground truth)", value=False)
    st.caption("Models ready: " + ", ".join(m["key"] for m in ready_models()))
SHEET = load_sheet(sheet_id)
LM = label_map()


def options_ui(prefix, slm_hint=True):
    c = st.columns(4)
    strat = c[0].selectbox("Strategy", ["per_question", "per_section", "full_sheet"], key=prefix + "s",
                           help="per_question = 1 API call per audit question (best for SLMs, costs more calls)")
    ctx = c[1].selectbox("Context", ["full", "regional"], key=prefix + "c", help="regional = send only opening/closing turns when relevant")
    pv = c[2].selectbox("Prompt variant", ["v1", "v1b"], key=prefix + "p", help="Same task, reworded - tests prompt sensitivity")
    red = c[3].checkbox("Redact PII first", key=prefix + "r")
    comp = st.checkbox("Compact prompt (SLM style)", key=prefix + "k")
    return dict(strategy=strat, context=ctx, prompt_variant=pv, redact=red, compact=comp)


# ------------------------------------------------------------------ HOME
if page == PAGES[0]:
    st.header("SmartAudit Lab: auditing insurance help-desk calls with LLMs and SLMs")
    c = st.columns(4)
    c[0].metric("Calls", len(CALLS)); c[1].metric("Audit questions", len(SHEET.questions))
    c[2].metric("Sections", len(SHEET.sections())); c[3].metric("Critical questions", sum(q.critical for q in SHEET.questions))
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff"];
      T[label="Call transcript\\n(ASR output)"]; R[label="PII redaction\\n(guardrail)"]; E[label="Audit engine\\nbaseline | SLM | LLM | cascade"];
      A[label="Answers + evidence\\n+ confidence"]; V[label="Evidence check\\n(grounding)"]; S[label="Scoring\\nsections, critical fails"]; P[label="Audit report\\n+ human review"];
      G[label="Gold labels", fillcolor="#e8f5e9"]; M[label="Evaluation\\nP/R/F1, cost, latency", fillcolor="#fff3e0"];
      T->R->E->A->V->S->P; A->M; G->M; }""")
    st.subheader("Model status")
    df = pd.DataFrame(list_models())[["key", "label", "tier", "ready", "status"]]
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.info("Tip: models marked not-ready need an API key (Kaggle: Add-ons → Secrets) or a GPU. The offline baselines and SIMULATED models always work, so you can rehearse the whole lab without keys.")

# ------------------------------------------------------------------ TRANSCRIPTS
elif page == PAGES[1]:
    st.header("Call transcripts")
    c = st.columns(3)
    scen = c[0].multiselect("Scenario", sorted({x.scenario for x in CALLS.values()}))
    agent = c[1].multiselect("Agent", sorted({x.agent_name for x in CALLS.values()}))
    tag = c[2].multiselect("Tag", sorted({t for x in CALLS.values() for t in x.tags}))
    rows = [x for x in CALLS.values() if (not scen or x.scenario in scen) and (not agent or x.agent_name in agent) and (not tag or set(tag) & set(x.tags))]
    st.dataframe(pd.DataFrame([dict(call_id=x.call_id, agent=x.agent_name, scenario=x.scenario, date=x.date, turns=len(x.turns),
                                    duration_s=x.duration_sec, tags=", ".join(x.tags)) for x in rows]), hide_index=True, use_container_width=True)
    if rows:
        cid = st.selectbox("Open call", [x.call_id for x in rows])
        call = CALLS[cid]
        show_red = st.checkbox("Show what an external API would receive (PII redacted)")
        render_transcript(redact_call(call) if show_red else call)
        if instructor:
            with st.expander("Ground-truth audit labels"):
                g = GOLD[cid]
                st.dataframe(pd.DataFrame([dict(qid=q, answer=v["answer"], variant=v["variant"], difficulty=v["difficulty"],
                                                evidence=",".join(map(str, v["evidence_turns"]))) for q, v in g.items()]), hide_index=True)

# ------------------------------------------------------------------ AUDIT SHEET
elif page == PAGES[2]:
    st.header(SHEET.title)
    st.caption(f"Version {SHEET.version} · pass mark {SHEET.pass_mark}% · any failed CRITICAL question = auto-fail")
    df = pd.DataFrame([dict(qid=q.qid, section=q.section, question=q.text, weight=q.weight, critical="⚠️" if q.critical else "",
                            region=q.region, guidance=q.guidance) for q in SHEET.questions])
    st.dataframe(df, hide_index=True, use_container_width=True, height=600)
    st.download_button("Download sheet (CSV)", df.to_csv(index=False), f"{SHEET.sheet_id}.csv")
    with st.expander("Knowledge base the auditor may use (for fact-check questions)"):
        for k in load_kb(): st.markdown(f"**{k['id']}** {k['text']}")

# ------------------------------------------------------------------ RUN AUDIT
elif page == PAGES[3]:
    st.header("Run an audit")
    c = st.columns([1, 2])
    cid = c[0].selectbox("Call", list(CALLS))
    engines = pick_models("Model(s) - choose several to compare", ["baseline-keyword"], key="ra_models")
    opts = options_ui("ra_")
    use_cache = st.checkbox("Use cached results", True)
    if st.button("Run audit", type="primary") and engines:
        st.session_state["runs"] = {}
        for k in engines:
            with st.spinner(f"{LM[k]}..."):
                try:
                    st.session_state["runs"][k] = run_audits(k, [cid], sheet_id, opts, use_cache)[0]
                except Exception as e:
                    st.error(f"{k}: {e}")
        st.session_state["ra_call"] = cid
    runs = st.session_state["runs"]
    if runs and st.session_state.get("ra_call") == cid:
        qm = SHEET.by_qid(); call = CALLS[cid]
        left, right = st.columns([3, 2])
        with left:
            tabs = st.tabs([LM[k] for k in runs])
            sel = {}
            for tab, (k, run) in zip(tabs, runs.items()):
                with tab:
                    sc = rp.score_call(SHEET, run.answers())
                    m = st.columns(4)
                    m[0].metric("Score", f"{sc['total']}%"); m[1].metric("Result", "PASS" if sc["passed"] else "FAIL")
                    m[2].metric("Cost", f"${run.usage.cost_usd:.4f}"); m[3].metric("Latency", f"{run.usage.latency_s:.1f}s")
                    if run.error: st.error(run.error)
                    if run.parse_failures: st.warning(f"{run.parse_failures} JSON parse failure(s)")
                    if run.meta.get("escalated") is not None: st.info(f"Escalated to large model: {run.meta['escalated']}")
                    rows = []
                    for x in run.results:
                        g = GOLD[cid][x.qid]["answer"]
                        rows.append(dict(qid=x.qid, question=qm[x.qid].text, answer=x.answer, conf=x.confidence,
                                         evidence=",".join(map(str, x.evidence_turns)), grounded=x.evidence_valid,
                                         **({"gold": g, "match": "✅" if g == x.answer else "❌"} if instructor else {}), rationale=x.rationale))
                    df = pd.DataFrame(rows)
                    st.dataframe(style_answers(df, ["answer"] + (["gold"] if instructor else [])), hide_index=True, use_container_width=True)
                    sel[k] = st.selectbox("Highlight evidence for", [x.qid for x in run.results], key="hl_" + k)
                    st.text(rp.call_feedback(SHEET, run))
            first = list(runs)[0]
            hl = next((x.evidence_turns for x in runs[first].results if x.qid == sel[first]), [])
        with right:
            st.subheader("Transcript")
            render_transcript(call, hl, GOLD[cid][sel[first]]["evidence_turns"] if instructor else ())
        if len(runs) > 1:
            st.subheader("Side by side")
            cmp = pd.DataFrame({"qid": [x.qid for x in runs[first].results]})
            for k, r in runs.items(): cmp[k] = [x.answer for x in r.results]
            if instructor: cmp["GOLD"] = [GOLD[cid][q]["answer"] for q in cmp.qid]
            st.dataframe(style_answers(cmp, list(cmp.columns[1:])), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ AUDIT REPORT
elif page == PAGES[4]:
    st.header("Overall audit report")
    c = st.columns([2, 1])
    eng = pick_models("Audit with", ["human-gold"], multi=False, key="rep_m", tiers=None)
    n = c[1].slider("Number of calls", 5, len(CALLS), min(30, len(CALLS)))
    opts = options_ui("rep_")
    if st.button("Build report", type="primary"):
        ids = sample_call_ids(n)
        st.session_state["report"] = (eng, run_with_progress(eng, ids, sheet_id, opts), sheet_id)
    if "report" in st.session_state and st.session_state["report"][2] == sheet_id:
        eng, runs, _ = st.session_state["report"]
        rep = rp.batch_report(runs, CALLS, SHEET); k = rep["kpis"]
        m = st.columns(5)
        m[0].metric("Calls audited", k["n"]); m[1].metric("Average score", f"{k['avg']}%"); m[2].metric("Pass rate", f"{k['pass_rate']}%")
        m[3].metric("Calls with critical fail", k["crit_calls"]); m[4].metric("Audit cost", f"${k['cost']}")
        t1, t2, t3, t4 = st.tabs(["Agents", "Questions", "Calls", "Call drill-down"])
        with t1:
            ag = rep["agents"]; st.bar_chart(ag.set_index("agent")["avg_score"])
            st.dataframe(ag, hide_index=True, use_container_width=True)
            secs = [c for c in ag.columns if c in SHEET.sections()]
            heat = ag.set_index("agent")[secs]
            st.caption("Section scores by agent (red = coaching need)")
            sty = heat.style
            f = lambda v: f"background-color: rgb({int(255 - min(v, 100) * 1.0 + 0)},{int(120 + min(v, 100) * 1.3)},130)"
            st.dataframe((sty.map if hasattr(sty, "map") else sty.applymap)(f), use_container_width=True)
        with t2:
            st.bar_chart(rep["questions"].set_index("qid")["fail_rate"])
            st.dataframe(rep["questions"], hide_index=True, use_container_width=True)
        with t3:
            st.dataframe(rep["calls"], hide_index=True, use_container_width=True)
        with t4:
            cid = st.selectbox("Call", [r.call_id for r in runs], key="rep_call")
            run = next(r for r in runs if r.call_id == cid)
            st.text(rp.call_feedback(SHEET, run)); render_transcript(CALLS[cid])
        buf = io.BytesIO(); rp.export_excel(buf, rep, runs, SHEET)
        st.download_button("⬇️ Download Excel report", buf.getvalue(), "audit_report.xlsx")
        st.download_button("⬇️ Download Markdown summary", rp.to_markdown(rep), "audit_report.md")

# ------------------------------------------------------------------ MODEL EVALUATION
elif page == PAGES[5]:
    st.header("Model evaluation on the audit question-answering task")
    st.caption("Positive class = a FAILED audit point (rare and important). Accuracy alone lies - see the lazy model.")
    engines = pick_models("Models to evaluate", ["baseline-lazy", "baseline-keyword", "sim-slm", "sim-llm"], key="ev_models")
    n = st.slider("Number of calls (more = better estimate, more cost)", 5, len(CALLS), 20)
    opts = options_ui("ev_")
    if st.button("Run evaluation", type="primary") and engines:
        ids = sample_call_ids(n); out = {}
        for k in engines:
            out[k] = run_with_progress(k, ids, sheet_id, opts)
        st.session_state["eval"] = (out, opts, sheet_id, ids)
    if "eval" in st.session_state and st.session_state["eval"][2] == sheet_id:
        out, opts_used, _, ids = st.session_state["eval"]
        tbl = ev.compare(out, SHEET)
        show = ["engine", "accuracy", "fail_precision", "fail_recall", "fail_f1", "kappa", "critical_recall", "subtle_acc", "evidence_valid",
                "score_mae", "pass_fail_agreement", "cost_per_1k_calls", "sec_per_call", "p95_latency_s", "json_failures"]
        st.dataframe(tbl[show].round(3), hide_index=True, use_container_width=True)
        st.download_button("Download metrics CSV", tbl.to_csv(index=False), "evaluation.csv")
        c = st.columns(2)
        c[0].subheader("Quality vs cost"); c[0].scatter_chart(tbl, x="cost_per_1k_calls", y="fail_f1", color="engine")
        c[1].subheader("Failure recall (incl. critical)"); c[1].bar_chart(tbl.set_index("engine")[["fail_recall", "critical_recall"]])
        pick = st.selectbox("Drill into model", list(out))
        df = ev.answers_df(out[pick], SHEET)
        t1, t2, t3, t4, t5, t6 = st.tabs(["Confusion", "By section", "By difficulty / variant", "Calibration", "Error explorer", "Consistency & judge"])
        with t1: st.dataframe(ev.confusion(df))
        with t2: st.dataframe(ev.by_group(df, "section").round(3), hide_index=True)
        with t3:
            st.dataframe(ev.by_group(df, "difficulty").round(3), hide_index=True)
            v = ev.by_group(df, "variant"); st.dataframe(v[v.n > 0].sort_values("accuracy").round(3), hide_index=True)
        with t4:
            cal, ece = ev.calibration(df)
            st.write(f"Expected calibration error: **{ece:.3f}** (0 = confidence matches accuracy)")
            st.dataframe(cal, hide_index=True)
        with t5:
            bad = df[df.pred != df.gold]
            st.write(f"{len(bad)} wrong answers")
            st.dataframe(bad[["call_id", "qid", "variant", "gold", "pred", "conf", "rationale"]], hide_index=True, use_container_width=True)
            if len(bad):
                r = bad.iloc[st.number_input("Show error #", 0, len(bad) - 1, 0)]
                render_transcript(CALLS[r.call_id], gold_turns=GOLD[r.call_id][r.qid]["evidence_turns"])
        with t6:
            if st.button("Re-run with reworded prompt (v1b) and compare"):
                alt = run_with_progress(pick, ids, sheet_id, {**opts_used, "prompt_variant": "v1b" if opts_used["prompt_variant"] == "v1" else "v1"})
                st.metric("Answer agreement between prompt variants", f"{100 * ev.consistency(out[pick], alt):.1f}%")
            jm = [m["key"] for m in ready_models(["llm"]) if not m["key"].startswith("sim")]
            if jm:
                judge = st.selectbox("Judge model", jm)
                if st.button("Rate rationales of 10 wrong/NO answers (LLM-as-judge)"):
                    from smartaudit.judge import rate_rationale
                    cl = get_client(judge); sc = []
                    for _, r in df[df.pred == "NO"].head(10).iterrows():
                        sc.append(rate_rationale(cl, r.call_id, SHEET.by_qid()[r.qid].text, r.pred, r.rationale).get("faithfulness"))
                    st.write("Faithfulness scores:", sc)
            else:
                st.caption("LLM-as-judge needs a ready Claude/OpenAI model.")

# ------------------------------------------------------------------ SLM vs LLM
elif page == PAGES[6]:
    st.header("When to use an SLM, when an LLM?")
    t1, t2, t3, t4 = st.tabs(["Decision advisor", "Cost break-even", "Cascade simulator", "SLM toolbox"])
    with t1:
        preset = st.selectbox("Start from an audit task", list(adv.PRESETS))
        f = {}
        cols = st.columns(2)
        for i, (k, (lab, _)) in enumerate(adv.FACTORS.items()):
            f[k] = cols[i % 2].slider(lab, 1, 5, adv.PRESETS[preset][k], key=f"{preset}{k}")
        rec = adv.recommend(f)
        st.success(f"**Recommendation: {rec.verdict}**  (score {rec.score}: positive = SLM, negative = LLM)")
        for r in rec.reasons: st.write("•", r)
        st.caption("Weights live in smartaudit/advisor.py - change them and see if you agree with the result (Assignment Part 5).")
    with t2:
        c = st.columns(3)
        avg_chars = sum(len(x.as_text()) for x in CALLS.values()) / len(CALLS)
        tin = c[0].number_input("Tokens in per audit", 100, 100000, int(avg_chars / 4) + 900, help="≈ transcript tokens + prompt")
        tout = c[1].number_input("Tokens out per audit", 10, 10000, 900)
        gpu = c[2].number_input("GPU $/hour (self-host SLM)", 0.0, 20.0, 1.0)
        c = st.columns(2)
        pin = c[0].number_input("LLM price in ($/1M)", 0.0, 100.0, 2.0); pout = c[1].number_input("LLM price out ($/1M)", 0.0, 100.0, 10.0)
        vols = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
        d = pd.DataFrame([dict(calls_per_month=v, **adv.monthly_cost(v, tin, tout, pin, pout, gpu)) for v in vols]).set_index("calls_per_month")
        st.line_chart(d); st.dataframe(d.round(0))
        be = adv.break_even_calls(tin, tout, pin, pout, gpu)
        st.metric("Break-even volume (calls / month)", f"{be:,.0f}")
        st.caption("Self-hosting wins only above this volume - and only if one GPU can serve it and you can run it (people cost not included).")
    with t3:
        st.write("Run a small and a large model once (cached), then replay the cascade at every threshold - no extra API cost.")
        c = st.columns(3)
        small = c[0].selectbox("Small model", [m["key"] for m in ready_models(["slm", "baseline"])] or ["sim-slm"])
        large = c[1].selectbox("Large model", [m["key"] for m in ready_models(["llm"])] or ["sim-llm"])
        n = c[2].slider("Calls", 5, len(CALLS), 20, key="cas_n")
        crit = st.checkbox("Always escalate a critical NO")
        if st.button("Simulate cascade"):
            ids = sample_call_ids(n); o = dict(DEFAULT_OPTIONS)
            rs = run_with_progress(small, ids, sheet_id, o, label="small"); rl = run_with_progress(large, ids, sheet_id, o, label="large")
            st.session_state["cas"] = ev.simulate_cascade(rs, rl, SHEET, escalate_critical_no=crit)
        if "cas" in st.session_state:
            d = st.session_state["cas"]
            st.line_chart(d.set_index("threshold")[["fail_f1", "critical_recall"]]); st.line_chart(d.set_index("threshold")[["cost_per_call", "pct_escalated"]])
            st.dataframe(d.round(3), hide_index=True)
            st.caption("Pick the threshold where quality is 'good enough' and cost is acceptable. The lowest threshold = small model only; 1.01 = large model for everything.")
    with t4:
        st.subheader("What makes small models work")
        for k, v in slm_tools.SLM_RECIPE.items(): st.markdown(f"**{k}** — {v}")
        cid = st.selectbox("Call", list(CALLS), key="tb_call"); qid = st.selectbox("Question", [q.qid for q in SHEET.questions], index=len(SHEET.questions) - 1)
        d = slm_tools.prompt_size_demo(CALLS[cid], SHEET.by_qid()[qid])
        c = st.columns(2); c[0].metric("Full prompt (tokens ≈)", d["full_tokens"]); c[1].metric("Windowed + compact", d["windowed_tokens"])
        with st.expander("Windowed prompt sent to the SLM"): st.code(d["windowed_prompt"])
        st.code("# Kaggle: pick GPU T4 x2, then add a model to models.json or use hf-qwen2.5-3b\nrun_audits('hf-qwen2.5-3b', ids, options=slm_tools.SLM_OPTIONS)")

# ------------------------------------------------------------------ ASK AI
elif page == PAGES[7]:
    st.header("Ask AI about a call")
    gen = [m["key"] for m in ready_models(["llm", "slm"]) if not m["key"].startswith("sim")]
    if not gen:
        st.warning("Needs a ready generative model (Claude / OpenAI key, or a local SLM).")
    else:
        cid = st.selectbox("Call", list(CALLS)); model = st.selectbox("Model", gen); red = st.checkbox("Redact PII")
        q = st.text_input("Question", "Did the agent handle the customer's worry well? Quote the turn.")
        c1, c2 = st.columns([2, 3])
        with c2: render_transcript(CALLS[cid])
        with c1:
            if st.button("Ask", type="primary"):
                from smartaudit.ask import ask_call
                st.session_state["ask"] = ask_call(get_client(model), cid, q, red)
            a = st.session_state.get("ask")
            if a:
                st.write(a.get("answer")); st.caption(f"cited turns: {a.get('cited_turns')} · ${a['cost']:.5f} · {a['latency']:.1f}s")
