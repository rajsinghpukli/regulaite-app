# rag/pipeline.py
# ------------------------------------------------------------
# RegulAIte — Hybrid pipeline (Vectors + optional Web via OpenAI tool)
# - Intent Router (regulatory | research | quick_fact | mixed_compare)
# - Hybrid retrieval: OpenAI Vector Store via file_search + OpenAI web_search
# - Natural consultant-style answers (paragraphs + bullets; mini-table when helpful)
# - Depth: concise | standard | deep
# - Guardrails: non-answer detector + one controlled retry (no forced web)
# - Multi-agent early path (Analyst->Retriever->Drafter->Judge->Reviser) when MULTIAGENT=1
# - Follow-ups: guaranteed (we synthesize if the model forgets) for non-concise depths
# - ALWAYS finish with a Citations section (auto-added for web sources)
# ------------------------------------------------------------

import os
import re
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from openai import OpenAI

from .router import classify_intent_and_scope
from .websearch import web_search   # local adapter; returns [] if not configured

# --- Multi-agent (optional) ---
MULTIAGENT = os.getenv("MULTIAGENT", "0") == "1"
ALLOW_MA_WEB = os.getenv("MULTIAGENT_ALLOW_WEB", "0") == "1"
try:
    from .agents import run_multi_agent_rag
except Exception:
    run_multi_agent_rag = None  # fall back to legacy path if import fails

# -------------------- Tunables (env) --------------------
RESPONSES_MODEL   = os.getenv("RESPONSES_MODEL", "gpt-4o-mini")
FIRST_TOKENS      = int(os.getenv("RESP_FIRST_TOKENS", "3200"))   # higher to reduce truncation
TIMEOUT_S         = int(os.getenv("RESP_FIRST_TIMEOUT", "60"))
TEMPERATURE_MAIN  = float(os.getenv("RESP_TEMP_MAIN", "0.18"))

# Vector store env (comma-separated supported)
_OPENAI_VS = os.getenv("OPENAI_VECTOR_STORE_ID", "")
OPENAI_VECTOR_STORE_IDS: List[str] = [v.strip() for v in _OPENAI_VS.split(",") if v.strip()]
USE_OPENAI_VECTOR = len(OPENAI_VECTOR_STORE_IDS) > 0

# Label for UI only
PERSIST_DIR = "(OpenAI Vector Store)"
TOP_K_DEFAULT = 8

# Optional: router mode hint via UI (None = auto)
DEFAULT_MODE_HINT = None  # "regulatory" | "research" | "quick_fact" | "mixed_compare"

# Evidence quotes section toggle (read live each call, but keep a cached default)
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "0") == "1"

# -------------------- OpenAI helpers --------------------
def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

def _attempts(vec_ids: List[str], enable_openai_web: bool) -> List[Dict[str, Any]]:
    """
    Try multiple tool argument shapes so it works across different server/SDKs.
    """
    fs_inline = {"type": "file_search", "vector_store_ids": vec_ids}
    fs_basic  = {"type": "file_search"}
    web_tool  = {"type": "web_search"}

    attempts: List[Dict[str, Any]] = []

    if enable_openai_web:
        attempts += [
            {"tools": [fs_inline, web_tool], "tool_choice": "auto"},
            {"tools": [fs_inline, web_tool],
             "tool_resources": {"file_search": {"vector_store_ids": vec_ids}},
             "tool_choice": "auto"},
            {"tools": [fs_basic, web_tool],
             "tool_resources": {"file_search": {"vector_store_ids": vec_ids}},
             "tool_choice": "auto"},
            {"tools": [fs_basic, web_tool],
             "store": {"file_search": {"vector_store_ids": vec_ids}},
             "tool_choice": "auto"},
        ]

    attempts += [
        {"tools": [fs_inline], "tool_choice": "auto"},
        {"tools": [fs_inline],
         "tool_resources": {"file_search": {"vector_store_ids": vec_ids}},
         "tool_choice": "auto"},
        {"tools": [fs_basic],
         "tool_resources": {"file_search": {"vector_store_ids": vec_ids}},
         "tool_choice": "auto"},
        {"tools": [fs_basic], "store": {"file_search": {"vector_store_ids": vec_ids}}, "tool_choice": "auto"},
        {"tools": [fs_basic]},
    ]
    return attempts

# -------------------- Rendering for multi-agent JSON --------------------
def _render_regulaite_json_to_md(ans: Dict[str, Any]) -> str:
    """
    Natural rendering:
    - Merge narrative parts into one flowing '## Answer' section.
    - Show '## Evidence (verbatim quotes)' ONLY if EVIDENCE_MODE=1.
    - Always finish with '## Citations' when available.
    """
    lines = []

    # Merge narrative content (summary first, then tails if present)
    body_parts = []
    for key in ["summary", "general_knowledge", "comparative_analysis", "recommendation", "gaps_or_next_steps"]:
        t = (ans.get(key) or "").strip()
        if t:
            body_parts.append(t)

    if body_parts:
        lines += ["## Answer", "\n\n".join(body_parts), ""]

    # Evidence — only when toggled ON
    if os.getenv("EVIDENCE_MODE", "0") == "1":
        per_src = ans.get("per_source") or {}
        if any(per_src.get(k) for k in per_src):
            lines += ["## Evidence (verbatim quotes)"]
            for fw, quotes in per_src.items():
                if not quotes:
                    continue
                lines.append(f"**{fw}**")
                for q in quotes[:6]:
                    q = (q or "").strip()
                    if not q:
                        continue
                    if len(q) > 320:
                        q = q[:317] + "…"
                    lines.append(f"- “{q}”")
                lines.append("")

    # Citations — always if provided
    cits = ans.get("citations") or []
    if cits:
        lines += ["## Citations"]
        for c in cits:
            fw  = (c.get("framework") or "").strip()
            sid = (c.get("source_id") or "").strip()
            pg  = (c.get("page") or "").strip()
            tail = f" (p. {pg})" if pg else ""
            label = f"{fw}: " if fw else ""
            lines.append(f"- {label}{sid}{tail}")

    return "\n".join(lines).strip() or "_No content._"

# -------------------- LLM call --------------------
def _ask_raw(client: OpenAI, messages: List[Dict[str, str]], vec_ids: List[str],
             temperature: float, max_output_tokens: int, timeout_s: int,
             enable_openai_web: bool):
    base = dict(
        model=RESPONSES_MODEL,
        input=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout_s,
    )
    last_err = None
    for kwargs in _attempts(vec_ids, enable_openai_web=enable_openai_web):
        try:
            return client.responses.create(**base, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Responses.create failed.")

# -------------------- Guards & helpers --------------------
_MARKET_TRIGGERS = [
    "revenue", "market", "sector", "npl", "ratio", "growth", "latest",
    "as of", "industry", "market size", "statistics", "press release",
    "roae", "roaa", "return on average equity", "return on average assets",
    "performance indicator", "kpi", "financial stability report", "annual report",
    "2023", "2024", "2025"
]

def _should_force_web(query: str) -> bool:
    ql = query.lower()
    return any(t in ql for t in _MARKET_TRIGGERS)

def _looks_like_non_answer(md: str) -> bool:
    """
    Heuristic: treat as 'non-answer' only if clearly low-info or too short.
    Do NOT trigger solely because '## Citations' is missing (avoids unnecessary web).
    """
    red_flags = [
        "do not contain explicit figures",
        "not published in these regulatory texts",
        "cannot provide the total revenue",
        "aggregate sector-wide revenue figures are not published",
        "do not state the total revenue",
        "the materials focus primarily on regulatory",
        "these disclosures are mandated",
        "must be publicly available",
        "consult the individual banks' annual reports",
        "not provided in the regulatory documents",
    ]
    low_info = sum(1 for t in red_flags if t.lower() in md.lower())
    too_short = len(md.strip()) < 420   # characters, not words
    return (low_info >= 1) or too_short

# ---- Follow-up synthesis (for chips) ----
def _has_followups(md: str) -> bool:
    return bool(re.search(r'^\s*-\s*\[Ask\]\s+', md, flags=re.MULTILINE))

def _synth_followups(query: str, intent: str) -> List[str]:
    ql = (query or "").lower()
    ideas: List[str] = []

    if "npl" in ql or "financial stability report" in ql or "macro" in ql:
        ideas += [
            "What is the latest NPL ratio by segment (retail/SME/corporate) as of the newest FSR?",
            "Which sectors drive Stage 2 migrations and how should overlays reflect them?",
            "If baseline macro worsens by 1 notch, how should PD/LGD/FV overlays change?",
        ]
    if "training" in ql:
        ideas += [
            "Show a Branch vs Risk training table mapped to CBB clauses.",
            "What evidence pack should Internal Audit request to test training effectiveness?",
        ]
    if "ecl" in ql or "sukuk" in ql:
        ideas += [
            "Show a side-by-side IFRS 9 vs AAOIFI table (SICR 30 DPD, default 90 DPD, interest on net).",
            "Give journal entries for Stage 3 interest on net and for a cure back to Stage 2.",
        ]
    if "liquidity" in ql:
        ideas += [
            "Build a gap table: CBB disclosure requirement vs what we publish vs remediation owner.",
        ]
    if not ideas:
        ideas = [
            "Summarize the top 3 risks and controls we should enhance next quarter.",
            "Which two data points would most improve model calibration and why?",
        ]

    # Deduplicate and keep it tight
    seen, out = set(), []
    for i in ideas:
        s = i.strip()
        if s and s not in seen:
            out.append(s if len(s) <= 120 else s[:117] + "…")
            seen.add(s)
        if len(out) >= 6: break
    return out

def _append_followups(md: str, follows: List[str]) -> str:
    if not follows:
        return md
    block = "## Follow-ups\n" + "\n".join(f"- [Ask] {q}" for q in follows) + "\n"
    if "## Citations" in md:
        head, tail = md.split("## Citations", 1)
        return head.rstrip() + "\n\n" + block + "\n## Citations" + tail
    return md.rstrip() + "\n\n" + block

# ---- Length helpers ----
def _word_count(md: str) -> int:
    # rough but reliable
    return len(re.findall(r"[A-Za-z0-9\-]+", md or ""))

def _is_big_topic(query: str) -> bool:
    ql = (query or "").lower()
    # topics that deserve long answers by default
    return any(t in ql for t in [
        "compare", "vs", "versus", "difference",
        "ecl", "sukuk", "npl", "financial stability report",
        "training", "plan", "liquidity", "disclosure", "kpi", "benchmark"
    ])

def _needs_length_expansion(query: str, md: str, depth: str) -> bool:
    if depth == "concise":
        return False
    wc = _word_count(md)
    # if it's a big topic and answer is under ~550 words, ask model once to expand
    return _is_big_topic(query) and wc < 550

# -------------------- Prompt builders --------------------
def _build_system_prompt(
    intent: str,
    frameworks: List[str],
    depth: str,
    include_web: bool,
    evidence_mode: bool | None = None,
) -> str:
    if evidence_mode is None:
        evidence_mode = os.getenv("EVIDENCE_MODE", "0") == "1"

    fw_hint = f"Focus frameworks: {', '.join(frameworks)}." if frameworks else "Focus frameworks: (auto)."
    web_hint = (
        "Web sources allowed; include dates and direct links. Cite web evidence inline as [W1], [W2] with short title + URL."
        if include_web else
        "Do NOT use web sources. Never fabricate links."
    )

    depth_hint = (
        "Write a short, flowing answer (3–6 sentences)."
        if depth == "concise" else
        "Write a thorough, flowing answer (≈350–700 words)."
        if depth == "standard" else
        "Write an in-depth, flowing answer (≈900–1500 words) with concrete examples."
    )

    evidence_rule = (
        "EvidenceMode=ON → include a separate '## Evidence (verbatim quotes)' with 2–5 short quotes per addressed framework (≤220 chars). "
        "EvidenceMode=OFF → do NOT create an Evidence section; at most 0–2 inline quotes if they help readability."
    )

    style = (
        "Tone: senior compliance consultant. Paragraphs first, then bullets where they improve clarity; avoid numbered lists (1., 2., 3.). "
        "You MAY use helpful sub-headings (e.g., '### Macro context', '### Policy thresholds & SICR overlays', '### Implementation notes'). "
        "Use a compact Markdown table (≤6 rows) when it clearly helps (comparisons, KPI snapshots, calendars)."
    )

    followups_rule = (
        "If depth != concise, finish with 3–6 follow-up lines exactly like: '- [Ask] <short question>'."
    )

    citations = (
        "Always finish with '## Citations' listing every source used. "
        "Vector: '- IFRS: IFRS_9.pdf#p12' (add page/para if known). "
        "Web: '- [Title](URL) — as of <date>'."
    )

    role = (
        "You are a regulatory assistant for Khaleeji Bank. Use vector documents via file_search as primary authority; "
        "blend trusted web sources only when allowed."
    )
    flags = f"Intent: {intent}. EvidenceMode: {'ON' if evidence_mode else 'OFF'}."

    return "\n".join([role, flags, fw_hint, web_hint, depth_hint, evidence_rule, style, followups_rule, citations])

def _build_user_prompt(query: str, web_items: List[Dict[str, Any]], frameworks: List[str]) -> str:
    # Base web snippet block
    if not web_items:
        base = f"Question: {query.strip()}\n\nNo web snippets provided."
    else:
        lines = [f"Question: {query.strip()}", "", "Web snippets (label and cite as [W1], [W2], ...):"]
        for i, item in enumerate(web_items, 1):
            t = item.get("title", "").strip()
            u = item.get("url", "").strip()
            d = item.get("date", "").strip()
            s = item.get("snippet", "").strip()
            lines.append(f"[W{i}] title={t} | url={u} | date={d}\n{(s[:1000])}")
        base = "\n".join(lines)

    if frameworks:
        base += "\n\nFrameworks to prioritize: " + ", ".join(frameworks)

    # ---- Universal + Topic-aware checklists (nudges structure without rigid templates) ----
    ql = query.lower()
    checklists: List[str] = []

    # Universal nudge (applies to most non-trivial queries)
    checklists.append(
        "WriterChecklist:\n"
        "- Prefer flowing paragraphs; add short bullet lists where they improve readability.\n"
        "- If the prompt implies comparison ('vs', 'compare', 'difference', 'table', 'benchmark'), include ONE compact table (4–8 rows).\n"
        "- If the prompt asks for a plan, policy, calendar, or KPIs, include ONE compact table listing owner/frequency/metric/thresholds (≤7 rows).\n"
        "- End with 3–6 follow-up lines '- [Ask] ...' unless the answer is very short."
    )

    # ECL staging / sukuk — long, sectioned, explicit points
    if (("ecl" in ql) and ("stage" in ql or "staging" in ql)) or (("sukuk" in ql) and ("ecl" in ql)):
        checklists.append(
            "TopicChecklist-ECL:\n"
            "- Write a consultant-style analysis (≈900–1400 words). Avoid numbered lists.\n"
            "- Include a compact table comparing IFRS 9 vs AAOIFI: stages; SICR backstop ≥30 DPD; default ≥90 DPD + 'unlikely to pay'; "
            "Stage 3 interest 'on net'; forward-looking info; sukuk-specific notes.\n"
            "- Use short sub-headings where helpful (e.g., 'IFRS 9 mechanics', 'AAOIFI FAS specifics', 'Practical implications for sukuk').\n"
            "- Explicitly explain interest on net (EIR on amortised cost net of allowance) and note differences in recognition/cease rules.\n"
            "- Add 3–6 'Implementation notes' bullets (data signals, overlays, staging triggers, documentation)."
        )

    # NPL / macro linkage (CBB/FSR)
    if ("npl" in ql or "non-performing" in ql) and ("bahrain" in ql or "cbb" in ql or "financial stability report" in ql or "latest" in ql):
        checklists.append(
            "TopicChecklist-NPL:\n"
            "- Use sub-headings: '### Macro context', '### Policy thresholds & SICR overlays', '### Implementation notes'.\n"
            "- Include ONE compact table titled 'Macro signal → Policy reaction' (5–7 rows) and label any web-cited number with 'as of <date>'."
        )

    # Compliance training plan
    if ("training" in ql and "compliance" in ql) or ("branch staff" in ql and "risk staff" in ql):
        checklists.append(
            "TopicChecklist-Training:\n"
            "- Provide a table comparing Branch vs Risk staff (frequency, modality, pass mark, record-keeping, attestations, owner).\n"
            "- Map each element back to relevant CBB clauses (use short clause refs in the table cells)."
        )

    # Liquidity risk disclosures / gaps
    if ("liquidity" in ql and "disclosure" in ql) or ("ldr" in ql) or ("nsfr" in ql) or ("lsl" in ql):
        checklists.append(
            "TopicChecklist-Liquidity:\n"
            "- Include a small table: Required disclosure (CBB) → What banks publish → Gap → Fix."
        )

    # KPI / benchmarking
    if ("kpi" in ql or "benchmark" in ql or "peer" in ql):
        checklists.append(
            "TopicChecklist-KPI:\n"
            "- Include a KPI snapshot table (metric, definition, source, current value 'as of', target/threshold)."
        )

    if checklists:
        base += "\n\n" + "\n\n".join(checklists)

    return base

# -------------------- Public API (drop-in) --------------------
def index_files(paths):  # no-op in vector mode
    return 0, 0

def wipe_index():        # no-op
    return None

# Add MULTIAGENT flag to cache key to avoid reusing hybrid results
@lru_cache(maxsize=64)
def _cached_answer(query: str, k: int, model: str, vec_key: str,
                   mode_hint: str, include_web: bool, depth: str,
                   ma_flag: str) -> Tuple[str, str, str]:
    client = _openai_client()
    if not client:
        return "", "OPENAI_API_KEY not set.", "vector_error"
    vec_ids = OPENAI_VECTOR_STORE_IDS
    if not vec_ids:
        return "", "OPENAI_VECTOR_STORE_ID not set.", "vector_error"

    # ---- Multi-agent early path ----
    if (ma_flag == "1") and run_multi_agent_rag:
        try:
            # Agents can use web only if allowed by env AND router signals market need
            if _should_force_web(query) and ALLOW_MA_WEB:
                ma = run_multi_agent_rag(user_query=query)
            elif not _should_force_web(query):
                ma = run_multi_agent_rag(user_query=query)
            else:
                ma = None
            if ma:
                md = _render_regulaite_json_to_md(ma.get("answer") or {})
                # Ensure '## Citations' presence for UI consistency
                if "## Citations" not in md:
                    cits = ma.get("answer", {}).get("citations") or []
                    if cits:
                        md += "\n\n## Citations\n" + "\n".join(
                            f"- {c.get('framework','')+': ' if c.get('framework') else ''}{c.get('source_id','')}{(' (p. '+c.get('page','')+')') if c.get('page') else ''}"
                            for c in cits
                        )
                # Guarantee follow-ups for non-concise depths
                if depth != "concise" and not _has_followups(md):
                    md = _append_followups(md, _synth_followups(query, ma.get("plan", {}).get("intent", "")))
                return md, "", "multiagent"
        except Exception:
            # fail-safe: fall back to legacy hybrid path
            pass

    # ---- Router (intent + scope) ----
    route = classify_intent_and_scope(query, mode_hint=mode_hint)
    intent = route["intent"]
    frameworks = route.get("frameworks", [])
    needs_web = route.get("needs_web", False)

    force_web = _should_force_web(query)
    use_web = bool(include_web and (needs_web or force_web))
    enable_openai_web = bool(use_web)

    web_items: List[Dict[str, Any]] = []
    if use_web:
        web_k = 6 if intent in ("research", "mixed_compare") or force_web else 3
        web_items = web_search(query, k=web_k)

    # read the live toggle for evidence mode, and pass it
    evidence_mode_flag = os.getenv("EVIDENCE_MODE", "0") == "1"

    system_prompt = _build_system_prompt(
        intent=intent,
        frameworks=frameworks,
        depth=depth,
        include_web=use_web,
        evidence_mode=evidence_mode_flag,
    )
    user_prompt = _build_user_prompt(query, web_items=web_items, frameworks=frameworks)

    # Depth-aware token budget (gentle bump for deep)
    max_tokens = FIRST_TOKENS + (800 if depth == "deep" else 0)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ---- FIRST CALL ----
    try:
        resp = _ask_raw(
            client, messages, vec_ids,
            temperature=TEMPERATURE_MAIN,
            max_output_tokens=max_tokens,
            timeout_s=TIMEOUT_S,
            enable_openai_web=enable_openai_web
        )
    except Exception as e:
        if use_web and web_items:
            lines = [f"**Question:** {query}", "", "**Sources provided:**"]
            for i, it in enumerate(web_items, 1):
                lines.append(f"- [W{i}] [{it.get('title','link')}]({it.get('url','')}) (as of {it.get('date','n/a')})")
            return "\n".join(lines), f"Vector retrieval failed: {e}", "hybrid_partial"
        return "", f"Vector call failed: {e}", "vector_error"

    # Output text (markdown)
    md = (getattr(resp, "output_text", "") or "").strip()

    # ---- Guard: Non-answer detector + one controlled retry (no forced web) ----
    if _looks_like_non_answer(md) and include_web:
        if not web_items:
            enriched_query = (
                f"{query} site:cbb.gov.bh OR (Bahrain banking sector) "
                f'("Financial Stability Report" OR "Sector Review" OR "Banking Sector Performance")'
            )
            web_items = web_search(enriched_query, k=8) or web_items
            if not web_items:
                web_items = web_search("Bahrain banking ROAE ROAA 2023 annual report PDF", k=8) or []

        if web_items:
            system_prompt = _build_system_prompt(
                intent=intent,
                frameworks=frameworks,
                depth=depth,
                include_web=enable_openai_web,  # keep decision; don't force web
                evidence_mode=evidence_mode_flag,
            )
            user_prompt = _build_user_prompt(query, web_items=web_items, frameworks=frameworks)
            try:
                resp_retry = _ask_raw(
                    client,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
                    vec_ids,
                    temperature=TEMPERATURE_MAIN,
                    max_output_tokens=max_tokens,
                    timeout_s=TIMEOUT_S,
                    enable_openai_web=enable_openai_web
                )
                md_retry = (getattr(resp_retry, "output_text", "") or "").strip()
                if md_retry and len(md_retry) >= len(md):
                    md = md_retry
            except Exception:
                pass

        if _looks_like_non_answer(md) and web_items:
            lines = [f"**Question:** {query}", "", "**Sources:**"]
            for i, it in enumerate(web_items, 1):
                lines.append(f"- [W{i}] [{it.get('title','link')}]({it.get('url','')}) (as of {it.get('date','n/a')})")
            md = "\n".join(lines) + ("\n\n" + md if md else "")

    # ---- Length nudge for big topics (single expansion attempt) ----
    if _needs_length_expansion(query, md, depth):
        expansion_hint = (
            "LengthNudge:\n"
            "- Expand to ≈900–1400 words with consultant-style clarity.\n"
            "- Use short sub-headings and at least one compact table if the prompt implies comparison/plan/KPIs.\n"
            "- Keep the '## Citations' section. Do not add fluff; deepen explanations and practical implications."
        )
        # Tailor an extra hint for ECL/sukuk comparisons
        if ("ecl" in query.lower()) or ("sukuk" in query.lower()):
            expansion_hint += (
                "\n- For ECL/sukuk: show IFRS 9 mechanics vs AAOIFI specifics, SICR ≥30 DPD rebuttable presumption, default ≥90 DPD + 'unlikely to pay', "
                "Stage 3 interest on net, forward-looking info, sukuk-specific notes, and 3–6 implementation bullets."
            )
        user_prompt_expanded = _build_user_prompt(query, web_items=web_items, frameworks=frameworks) + "\n\n" + expansion_hint
        try:
            resp_expand = _ask_raw(
                client,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt_expanded}],
                vec_ids,
                temperature=TEMPERATURE_MAIN,
                max_output_tokens=max_tokens,
                timeout_s=TIMEOUT_S,
                enable_openai_web=enable_openai_web
            )
            md_expand = (getattr(resp_expand, "output_text", "") or "").strip()
            if md_expand and len(md_expand) > len(md) + 400:
                md = md_expand
        except Exception:
            pass

    # Ensure we always append a Citations section if model forgot AND we have web_items
    if "## Citations" not in md and web_items:
        md += "\n\n## Citations\n" + "\n".join(
            f"- [{it.get('title','link')}]({it.get('url','')}) — as of {it.get('date','n/a')}"
            for it in web_items
        )

    # Guarantee follow-up chips unless concise
    if depth != "concise" and not _has_followups(md):
        follows = _synth_followups(query, intent)
        md = _append_followups(md, follows)

    if not md:
        if use_web and web_items:
            lines = [f"**Question:** {query}", "", "**Sources provided:**"]
            for i, it in enumerate(web_items, 1):
                lines.append(f"- [W{i}] [{it.get('title','link')}]({it.get('url','')}) (as of {it.get('date','n/a')})")
            return "\n".join(lines), "", "hybrid_partial"
        return "_No answer returned._", "", "hybrid_auto"

    mode = "hybrid_auto" if mode_hint in (None, "", "auto") else f"hybrid_{mode_hint}"
    return md, "", mode

def ask(query: str, k: int = TOP_K_DEFAULT,
        mode_hint: str = DEFAULT_MODE_HINT, include_web: bool = True, depth: str = "standard") -> Dict[str, Any]:
    """
    mode_hint: None/'auto' | 'regulatory' | 'research' | 'quick' | 'mixed'
    include_web: if True, router can use web for research/mixed or when forced by triggers
    depth: 'concise' | 'standard' | 'deep'
    """
    mh = (mode_hint or "").strip().lower()
    if mh in ("", "auto"): mh = None
    dp = depth if depth in ("concise","standard","deep") else "standard"

    md, err, mode = _cached_answer(
        query.strip(), k, RESPONSES_MODEL, ",".join(OPENAI_VECTOR_STORE_IDS),
        mh or "auto", bool(include_web), dp,
        "1" if MULTIAGENT else "0"       # <-- include MA flag in cache key
    )
    if err and mode == "vector_error":
        return {"answer_markdown": "", "answer": err, "mode": "vector_error"}
    return {"answer_markdown": md, "mode": mode}
