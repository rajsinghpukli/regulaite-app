# rag/agents.py
# Multi-agent RAG over OpenAI Vector Store (file_search), with optional web fallback.
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os, json, re
from openai import OpenAI

# ---------------- Env / client ----------------
RESPONSES_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4o-mini")
TOP_K           = int(os.getenv("TOP_K", "12"))
DEBATE          = os.getenv("MULTIAGENT_DEBATE", "0") == "1"
AGENT_MAX_TOKENS= int(os.getenv("AGENT_MAX_TOKENS", "2600"))  # a bit higher
ALLOW_WEB       = os.getenv("MULTIAGENT_ALLOW_WEB", "0") == "1"  # << enable web fallback

_OPENAI_VS = os.getenv("OPENAI_VECTOR_STORE_ID", "")
OPENAI_VECTOR_STORE_IDS: List[str] = [v.strip() for v in _OPENAI_VS.split(",") if v.strip()]

def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=key)

# ---------------- Heuristics ----------------
FRAMEWORK_KEYS = {
    "IFRS": ["IFRS", "IFRS 9", "IAS ", "International Financial Reporting", "IFRS®"],
    "AAOIFI": ["AAOIFI", "FAS 30", "Accounting and Auditing Organization for Islamic"],
    "CBB": ["CBB", "Central Bank of Bahrain", "Volume 2", "CM-"],
    "InternalPolicy": ["Policy", "Accounting Policy", "Manual", "Internal"],
}
def _source_matches_framework(fw: str, title: str, source_id: str) -> bool:
    hay = f"{title} {source_id}".lower()
    return any(k.lower() in hay for k in FRAMEWORK_KEYS.get(fw, []))

def _canon_fw(name: str) -> str:
    n = (name or "").strip().lower()
    if "aaoifi" in n or "fas 30" in n: return "AAOIFI"
    if "ifrs" in n: return "IFRS"
    if "cbb" in n or "central bank of bahrain" in n: return "CBB"
    if "policy" in n or "manual" in n or "internal" in n: return "InternalPolicy"
    return name or "Other"

# ---------------- JSON extraction ----------------
import json as _json
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)
_SLASH_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
def _extract_json_text(text: str) -> str:
    if text is None: raise ValueError("Empty model output.")
    s = text.strip()
    s = _CODE_FENCE_RE.sub("", s)
    s = _SLASH_COMMENT_RE.sub("", s)
    s = _BLOCK_COMMENT_RE.sub("", s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    try:
        _json.loads(s); return s
    except Exception:
        pass
    first_obj = s.find("{"); first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if starts:
        start = min(starts); stack=[]; end=None
        for i,ch in enumerate(s[start:], start):
            if ch in "{[": stack.append(ch)
            elif ch in "}]":
                if not stack: continue
                top=stack.pop()
                if (top=="{" and ch!="}") or (top=="[" and ch!="]"): continue
                if not stack: end=i; break
        if end is not None and end>start:
            cand = _TRAILING_COMMA_RE.sub(r"\1", s[start:end+1])
            try: _json.loads(cand); return cand
            except Exception: pass
    s2 = s.replace("“","\"").replace("”","\"")
    _json.loads(s2); return s2

# ---------------- Data contracts ----------------
class QueryPlan(BaseModel):
    intent: str
    frameworks: List[str]
    subqueries: List[str]
    key_terms: List[str] = []

class EvidenceItem(BaseModel):
    framework: str
    quote: str
    source_title: str
    source_id: str
    page: Optional[str] = None

class EvidenceBundle(BaseModel):
    per_framework: Dict[str, List[EvidenceItem]]

class RegulAIteAnswer(BaseModel):
    summary: str
    per_source: Dict[str, List[str]]
    comparative_analysis: str
    recommendation: str
    general_knowledge: str
    gaps_or_next_steps: str
    citations: List[Dict[str, str]]
    # extra (not required by renderer, but used): followups: List[str] = []

class ReviewCard(BaseModel):
    grounded: int = Field(ge=1, le=5)
    coverage: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    missing_points: List[str] = []
    hallucination_risks: List[str] = []
    must_fix: List[str] = []
    ok_to_ship: bool = False

# ---------------- LLM helper ----------------
def _llm(system: str, user: str, temperature: float = 0.2, max_tokens: int = AGENT_MAX_TOKENS) -> str:
    cli = _client()
    r = cli.responses.create(
        model=RESPONSES_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return r.output_text

def _to_str(x: Any) -> str:
    if isinstance(x, str): return x.strip()
    if isinstance(x, list): return " ".join(_to_str(i) for i in x if _to_str(i))
    if isinstance(x, dict):
        return " ".join(f"{k}: {_to_str(v)}" for k,v in x.items() if _to_str(v))
    return ("" if x is None else str(x)).strip()

def _ensure_list(x: Any) -> List[Any]:
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, (tuple,set)): return list(x)
    if isinstance(x, dict): return list(x.keys())
    if isinstance(x, str):
        parts=[p.strip() for p in re.split(r"[;,]", x) if p.strip()]
        return parts if parts else [x.strip()]
    return [x]

def _clamp_int(v: int, lo: int = 1, hi: int = 5, default: int = 3) -> int:
    try: i=int(v)
    except Exception: return default
    return max(lo, min(hi, i))

# ------------- Normalize drafter JSON -------------
def _coerce_answer_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ["summary","comparative_analysis","recommendation","general_knowledge","gaps_or_next_steps"]:
        out[k] = _to_str(d.get(k,""))
    src = d.get("per_source",{}); new_ps: Dict[str, List[str]] = {}
    if isinstance(src, dict):
        for k,v in src.items():
            canon=_canon_fw(k); bucket=new_ps.setdefault(canon,[])
            if isinstance(v,list):
                for it in v:
                    s=_to_str(it); 
                    if s: bucket.append(s)
            elif isinstance(v,dict):
                for _,sv in v.items():
                    if isinstance(sv,list):
                        for it in sv:
                            s=_to_str(it); 
                            if s: bucket.append(s)
                    else:
                        s=_to_str(sv); 
                        if s: bucket.append(s)
            else:
                s=_to_str(v); 
                if s: bucket.append(s)
    out["per_source"]=new_ps
    cits=d.get("citations",[]); new_cits: List[Dict[str,str]]=[]
    if isinstance(cits,list):
        for c in cits:
            if isinstance(c,dict):
                fw=_to_str(c.get("framework",""))
                sid=_to_str(c.get("source_id","")) or _to_str(c.get("id",""))
                pg=_to_str(c.get("page","")) or _to_str(c.get("paragraph",""))
                new_cits.append({"framework": _canon_fw(fw) if fw else "", "source_id": sid, "page": pg})
            else:
                sid=_to_str(c); 
                if sid: new_cits.append({"framework":"","source_id":sid,"page":""})
    elif isinstance(cits,dict):
        for k,v in cits.items():
            sid = _to_str(v.get("source_id","")) if isinstance(v,dict) else _to_str(v)
            pg  = _to_str(v.get("page","")) if isinstance(v,dict) else ""
            new_cits.append({"framework": _canon_fw(k), "source_id": sid, "page": pg})
    out["citations"]=new_cits
    return out

# ------------- Normalize judge JSON -------------
def _to_int_score(v: Any, default: int = 3) -> int:
    if isinstance(v,bool): return 5 if v else default
    if isinstance(v,(int,float)): return _clamp_int(round(v))
    if isinstance(v,str):
        m=re.search(r"-?\d+", v); return _clamp_int(int(m.group())) if m else default
    if isinstance(v,dict):
        for vv in v.values():
            if isinstance(vv,(int,float)) or (isinstance(vv,str) and re.search(r"-?\d+", vv or "")):
                return _to_int_score(vv, default=default)
        return default
    return default
def _to_list_of_strings(v: Any) -> List[str]:
    if v is None: return []
    if isinstance(v,list): return [_to_str(i) for i in v if _to_str(i)]
    if isinstance(v,dict):
        out=[]; 
        for k,val in v.items():
            ks=_to_str(k); vs=_to_str(val)
            combo=(f"{ks}: {vs}".strip() if vs else ks)
            if combo: out.append(combo)
        return out
    if isinstance(v,bool): return [] if not v else ["true"]
    s=_to_str(v); return [s] if s else []
def _to_bool(v: Any) -> bool:
    if isinstance(v,bool): return v
    if isinstance(v,(int,float)): return v>=1
    if isinstance(v,str): return v.strip().lower() in ("true","yes","y","ok","1","ready","ship","ok_to_ship")
    if isinstance(v,dict): return bool(v.get("ok_to_ship") or v.get("ok") or v.get("ready"))
    return False
def _coerce_review_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "grounded":      _to_int_score(d.get("grounded"), default=4),
        "coverage":      _to_int_score(d.get("coverage"), default=3),
        "clarity":       _to_int_score(d.get("clarity"), default=3),
        "structure":     _to_int_score(d.get("structure"), default=3),
        "actionability": _to_int_score(d.get("actionability"), default=3),
        "missing_points":     _to_list_of_strings(d.get("missing_points")),
        "hallucination_risks":_to_list_of_strings(d.get("hallucination_risks")),
        "must_fix":           _to_list_of_strings(d.get("must_fix")),
        "ok_to_ship":         _to_bool(d.get("ok_to_ship")),
    }

# ------------- Evidence helpers -------------
def _ev_to_per_source(ev: "EvidenceBundle", max_quotes_per_fw: int = 4) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for fw, items in ev.per_framework.items():
        quotes=[]
        for it in items[:max_quotes_per_fw]:
            q=(it.quote or "").strip().strip('"')
            if q:
                if len(q) > 420:
                    q = q[:417] + "…"
                quotes.append(q)
        if quotes: res[fw]=quotes
    return res

def _ev_to_citations(ev: "EvidenceBundle", per_fw_limit: int = 3) -> List[Dict[str, str]]:
    cits: List[Dict[str, str]]=[]
    for fw, items in ev.per_framework.items():
        for it in items[:per_fw_limit]:
            cits.append({"framework":fw,"source_id":(it.source_id or it.source_title or "").strip(),"page":(it.page or "").strip()})
    return cits

def _ensure_backfill_from_evidence(ev: "EvidenceBundle", ans: "RegulAIteAnswer") -> "RegulAIteAnswer":
    total_quotes=sum(len(v) for v in (ans.per_source or {}).values())
    if total_quotes < 2:
        ans.per_source=_ev_to_per_source(ev)
    if not ans.citations:
        ans.citations=_ev_to_citations(ev)
    return ans

# ---------------- Agent 1: Analyst ----------------
ANALYST_SYS = """You are a regulatory RAG query analyst for IFRS, AAOIFI, CBB, and internal policies.
Return STRICT JSON only (no code fences) with keys: intent, frameworks, subqueries, key_terms."""
def analyze_query(user_query: str) -> "QueryPlan":
    user = f"""Question: {user_query}
Rules:
- If comparison is implied, include multiple frameworks (use keys: IFRS, AAOIFI, optionally CBB, InternalPolicy).
- Produce 2–4 targeted subqueries using synonyms (e.g., 'held to maturity'→'amortized cost', 'Stage 3'→'credit-impaired', 'POCI'→'purchased or credit-impaired').
- Return STRICT JSON only."""
    raw = _llm(ANALYST_SYS, user)
    js = _extract_json_text(raw)
    try: data=json.loads(js)
    except Exception: data={}
    intent=_to_str(data.get("intent")) or "regulatory_compare"
    fw_list=[_canon_fw(x) for x in _ensure_list(data.get("frameworks", []))] or []
    seen=set(); frameworks=[]
    for f in fw_list:
        if f in ("IFRS","AAOIFI","CBB","InternalPolicy") and f not in seen:
            frameworks.append(f); seen.add(f)
    if not frameworks:
        uq=(user_query or "").lower()
        frameworks=["IFRS","AAOIFI"] if ("aaoifi" in uq or "fas 30" in uq or "sukuk" in uq) else ["IFRS"]
    subqueries=[s for s in (_ensure_list(data.get("subqueries", []))) if _to_str(s)]
    if not subqueries:
        subqueries=[
            "IFRS 9 classification: business model + SPPI",
            "IFRS 9 impairment staging: 12-month vs lifetime; Stage 3 interest on net",
            "AAOIFI FAS 30: ECL allowances for debt-type sukuk",
            "AAOIFI FAS 30: measurement & quotes for sukuk"
        ][:4]
    key_terms=[k for k in (_ensure_list(data.get("key_terms", []))) if _to_str(k)]
    if not key_terms:
        key_terms=["SPPI","FVOCI","FVTPL","amortised cost","Stage 3","credit-impaired","ECL"]
    return QueryPlan.model_validate({
        "intent": intent,
        "frameworks": frameworks,
        "subqueries": [ _to_str(s) for s in subqueries if _to_str(s) ],
        "key_terms": [ _to_str(k) for k in key_terms if _to_str(k) ],
    })

# ---------------- Agent 2: Retriever ----------------
RETRIEVER_SYS = """You are a retrieval assistant with access to the file_search tool (OpenAI Vector Store).
Return STRICT JSON:
{
  "per_framework": {
    "IFRS": [ { "framework": "IFRS", "quote": "...", "source_title": "...", "source_id": "...", "page": "..." } ],
    "AAOIFI": [],
    "CBB": [],
    "InternalPolicy": []
  }
}
Rules:
- Use the tool to find relevant snippets for each framework (aim 2–4 per framework).
- QUOTES MUST BE VERBATIM (≤220 chars). Include source_id and page/paragraph if available.
- Return STRICT JSON only (no prose)."""
def _retrieve_once(cli: OpenAI, user_payload: str, tools: List[Dict[str, Any]]) -> "EvidenceBundle":
    r = cli.responses.create(
        model=RESPONSES_MODEL,
        input=[{"role":"system","content":RETRIEVER_SYS},{"role":"user","content":user_payload}],
        temperature=0.0,
        max_output_tokens=AGENT_MAX_TOKENS,
        tools=tools,
        tool_choice="auto",
    )
    text=(getattr(r,"output_text","") or "").strip()
    js=_extract_json_text(text)
    data=json.loads(js)
    return EvidenceBundle.model_validate(data)

def retrieve(plan: "QueryPlan") -> "EvidenceBundle":
    if not OPENAI_VECTOR_STORE_IDS:
        raise RuntimeError("OPENAI_VECTOR_STORE_ID not set.")
    cli=_client()
    payload=json.dumps({"frameworks": plan.frameworks, "subqueries": plan.subqueries, "top_k": TOP_K})
    # 1) vector first
    bundle=_retrieve_once(cli, payload, tools=[{"type":"file_search","vector_store_ids":OPENAI_VECTOR_STORE_IDS}])
    # purge mis-attributions
    for fw, items in list(bundle.per_framework.items()):
        bundle.per_framework[fw]=[
            it for it in items if _source_matches_framework(fw, it.source_title or "", it.source_id or "")
        ]
    # 2) optional web fallback per framework
    if ALLOW_WEB:
        need_fw=[fw for fw,items in bundle.per_framework.items() if len(items)<2 and fw in ("IFRS","AAOIFI","CBB")]
        if need_fw:
            bundle2=_retrieve_once(cli, payload, tools=[
                {"type":"file_search","vector_store_ids":OPENAI_VECTOR_STORE_IDS},
                {"type":"web_search"},
            ])
            # merge additional items that look like the framework
            for fw in need_fw:
                extra=[it for it in (bundle2.per_framework.get(fw,[]) or [])
                       if _source_matches_framework(fw, it.source_title or "", it.source_id or "")]
                if extra:
                    merged=(bundle.per_framework.get(fw,[]) or []) + extra
                    # keep up to 6 per framework to avoid bloat
                    bundle.per_framework[fw]=merged[:6]
    return bundle

# ---------------- Agent 3: Drafter ----------------
DRAFTER_SYS = """You draft answers for a regulatory RAG. Return STRICT JSON only:

{
  "summary": "<220–400 words, natural consultant tone in flowing paragraphs (no numbered lists; use bullets only if unavoidable). "
             "Mini Markdown table allowed if it clearly improves clarity (≤6 rows).>",
  "per_source": {
    "IFRS": ["<short verbatim quote>", "..."],
    "AAOIFI": ["..."],
    "CBB": ["..."],
    "InternalPolicy": ["..."]
  },
  "comparative_analysis": "<short paragraph (not bullets) explaining key contrasts>",
  "recommendation": "<short paragraph with actionable guidance; no bullet list>",
  "general_knowledge": "<optional paragraph for context>",
  "gaps_or_next_steps": "<optional paragraph>",
  "citations": [
    {"framework":"IFRS","source_id":"<file_or_id>","page":"<page_or_para>"},
    {"framework":"AAOIFI","source_id":"...","page":"..."}
  ]
}

Rules:
- Use ONLY the provided evidence quotes. If a framework has <2 quotes, avoid definitive assertions; move items to 'gaps_or_next_steps'.
- IFRS: use AC / FVOCI / FVTPL (no HTM/FVTIS/FVTE); mention business model + SPPI; include ECL staging (12-month vs lifetime; Stage 3 interest on net) where pertinent.
- AAOIFI FAS 30: describe ECL-style allowances for debt-type sukuk with AAOIFI quotes.
- Keep the voice natural; avoid rigid headings or enumerations.
- Return STRICT JSON only (no prose outside JSON)."""

def draft_answer(plan: "QueryPlan", ev: "EvidenceBundle", temperature: float = 0.2) -> "RegulAIteAnswer":
    user=f"""Intent: {plan.intent}
Frameworks: {plan.frameworks}
Evidence JSON: {ev.model_dump()}
Return ONLY the JSON in the exact schema."""
    raw=_llm(DRAFTER_SYS, user, temperature=temperature)
    js=_extract_json_text(raw)
    try: data=json.loads(js)
    except Exception:
        data={"summary":_to_str(js),"per_source":{},"comparative_analysis":"","recommendation":"","general_knowledge":"","gaps_or_next_steps":"","citations":[]}
    norm=_coerce_answer_dict(data)
    for k in ["summary","comparative_analysis","recommendation","general_knowledge","gaps_or_next_steps","per_source","citations"]:
        norm.setdefault(k, "" if k not in ("per_source","citations") else ({} if k=="per_source" else []))
    ans=RegulAIteAnswer.model_validate(norm)
    return _ensure_backfill_from_evidence(ev, ans)

# ---------------- Agent 4: Judge ----------------
RUBRIC = """
MUST-FIX before ok_to_ship=true:
- Framework purity: IFRS uses only IFRS sources; AAOIFI uses only AAOIFI sources; CBB uses only CBB.
- IFRS 9: mention business model + SPPI; use AC/FVOCI/FVTPL; include ECL staging (12-month vs lifetime; Stage 3 interest on net).
- AAOIFI FAS 30: ECL-style allowances for debt-type sukuk with quotes.
- Evidence minimum: each addressed framework has ≥2 short verbatim quotes with source_id and page/para.
Return STRICT JSON with these fields/types:
- grounded, coverage, clarity, structure, actionability (ints 1–5)
- missing_points, hallucination_risks, must_fix (arrays of strings)
- ok_to_ship (boolean)
"""
JUDGE_SYS=f"You are a strict QA judge for a regulatory RAG.\nRubric:\n{RUBRIC}"
def review_answer(plan: "QueryPlan", ev: "EvidenceBundle", ans: "RegulAIteAnswer") -> "ReviewCard":
    user=f"""Question intent: {plan.intent}
Evidence: {ev.model_dump()}
Answer JSON: {ans.model_dump()}
Return JSON: grounded, coverage, clarity, structure, actionability,
missing_points, hallucination_risks, must_fix, ok_to_ship."""
    raw=_llm(JUDGE_SYS, user)
    js=_extract_json_text(raw)
    try: data=json.loads(js)
    except Exception: data={}
    norm=_coerce_review_dict(data)
    return ReviewCard.model_validate(norm)

# ---------------- Agent 5: Reviser ----------------
REVISER_SYS = """Revise the answer to satisfy the judge's MUST FIX list.
Do not invent facts. Keep the same JSON schema. Return STRICT JSON only."""
def revise_answer(plan: "QueryPlan", ev: "EvidenceBundle", ans: "RegulAIteAnswer", rev: "ReviewCard") -> "RegulAIteAnswer":
    user=f"""Original Answer: {ans.model_dump()}
Judge Review: {rev.model_dump()}
Evidence: {ev.model_dump()}
Return corrected RegulAIteAnswer JSON only."""
    raw=_llm(REVISER_SYS, user)
    js=_extract_json_text(raw)
    try: data=json.loads(js)
    except Exception: data=ans.model_dump()
    norm=_coerce_answer_dict(data)
    for k in ["summary","comparative_analysis","recommendation","general_knowledge","gaps_or_next_steps","per_source","citations"]:
        norm.setdefault(k, "" if k not in ("per_source","citations") else ({} if k=="per_source" else []))
    fixed=RegulAIteAnswer.model_validate(norm)
    return _ensure_backfill_from_evidence(ev, fixed)

# ---------------- Optional: Debate ----------------
def debate_and_select(plan: "QueryPlan", ev: "EvidenceBundle") -> "RegulAIteAnswer":
    precise=draft_answer(plan, ev, temperature=0.0)
    broad=draft_answer(plan, ev, temperature=0.6)
    editor_sys="You are an editor selecting/merging the better draft; return STRICT JSON only."
    editor_user=f"""Draft A (precise): {precise.model_dump()}
Draft B (broad): {broad.model_dump()}
Pick the better one OR merge them, optimizing for groundedness and coverage.
Return RegulAIteAnswer JSON only."""
    raw=_llm(editor_sys, editor_user)
    js=_extract_json_text(raw)
    try: data=json.loads(js)
    except Exception: return precise
    norm=_coerce_answer_dict(data)
    for k in ["summary","comparative_analysis","recommendation","general_knowledge","gaps_or_next_steps","per_source","citations"]:
        norm.setdefault(k, "" if k not in ("per_source","citations") else ({} if k=="per_source" else []))
    merged=RegulAIteAnswer.model_validate(norm)
    return _ensure_backfill_from_evidence(ev, merged)

# ---------------- Hard gates ----------------
def _text_has_any(t: str, keys: List[str]) -> bool:
    low=(t or "").lower()
    return any(k.lower() in low for k in keys)

def _hard_gate_issues(ans: "RegulAIteAnswer", ev: "EvidenceBundle") -> List[str]:
    issues: List[str]=[]
    for fw in ev.per_framework:
        if len(ev.per_framework.get(fw,[]))<2:
            issues.append(f"{fw}: has <2 quotes; move content to 'Gaps' and add docs.")
    for fw,items in ev.per_framework.items():
        for it in items:
            if not _source_matches_framework(fw, it.source_title or "", it.source_id or ""):
                issues.append(f"{fw}: drop mis-attributed source '{it.source_title}' ({it.source_id}).")
    if _text_has_any(ans.summary+ans.comparative_analysis, ["htm","fvtis","fvte"]):
        issues.append("IFRS: ban HTM/FVTIS/FVTE; use AC/FVOCI/FVTPL only.")
    if not _text_has_any(ans.summary+" "+ans.comparative_analysis, ["sppi","solely payments of principal"]):
        issues.append("IFRS: mention SPPI explicitly.")
    if not _text_has_any(ans.summary+ans.general_knowledge+ans.comparative_analysis, ["12-month","lifetime","stage 3","credit-impaired"]):
        issues.append("IFRS: include ECL staging (12-month vs lifetime; Stage 3 interest on net).")
    aa=ans.summary+ans.general_knowledge+ans.comparative_analysis
    if not _text_has_any(aa, ["expected credit loss","ecl"]):
        issues.append("AAOIFI: describe ECL-style allowances for debt-type sukuk.")
    cits=ans.citations or []
    if not cits: issues.append("Citations: add source_id and page/paragraph.")
    else:
        for c in cits:
            if not (c.get("source_id") or "").strip(): issues.append("Citations: ensure each has source_id.")
    return list(dict.fromkeys(issues))

def _synthesize_followups(plan: "QueryPlan", rev: Optional["ReviewCard"]) -> List[str]:
    qs=[]
    for s in plan.subqueries[:3]:
        qs.append(s if len(s)<=120 else s[:117]+"…")
    if rev:
        for m in (rev.missing_points or [])[:3]:
            if m and m not in qs:
                qs.append(m if len(m)<=120 else m[:117]+"…")
        for m in (rev.must_fix or [])[:2]:
            if m and m not in qs:
                qs.append(m if len(m)<=120 else m[:117]+"…")
    # prefix formatting is added by renderer
    return qs[:6]

# ---------------- Orchestrator ----------------
def run_multi_agent_rag(user_query: str, max_loops: int = 2) -> Dict[str, Any]:
    plan=analyze_query(user_query)
    ev=retrieve(plan)
    draft=debate_and_select(plan, ev) if DEBATE else draft_answer(plan, ev)

    last_card: Optional[ReviewCard]=None
    for _ in range(max_loops):
        card=review_answer(plan, ev, draft)
        hard=_hard_gate_issues(draft, ev)
        if hard:
            card.ok_to_ship=False
            seen=set(card.must_fix or [])
            for i in hard:
                if i not in seen:
                    card.must_fix.append(i)
        last_card=card
        if card.ok_to_ship: break
        if card.must_fix:
            draft=revise_answer(plan, ev, draft, card)
        else:
            break

    draft=_ensure_backfill_from_evidence(ev, draft)

    # Attach follow-ups so the renderer can show them
    out=draft.model_dump()
    out["followups"]=_synthesize_followups(plan, last_card)
    return {
        "plan": plan.model_dump(),
        "answer": out,
        "qa_review": last_card.model_dump() if last_card else None
    }
