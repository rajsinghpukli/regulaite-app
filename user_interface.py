# user_interface.py
# ------------------------------------------------------------
# RegulAIte — Pro Chat UI (hybrid-ready; no logic breaks)
# - Auto-sizing chat surface (no clipping / no big blank box)
# - Follow-up chips wrap text; 1–3 columns for better readability
# - REG_THEME presets: khaleeji | sky | slate | emerald | violet | rose | desert | brand
#   brand can be customized with env: BRAND_PRIMARY(_2), BRAND_BG, BRAND_PANEL,
#   BRAND_INK(_SUB), BRAND_USER, BRAND_ASSISTANT, BRAND_TAG (hex)
# - Behaviours preserved: follow-up chips auto-ask, signup, status expander,
#   ask-anything card, staged loading messages
# - NEW: Mode (Auto/Regulatory/Research/Quick/Mixed), Include Web, Depth controls
# ------------------------------------------------------------
import os, sys, json, time, secrets, hashlib, re
from datetime import datetime
import streamlit as st
from streamlit.components.v1 import html

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "rag"))

from rag.pipeline import ask as be_ask, PERSIST_DIR, TOP_K_DEFAULT

USE_OPENAI_VECTOR = bool(os.getenv("OPENAI_VECTOR_STORE_ID"))
HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))

AUTH_MODE = os.getenv("REG_AUTH_MODE", "file").lower()  # "file" or "env"
USERS_FILE = os.getenv("REG_USERS_FILE", os.path.join(ROOT, "users.json"))
PEPPER = os.getenv("REG_PEPPER", "dev-pepper-change-me")
REG_THEME = os.getenv("REG_THEME", "khaleeji").lower()

# ---------------------------- auth helpers (unchanged logic) ----------------------------
def _pbkdf2_hash(p, s): return hashlib.pbkdf2_hmac("sha256", (PEPPER + p).encode(), bytes.fromhex(s), 200_000, dklen=32).hex()
def _gen_salt_hex(n=16): return secrets.token_hex(n)
def _load_users_file():
    if not os.path.exists(USERS_FILE): return {}
    try:
        with open(USERS_FILE,"r",encoding="utf-8") as f:
            d=json.load(f); return d if isinstance(d,dict) else {}
    except Exception: return {}
def _save_users_file(u):
    tmp=USERS_FILE+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(u,f,indent=2)
    os.replace(tmp,USERS_FILE)
def _ensure_demo_user_file():
    u=_load_users_file()
    if not u:
        s=_gen_salt_hex()
        u["demo"]={"pwd_hash":_pbkdf2_hash("demo",s),"salt":s,"created_at":datetime.utcnow().isoformat()}
        _save_users_file(u)
def _create_user_file(username,password) -> bool:
    username=username.strip().lower()
    if not username or not password: return False
    u=_load_users_file()
    if username in u: return False
    s=_gen_salt_hex()
    u[username]={"pwd_hash":_pbkdf2_hash(password,s),"salt":s,"created_at":datetime.utcnow().isoformat()}
    _save_users_file(u); return True
def _verify_user_file(username,password) -> bool:
    username=username.strip().lower()
    u=_load_users_file(); rec=u.get(username)
    return bool(rec and secrets.compare_digest(rec["pwd_hash"], _pbkdf2_hash(password, rec["salt"])))
def _load_env_users():
    salt=os.getenv("REG_SALT","dev-salt-change-me"); raw=os.getenv("REG_USERS",""); users={}
    if raw.strip():
        try:
            data=json.loads(raw)
            for name,pwd in data.items(): users[name.lower()]=hashlib.sha256((salt+str(pwd)).encode()).hexdigest()
        except Exception: pass
    if not users: users["demo"]=hashlib.sha256((salt+"demo").encode()).hexdigest()
    return users,salt
ENV_USERS, ENV_SALT = _load_env_users()
def _env_verify(username,password) -> bool:
    if not username or not password: return False
    digest=hashlib.sha256((ENV_SALT+password).encode()).hexdigest()
    return ENV_USERS.get(username.strip().lower())==digest
def _is_authenticated(): return bool(st.session_state.get("auth_user"))
def _logout():
    for k in ("auth_user","chat","composer","busy","auto_send","login_attempts_until","login_attempts"):
        st.session_state.pop(k, None)

st.set_page_config(page_title="RegulAIte — Regulatory Assistant", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# ---------------------------- theme (presets + brand) ----------------------------
def _hx(name, default):
    val = os.getenv(name, "").strip()
    if re.fullmatch(r"#?[0-9A-Fa-f]{6}", val or ""):
        return val if val.startswith("#") else f"#{val}"
    return default

def palette(theme: str):
    presets = {
        "khaleeji": dict(bg="#EAF2FF", panel="#FFFFFF", ink="#0B2441", ink_sub="#5C7A99",
                         primary="#1D6FB7", primary2="#2A8BD8", user="#E5F1FF", assistant="#FFFFFF", tag="#FFF7C2"),
        "sky":      dict(bg="#ECF6FF", panel="#FFFFFF", ink="#0F2F49", ink_sub="#6B88A6",
                         primary="#0EA5E9", primary2="#60A5FA", user="#E9F3FF", assistant="#FFFFFF", tag="#FFF5BF"),
        "slate":    dict(bg="#F5F7FB", panel="#FFFFFF", ink="#1F2937", ink_sub="#6B7280",
                         primary="#334155", primary2="#64748B", user="#EEF2FB", assistant="#FFFFFF", tag="#FFF0CC"),
        "emerald":  dict(bg="#ECFDF5", panel="#FFFFFF", ink="#052e2b", ink_sub="#5B8C82",
                         primary="#10B981", primary2="#059669", user="#E6FFF7", assistant="#FFFFFF", tag="#FFF4BF"),
        "violet":   dict(bg="#F3F1FF", panel="#FFFFFF", ink="#26164B", ink_sub="#7C7AA1",
                         primary="#7C3AED", primary2="#5B21B6", user="#F0E9FF", assistant="#FFFFFF", tag="#FFE8A3"),
        "rose":     dict(bg="#FFF1F5", panel="#FFFFFF", ink="#4A1020", ink_sub="#A67184",
                         primary="#E11D48", primary2="#BE123C", user="#FFE6EC", assistant="#FFFFFF", tag="#FFF0C2"),
        "desert":   dict(bg="#FFF7ED", panel="#FFFFFF", ink="#3F2A18", ink_sub="#A28970",
                         primary="#C2410C", primary2="#9A3412", user="#FFF0E2", assistant="#FFFFFF", tag="#FFF3C7"),
    }
    base = presets.get(theme)
    if not base:  # brand or unknown → use env overrides
        base = dict(
            bg=_hx("BRAND_BG", "#EDF5FF"), panel=_hx("BRAND_PANEL", "#FFFFFF"),
            ink=_hx("BRAND_INK", "#0E3C66"), ink_sub=_hx("BRAND_INK_SUB", "#5C7A99"),
            primary=_hx("BRAND_PRIMARY", "#0EA5E9"), primary2=_hx("BRAND_PRIMARY_2", "#2563EB"),
            user=_hx("BRAND_USER", "#E6F7FF"), assistant=_hx("BRAND_ASSISTANT", "#FFFFFF"),
            tag=_hx("BRAND_TAG", "#FFF7C2"),
        )
    base["border"]  = "rgba(37,99,235,0.18)"
    base["accent"]  = base["primary"]
    base["accent2"] = base["primary2"]
    return base

VAR = palette(REG_THEME)

# ---------------------------- CSS (presentation only) ----------------------------
st.markdown(f"""
<style>
:root {{
  --kh-primary:{VAR['primary']}; --kh-primary-2:{VAR['primary2']};
  --kh-bg:{VAR['bg']}; --kh-panel:{VAR['panel']};
  --kh-ink:{VAR['ink']}; --kh-ink-sub:{VAR['ink_sub']};
  --kh-user:{VAR['user']}; --kh-assistant:{VAR['assistant']};
  --kh-border:{VAR['border']}; --kh-accent:{VAR['accent']};
  --kh-accent-2:{VAR['accent2']}; --kh-tag:{VAR['tag']};
}}
body {{
  color:var(--kh-ink);
  background:
    radial-gradient(1100px 520px at 8% -6%, rgba(14,165,233,0.16), transparent 60%),
    radial-gradient(900px 420px at 110% 10%, rgba(37,99,235,0.12), transparent 60%),
    linear-gradient(180deg, var(--kh-bg), #FFFFFF 60%);
}}
.block-container {{ max-width:1100px; padding-top:2.2rem; padding-bottom:110px; }}
p, li {{ font-size:1rem; line-height:1.65; }}
ul {{ margin: .25rem 0 .9rem 1.2rem; }}
li {{ margin:.18rem 0; }}

/* Hero */
.kh-hero {{
  background:
    radial-gradient(40% 140% at 100% -20%, rgba(255,255,255,0.18), transparent 60%),
    linear-gradient(135deg, var(--kh-primary) 0%, var(--kh-primary-2) 70%);
  color:#fff; border-radius:20px; padding:22px 24px; margin-bottom:14px; display:flex; gap:18px; align-items:center;
  box-shadow: 0 18px 40px rgba(16,84,150,0.24);
}}
.kh-hero .logo {{ font-size:2.1rem; }}
.kh-title {{ font-weight:850;font-size:1.26rem;letter-spacing:.2px }}
.kh-sub {{ opacity:.97;margin-top:2px }}

/* Ask-anything */
.ask-card {{
  border:2px dashed rgba(29,111,183,0.20); background:#fff; color:var(--kh-ink);
  border-radius:16px; padding:14px 16px; margin-top:8px; box-shadow:0 8px 24px rgba(0,0,0,0.05);
}}

/* Chat surface — AUTO SIZE, NO CLIP */
.chat-wrapper {{
  border:1px solid var(--kh-border); background:var(--kh-panel); border-radius:16px; padding:12px; margin-top:16px;
  display:flex; flex-direction:column; box-shadow: 0 10px 26px rgba(13,62,110,0.10);
  height:auto;             /* no fixed height */
}}
.chat-scroller {{
  flex:1 1 auto; height:auto;            /* no height limit */
  overflow:visible;                       /* no clipping */
  padding:6px 6px 8px 6px;
}}
.chat-scroller::after{{ content:""; display:block; height:18px; }} /* bottom breathing room */

/* Bubbles */
.msg {{
  margin:10px 0 16px 0; max-width:88%; padding:14px 16px; border-radius:18px;
  border:1px solid var(--kh-border);
  background: linear-gradient(180deg, var(--kh-panel), #fff);
  box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}}
.user {{ margin-left:auto; background: linear-gradient(180deg, var(--kh-user), #fff); }}
.assistant {{ margin-right:auto; background: var(--kh-assistant); }}

/* Assistant markdown accents */
.assistant h1, .assistant h2, .assistant h3 {{ color:var(--kh-ink); font-weight:800; margin:.35rem 0 .75rem 0; }}
.assistant h1 {{ font-size:1.62rem; }}
.assistant h2 {{
  font-size:1.28rem; background:linear-gradient(90deg,rgba(14,165,233,0.10),transparent);
  padding:8px 12px; border-radius:10px; border-left:6px solid var(--kh-accent);
}}
.assistant h3 {{ font-size:1.12rem; border-left:4px solid var(--kh-accent-2); padding-left:10px; }}
.assistant strong {{ background: var(--kh-tag); padding:0 .18rem; border-radius:.28rem; }}

/* Follow-up chips (WRAP TEXT) */
.followups {{ margin-top: 10px; }}
.followups .stButton>button {{
  background:linear-gradient(180deg,#F1F8FF,#FFFFFF); color:var(--kh-ink); border:1px solid var(--kh-border);
  border-radius:14px; padding:10px 14px; box-shadow:0 3px 10px rgba(0,0,0,0.06);
  width:100%; max-width:none; font-weight:600;
  white-space: normal !important; word-break: break-word; text-wrap: pretty; text-align:left; line-height:1.25;
}}

/* Composer (sticky) */
.composer-wrap {{ position: sticky; bottom: 10px; z-index: 20; background: transparent; padding-top: 8px; }}
.composer {{
  border:1px solid var(--kh-border); background:var(--kh-panel); border-radius:16px; padding:14px; margin-top:10px;
  box-shadow: 0 10px 24px rgba(29,111,183,0.18);
}}
.composer .stButton>button {{
  background:linear-gradient(135deg, var(--kh-primary), var(--kh-primary-2));
  color:#fff; border:0; border-radius:999px; padding:10px 20px; font-weight:700;
  box-shadow:0 12px 28px rgba(29,111,183,0.28);
}}

.dot {{ display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:8px; box-shadow:0 0 0 3px rgba(0,0,0,0.05) inset; }}
.ok {{ background:#22c55e; }} .bad {{ background:#ef4444; }}

@media (max-width:640px) {{
  .msg {{ max-width:94%; padding:12px 14px; border-radius:14px; }}
  .assistant h1 {{ font-size:1.35rem; }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------- UI bits (same behaviour) ----------------------------
def hero():
    st.markdown("""
    <div class="kh-hero">
      <div class="logo">🏦</div>
      <div>
        <div class="kh-title">RegulAIte — Regulatory Assistant</div>
        <div class="kh-sub">Empowering Khaleeji Bank with AI-Driven Accounting Standards & Policies</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def ask_anything(compact: bool):
    # Encourage mixed policy + market use
    example = "Compare IFRS vs AAOIFI impairment, and relate to Bahrain NPL trends (latest)."
    size = ".92rem" if compact else "1rem"
    st.markdown(
        f"""
        <div class="ask-card">
          <div style="font-weight:800;">Ask anything about IFRS • AAOIFI • CBB • Internal Policies</div>
          <div style="opacity:.9; margin-top:6px; font-size:{size}">Example: <code>{example}</code></div>
        </div>
        """,
        unsafe_allow_html=True
    )

def sidebar():
    with st.sidebar:
        st.header("Settings")

        # --- High-level style dropdown (for clients) ---
        style = st.selectbox(
            "Answer style",
            ["Long / Consultant (recommended)", "Smart (auto)", "Concise"],
            index=0,
            help="Sets tone/length automatically. Advanced overrides are optional below."
        )

        # Defaults derived from the chosen style
        if style.startswith("Long"):
            default_depth = "deep"
            default_include_web = True
            default_evidence_mode = True
            default_mode = "Auto"
        elif style.startswith("Concise"):
            default_depth = "concise"
            default_include_web = True
            default_evidence_mode = False
            default_mode = "Auto"
        else:  # Smart (auto)
            default_depth = "standard"
            default_include_web = True
            default_evidence_mode = False
            default_mode = "Auto"

        # Retrieval knob
        top_k = st.slider("Top-K hint", 1, 12, TOP_K_DEFAULT)

        st.caption("Style applies by default. Use Advanced to override any setting.")

        # --- Advanced controls (optional) ---
        with st.expander("Advanced (optional overrides)", expanded=False):
            # Mode / Router hint
            modes = ["Auto", "Regulatory", "Research", "Quick", "Mixed"]
            try:
                mode_index = modes.index(default_mode)
            except Exception:
                mode_index = 0
            mode = st.radio("Mode", modes, horizontal=False, index=mode_index)
            mode_hint = dict(Auto=None, Regulatory="regulatory", Research="research", Quick="quick", Mixed="mixed")[mode]

            # Include web
            include_web = st.toggle("Include web sources (when relevant)", value=default_include_web)

            # Depth
            depth_options = ["Concise", "Standard", "Deep"]
            depth_map = {"concise": 0, "standard": 1, "deep": 2}
            depth_index = depth_map.get(default_depth, 1)
            depth = st.radio("Depth", depth_options, index=depth_index).lower()

            # Evidence Mode (quotes block)
            # Respect the toggle and also write to env so the pipeline picks it up live
            evidence_mode = st.toggle("Evidence Mode (2–5 short quotes per framework)", value=default_evidence_mode)
            os.environ["EVIDENCE_MODE"] = "1" if evidence_mode else "0"

        st.divider()
        with st.expander("ⓘ System status", expanded=False):
            vs_state = "ok" if USE_OPENAI_VECTOR else "bad"
            api_state = "ok" if HAS_API_KEY else "bad"
            st.markdown(
                f"""
                <div><span class="dot {vs_state}"></span>Vector store: {"connected" if USE_OPENAI_VECTOR else "not set"}</div>
                <div style="opacity:.85;margin-left:18px;">Label: <code>{PERSIST_DIR}</code></div>
                <div><span class="dot {api_state}"></span>LLM API key: {"available" if HAS_API_KEY else "missing"}</div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🆕 New chat"): st.session_state["chat"] = []
        with c2:
            md = export_chat_markdown()
            st.download_button(
                "⬇️ Chat (.md)", md,
                file_name=f"regulaite_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown", key="dl_md_btn"
            )

        st.divider()
        if _is_authenticated():
            if st.button("🚪 Logout"):
                _logout(); st.rerun()

    # Map 'style' default into return values when user didn't override in Advanced.
    # Because Streamlit widgets persist, the values we collected above are always authoritative.
    # Return knobs for the pipeline call:
    return top_k, mode_hint, include_web, depth

def _framework_chips(md: str) -> str:
    frs=[]
    if "IFRS" in md: frs.append("IFRS")
    if ("AAOIFI" in md) or ("FAS " in md): frs.append("AAOIFI")
    if "CBB" in md: frs.append("CBB")
    if ("Internal Policy" in md) or ("InternalPolicy" in md): frs.append("Internal Policy")
    # Heuristic Web chip when web snippet labels or links exist
    if ("[W1]" in md) or ("http://" in md) or ("https://" in md):
        frs.append("Web")
    return "".join([f"<span style='display:inline-block;margin-right:8px;padding:6px 10px;border-radius:999px;background:#F1F8FF;border:1px solid var(--kh-border);font-size:.82rem;color:var(--kh-ink);'>{f}</span>" for f in frs]) or ""

def export_chat_markdown() -> str:
    out=[f"# RegulAIte — Transcript", f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", ""]
    for m in st.session_state.get("chat", []):
        if m["role"]=="user": out.append(f"## ❓ {m['content']}\n")
        else: out.append(m["content"]); out.append("")
    return "\n".join(out)

_FOLLOWUP_RE = re.compile(r"^\s*-\s*\[Ask\]\s*(.+?)\s*$", re.I)
def extract_followups(md:str):
    ret=[]
    for ln in md.splitlines():
        m=_FOLLOWUP_RE.match(ln)
        if m:
            q=m.group(1).strip()
            if q and q not in ret: ret.append(q)
    return ret[:8]
def strip_followup_bullets(md:str) -> str:
    return "\n".join([ln for ln in md.splitlines() if not _FOLLOWUP_RE.match(ln)])

# ---------------------------- pages (behaviour unchanged) ----------------------------
def page_login():
    if AUTH_MODE=="file":
        _ensure_demo_user_file()

    hero()
    st.subheader("Sign in")

    now=time.time(); until=st.session_state.get("login_attempts_until",0)
    if now<until:
        st.error(f"Too many attempts. Try again in {int(until-now)}s."); return

    with st.form("login_form"):
        u=st.text_input("Username")
        p=st.text_input("Password", type="password")
        ok=st.form_submit_button("Sign in")

    if ok:
        valid=_env_verify(u,p) if AUTH_MODE=="env" else _verify_user_file(u,p)
        if valid:
            st.session_state["auth_user"]=u.strip().lower()
            st.success("Signed in."); st.session_state.pop("login_attempts",None); st.session_state.pop("login_attempts_until",None)
            st.rerun()
        else:
            attempts=st.session_state.get("login_attempts",0)+1
            st.session_state["login_attempts"]=attempts
            if attempts>=5: st.session_state["login_attempts_until"]=time.time()+30
            st.error("Invalid credentials.")

    if AUTH_MODE=="file":
        st.divider(); st.caption("No account? Create one:")
        with st.form("signup_form"):
            su = st.text_input("New username")
            sp = st.text_input("New password", type="password")
            sc = st.text_input("Confirm password", type="password")
            create = st.form_submit_button("Create account")
        if create:
            errs=[]
            if not su.strip(): errs.append("Username required.")
            if len(sp) < 10: errs.append("Password must be at least 10 characters.")
            if sp != sc: errs.append("Passwords do not match.")
            if errs:
                for e in errs: st.error(e)
            else:
                ok=_create_user_file(su, sp)
                st.success("Account created. You can sign in now.") if ok else st.error("That username is taken.")

def hero_top_and_ask_card():
    hero()
    ask_anything(compact=bool(st.session_state.get("chat")))

def page_chat():
    hero_top_and_ask_card()
    top_k, mode_hint, include_web, depth = sidebar()

    if "chat" not in st.session_state: st.session_state.chat=[]
    if "composer" not in st.session_state: st.session_state["composer"]=""
    if "busy" not in st.session_state: st.session_state["busy"]=False
    if "auto_send" not in st.session_state: st.session_state["auto_send"]=False

    # ----- CHAT HISTORY -----
    if st.session_state.chat:
        st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-scroller'>", unsafe_allow_html=True)

        for m in st.session_state.chat:
            klass = "user" if m["role"] == "user" else "assistant"
            st.markdown(f"<div class='msg {klass}'>", unsafe_allow_html=True)

            if m["role"] == "user":
                st.markdown(f"**🧑‍💼** {m['content']}")
            else:
                display_md = strip_followup_bullets(m["content"])
                st.markdown(display_md, unsafe_allow_html=True)

                # FOLLOW-UP CHIPS (auto-sends on click) — wider columns (max 3)
                follows = extract_followups(m["content"])
                if follows:
                    st.markdown("<div class='followups'>", unsafe_allow_html=True)
                    cols = st.columns(min(3, len(follows)))  # <-- wider
                    for i, q in enumerate(follows):
                        with cols[i % len(cols)]:
                            if st.button(q, key=f"follow_{hash(q)}"):
                                st.session_state["composer"] = q
                                st.session_state["auto_send"] = True
                                st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)   # close scroller
        # Ensure first render always shows the end
        html("<script>setTimeout(()=>{window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});},0);</script>", height=0)
        st.markdown("</div>", unsafe_allow_html=True)   # close wrapper

    # ----- COMPOSER -----
    st.markdown("<div class='composer-wrap'><div class='composer'>", unsafe_allow_html=True)
    with st.form("composer_form", clear_on_submit=False):
        user_text = st.text_area("Type here…", key="composer", height=120, label_visibility="collapsed",
                                 placeholder="e.g., Assess impairment triggers and ECL for sukuk in default or restructuring.")
        submit_btn = st.form_submit_button("Ask", disabled=st.session_state["busy"])
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Submit flow
    if st.session_state["auto_send"] and not st.session_state["busy"] and st.session_state["composer"].strip():
        submit_btn = True; user_text = st.session_state["composer"]; st.session_state["auto_send"] = False

    if submit_btn and user_text.strip() and not st.session_state["busy"]:
        st.session_state["busy"] = True
        try:
            qtxt = user_text.strip()
            status = st.status("Understanding your question…", expanded=True)
            status.update(label="Retrieving evidence from approved sources…")
            st.session_state.chat.append({"role": "user", "content": qtxt, "question": qtxt})

            # NEW: pass mode_hint/include_web/depth to pipeline
            res = be_ask(qtxt, k=top_k, mode_hint=mode_hint, include_web=include_web, depth=depth)

            status.update(label="Selecting evidence & compiling citations…")
            status.update(label="Formatting answer…")

            if res.get("mode") == "vector_error":
                st.session_state.chat.append({"role": "assistant", "content": f"**Error:** {res.get('answer')}", "question": qtxt})
                status.update(state="error", label="Something went wrong while answering.")
            else:
                md = (res.get("answer_markdown") or "").strip() or "_No answer returned._"
                chips_html = _framework_chips(md)
                md_with_heading = f"## ❓ {qtxt}\n\n{chips_html}\n\n" + md
                st.session_state.chat.append({"role": "assistant", "content": md_with_heading, "question": qtxt})
                status.update(state="complete", label="Answer ready ✅")
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"**Error:** {e}", "question": qtxt})
        finally:
            st.session_state["busy"] = False
            if len(st.session_state.chat) > 30:
                st.session_state.chat = st.session_state.chat[-30:]
            st.rerun()

def page_about():
    hero()
    st.subheader("About RegulAIte")
    st.markdown(
        """
**RegulAIte** is a cloud-ready regulatory assistant for **Khaleeji Bank**.  
It answers questions from **IFRS, AAOIFI, CBB Rulebook Vol.2** and **Internal Policies** with short quotes and sources.  
When users ask about market context (e.g., sector revenues, ratios), it can also blend **trusted web sources** with internal rulebooks.

**What we solve**
- Faster, defensible answers with citations  
- Lower compliance risk via consistent interpretation  
- Clear next steps via follow-up suggestions and practical guidance

**Scope & governance**
- Phase 1: AAOIFI, IFRS, CBB Vol.2, Internal Finance Policies  
- Additions via **Change Request (CR)** to keep scope focused for the CFO

**How it works**
- Intent routing (Regulatory • Research • Quick • Mixed/Compare-to-market)  
- Hybrid retrieval: Vector store (primary for policies) + optional web (for market data, news, stats)  
- Informative narrative formatting by default; depth can be **Concise / Standard / Deep**
        """.strip()
    )

# ---------------------------- router ----------------------------
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
if not st.session_state.get("auth_user"):
    if AUTH_MODE=="file": _ensure_demo_user_file()
    st.subheader("Sign in")
    now=time.time(); until=st.session_state.get("login_attempts_until",0)
    if now<until:
        st.error(f"Too many attempts. Try again in {int(until-now)}s.")
    else:
        with st.form("login_form"):
            u=st.text_input("Username")
            p=st.text_input("Password", type="password")
            ok=st.form_submit_button("Sign in")
        if ok:
            valid=_env_verify(u,p) if AUTH_MODE=="env" else _verify_user_file(u,p)
            if valid:
                st.session_state["auth_user"]=u.strip().lower()
                st.success("Signed in."); st.session_state.pop("login_attempts",None); st.session_state.pop("login_attempts_until",None)
                st.rerun()
            else:
                attempts=st.session_state.get("login_attempts",0)+1; st.session_state["login_attempts"]=attempts
                if attempts>=5: st.session_state["login_attempts_until"]=time.time()+30
                st.error("Invalid credentials.")
        if AUTH_MODE=="file":
            st.divider(); st.caption("No account? Create one:")
            with st.form("signup_form"):
                su = st.text_input("New username")
                sp = st.text_input("New password", type="password")
                sc = st.text_input("Confirm password", type="password")
                create = st.form_submit_button("Create account")
            if create:
                errs=[]
                if not su.strip(): errs.append("Username required.")
                if len(sp) < 10: errs.append("Password must be at least 10 characters.")
                if sp != sc: errs.append("Passwords do not match.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    ok=_create_user_file(su, sp)
                    st.success("Account created. You can sign in now.") if ok else st.error("That username is taken.")
else:
    tab = st.radio("Navigation", ["Chat", "About"], horizontal=True, label_visibility="collapsed")
    page_chat() if tab=="Chat" else page_about()
