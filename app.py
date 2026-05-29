"""
PDF Chat — Streamlit UI.
Run: streamlit run app.py
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from modules.pdf_processor import process_pdf, get_page_count
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════
   SIDEBAR  —  deep navy dark theme
═══════════════════════════════════ */
section[data-testid="stSidebar"] > div:first-child {
    background: #0b0f1a;
    padding: 1.5rem 1rem 1rem;
}
/* Generic text inside sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }

section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

section[data-testid="stSidebar"] label { color: #64748b !important; }

/* Metrics */
section[data-testid="stSidebar"] [data-testid="stMetricValue"]  { color: #38bdf8 !important; font-size: 1.5rem !important; }
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color: #475569 !important; }
section[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.5rem;
}

/* Dividers */
section[data-testid="stSidebar"] hr { border-color: #1e293b !important; }

/* Regular buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.07) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.14) !important;
    color: #fff !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* Download button */
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(37,99,235,0.25) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(37,99,235,0.4) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: rgba(37,99,235,0.45) !important;
    color: #fff !important;
}

/* File uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #64748b !important; }

/* Alert / success in sidebar */
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] p { color: #6ee7b7 !important; }

/* Expander */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span { color: #64748b !important; }

/* Progress bar */
section[data-testid="stSidebar"] .stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border-radius: 999px !important;
}

/* ═══════════════════════════════════
   MAIN AREA
═══════════════════════════════════ */
.stApp { background: #f8fafc; }

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    margin: 0.5rem 0;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-size: 0.94rem !important;
    background: white !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* Expander (citations) */
[data-testid="stExpander"] {
    background: #f0f7ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 10px !important;
}

/* ═══════════════════════════════════
   SUGGESTION CHIPS
═══════════════════════════════════ */
.chip-row .stButton > button {
    border-radius: 999px !important;
    border: 1.5px solid #2563eb !important;
    color: #2563eb !important;
    background: white !important;
    font-size: 0.77rem !important;
    padding: 0.35rem 0.9rem !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.35 !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.1) !important;
    transition: all 0.15s !important;
}
.chip-row .stButton > button:hover {
    background: #2563eb !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.3) !important;
}

/* ═══════════════════════════════════
   STARTER QUESTION CARDS
═══════════════════════════════════ */
.starter-row .stButton > button {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    color: #374151 !important;
    background: white !important;
    font-size: 0.83rem !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.45 !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: all 0.15s !important;
}
.starter-row .stButton > button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.15) !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _conv_path(pdf_name: str) -> str:
    h = hashlib.md5(pdf_name.encode()).hexdigest()[:8]
    os.makedirs("./storage", exist_ok=True)
    return f"./storage/conv_{h}.json"

def save_conversation(pdf_name: str, messages: list) -> None:
    with open(_conv_path(pdf_name), "w") as f:
        json.dump({"pdf": pdf_name, "saved_at": datetime.now().isoformat(),
                   "messages": messages}, f, indent=2, default=str)

def load_conversation(pdf_name: str) -> list:
    path = _conv_path(pdf_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("messages", [])
    except Exception:
        return []

def export_json(messages: list, pdf_name: str) -> str:
    return json.dumps({"pdf": pdf_name, "exported_at": datetime.now().isoformat(),
                       "conversation": [{"role": m["role"], "content": m["content"]} for m in messages]},
                      indent=2)

def _render_citations(citations: list) -> None:
    """Always-visible numbered source cards, Perplexity-style."""
    if not citations:
        return
    cards = "".join(
        f'<div style="display:flex;gap:0.75rem;align-items:flex-start;'
        f'background:white;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:0.6rem 0.85rem;margin:0.35rem 0;">'
        f'<div style="min-width:22px;height:22px;background:#2563eb;color:white;'
        f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
        f'font-size:0.7rem;font-weight:700;flex-shrink:0;">{i+1}</div>'
        f'<div style="min-width:0;">'
        f'<div style="font-weight:600;color:#1e293b;font-size:0.82rem;">Page {c["page"]}'
        f'<span style="color:#94a3b8;font-weight:400;font-size:0.73rem;margin-left:0.4rem;">'
        f'· {c["score"]} relevance</span></div>'
        f'<div style="color:#64748b;font-size:0.79rem;margin-top:0.2rem;'
        f'line-height:1.5;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;'
        f'-webkit-box-orient:vertical;">{c["snippet"]}</div>'
        f'</div></div>'
        for i, c in enumerate(citations)
    )
    st.markdown(
        f'<div style="margin-top:0.75rem;">'
        f'<div style="color:#64748b;font-size:0.75rem;font-weight:600;'
        f'letter-spacing:0.05em;margin-bottom:0.35rem;">📎 SOURCES</div>'
        f'{cards}</div>',
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"messages": [], "vector_store": None, "rag": None, "active_pdf": None,
             "chunk_count": 0, "page_count": 0, "suggestions": [],
             "pending_question": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-size:1.45rem;font-weight:800;color:white;letter-spacing:-0.03em;">
            📄 PDF Chat
        </div>
        <div style="font-size:0.75rem;color:#475569;margin-top:3px;">
            AI-powered document Q&amp;A
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop PDF here", type="pdf", label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.active_pdf:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            st.error("Set `GROQ_API_KEY` in `.env`")
        else:
            bar = st.progress(0, "Reading PDF…")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                bar.progress(20, "Chunking…")
                chunks = process_pdf(tmp_path)
                page_count = get_page_count(tmp_path)

                bar.progress(48, "Loading BGE model…")
                vs = VectorStore()

                bar.progress(72, "Embedding chunks…")
                vs.add_chunks(chunks)

                bar.progress(92, "Wiring pipeline…")
                rag = RAGPipeline(vs)

                st.session_state.update({
                    "vector_store": vs, "rag": rag,
                    "active_pdf": uploaded.name,
                    "chunk_count": len(chunks), "page_count": page_count,
                    "suggestions": [], "pending_question": None,
                    "messages": load_conversation(uploaded.name),
                })
                bar.progress(100, "✅ Ready!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── PDF info ──────────────────────────────────────────────────────────
    if st.session_state.active_pdf:
        st.divider()
        name = st.session_state.active_pdf
        short = (name[:24] + "…") if len(name) > 27 else name
        st.markdown(f"""
        <div style="background:rgba(37,99,235,0.15);border:1px solid rgba(37,99,235,0.25);
                    border-radius:10px;padding:0.65rem 0.9rem;margin-bottom:0.75rem;">
            <div style="color:#93c5fd;font-weight:600;font-size:0.83rem;">📄 {short}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.page_count)
        c2.metric("Chunks", st.session_state.chunk_count)

        st.divider()

        ca, cb = st.columns(2)
        with ca:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.suggestions = []
                save_conversation(st.session_state.active_pdf, [])
                st.rerun()
        with cb:
            if st.session_state.messages:
                st.download_button("💾 Export",
                    data=export_json(st.session_state.messages, st.session_state.active_pdf),
                    file_name="chat.json", mime="application/json",
                    use_container_width=True)

        n = len(st.session_state.messages)
        if n:
            st.markdown(
                f"<div style='color:#334155;font-size:0.74rem;text-align:center;"
                f"margin-top:0.5rem;'>💬 {n} messages · auto-saved</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("⚙️ Model details"):
        st.markdown(
            "**Embeddings**  \n`BAAI/bge-base-en-v1.5`  \n768 dims · runs locally\n\n"
            "**LLM**  \n`llama-3.1-8b-instant`  \nGroq free tier\n\n"
            "**Vector DB**  \nChromaDB · cosine\n\n"
            "**Chunking**  \n800 chars · 100 overlap · page-local"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Header banner ─────────────────────────────────────────────────────────────
if st.session_state.active_pdf:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 45%,#1d4ed8 100%);
                color:white;padding:1.1rem 1.75rem;border-radius:14px;
                margin-bottom:1.25rem;
                box-shadow:0 6px 24px rgba(29,78,216,0.22);
                display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2.2rem;line-height:1;">📄</span>
        <div style="min-width:0;">
            <div style="font-weight:700;font-size:1.05rem;white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis;">
                {st.session_state.active_pdf}
            </div>
            <div style="opacity:0.65;font-size:0.78rem;margin-top:3px;">
                {st.session_state.page_count} pages &nbsp;·&nbsp;
                {st.session_state.chunk_count} chunks &nbsp;·&nbsp;
                BGE-base embeddings &nbsp;·&nbsp; Llama 3.1 via Groq
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 45%,#1d4ed8 100%);
                color:white;padding:1.6rem 2rem;border-radius:14px;margin-bottom:1.5rem;
                box-shadow:0 6px 24px rgba(29,78,216,0.22);">
        <div style="font-weight:800;font-size:1.55rem;letter-spacing:-0.03em;">📄 PDF Chat</div>
        <div style="opacity:0.68;font-size:0.88rem;margin-top:5px;">
            Upload a PDF · Ask anything · Get grounded answers with page citations
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── No PDF: onboarding ────────────────────────────────────────────────────────
if not st.session_state.active_pdf:
    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1.25rem;">
            <div style="font-size:3.8rem;margin-bottom:0.75rem;">📂</div>
            <div style="font-size:1.2rem;font-weight:700;color:#1e293b;margin-bottom:0.4rem;">
                Upload a PDF to get started
            </div>
            <div style="color:#64748b;font-size:0.88rem;line-height:1.6;">
                Drag and drop a PDF into the sidebar.<br>
                The AI will read, chunk, and index it instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        for icon, title, desc in [
            ("🔍", "Semantic search", "BGE embeddings find relevant passages — not just keyword matches."),
            ("📎", "Page citations", "Every answer cites the exact pages it's drawing from."),
            ("💡", "Follow-up suggestions", "Get 3 smart follow-up questions after each answer."),
            ("💾", "Conversation saved", "Chats persist to disk and reload when you re-open the same PDF."),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.85rem;background:white;
                        border:1px solid #e2e8f0;border-radius:12px;padding:0.9rem 1.1rem;
                        margin:0.45rem 0;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                <span style="font-size:1.35rem;line-height:1.4;">{icon}</span>
                <div>
                    <div style="font-weight:600;color:#1e293b;font-size:0.88rem;">{title}</div>
                    <div style="color:#64748b;font-size:0.8rem;margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ── Welcome state: PDF loaded, no messages yet ────────────────────────────────
show_welcome = (not st.session_state.messages) and (not st.session_state.pending_question)

if show_welcome:
    st.markdown("""
    <div style="text-align:center;padding:1.75rem 0 1rem;">
        <div style="font-size:3rem;margin-bottom:0.6rem;">🤖</div>
        <div style="font-weight:700;font-size:1.1rem;color:#1e293b;margin-bottom:0.3rem;">
            Ready to answer your questions
        </div>
        <div style="color:#64748b;font-size:0.86rem;">
            Click a starter below or type your own question
        </div>
    </div>
    """, unsafe_allow_html=True)

    STARTERS = [
        "What is this document about?",
        "What are the key findings or conclusions?",
        "Summarise the main points in bullet form.",
        "What methodology or approach is described?",
    ]
    st.markdown('<div class="starter-row">', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    for i, s in enumerate(STARTERS):
        col = sc1 if i % 2 == 0 else sc2
        with col:
            if st.button(f"✦  {s}", key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_question = s
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"])

# ── Suggestion chips ──────────────────────────────────────────────────────────
if st.session_state.suggestions:
    st.markdown(
        "<div style='margin:0.75rem 0 0.3rem;'>"
        "<span style='color:#64748b;font-size:0.79rem;font-weight:600;letter-spacing:0.02em;'>"
        "💡 SUGGESTED FOLLOW-UPS</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    cols = st.columns(min(len(st.session_state.suggestions), 3))
    for col, sug in zip(cols, st.session_state.suggestions):
        with col:
            if st.button(sug, key=f"chip_{hash(sug) % 99999}", use_container_width=True):
                st.session_state.pending_question = sug
                st.session_state.suggestions = []
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
chat_input = st.chat_input("Ask anything about the document…")
prompt = st.session_state.pending_question or chat_input
if st.session_state.pending_question:
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reading document…"):
            try:
                answer, citations = st.session_state.rag.query(
                    prompt, st.session_state.messages[:-1]
                )
                suggestions = st.session_state.rag.suggest_questions(
                    answer, st.session_state.messages
                )
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                citations, suggestions = [], []

        st.markdown(answer)
        if citations:
            _render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
    st.session_state.suggestions = suggestions
    save_conversation(st.session_state.active_pdf, st.session_state.messages)
    st.rerun()
