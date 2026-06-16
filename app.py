"""
app.py — Streamlit Web Interface
==================================
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from pathlib import Path

# Inject API key from Streamlit secrets into env before rag.py loads
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TaxAI Research Assistant",
    page_icon="⚖️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, h1, h2, h3 {
        font-family: 'EB Garamond', Georgia, serif !important;
    }

    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    /* Top bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #2a2a3a;
    }

    .topbar-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    .topbar-subtitle {
        font-size: 0.82rem;
        color: #666;
        margin-top: 2px;
    }

    .badge {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #3a3a5a;
        color: #8888cc;
        font-size: 0.68rem;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Left panel */
    .left-panel {
        padding-right: 1.5rem;
        border-right: 1px solid #1e1e2e;
    }

    .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #555;
        margin-bottom: 0.6rem;
    }

    /* Answer box */
    .answer-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-left: 3px solid #5555cc;
        border-radius: 8px;
        padding: 1.5rem 1.8rem;
        line-height: 1.8;
        font-size: 1rem;
    }

    /* Sources */
    .sources-box {
        background: #111120;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-top: 1rem;
    }

    .sources-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #555;
        margin-bottom: 0.5rem;
    }

    .source-item {
        font-size: 0.88rem;
        color: #8888cc;
        padding: 3px 0;
    }

    .source-item a:hover {
        color: #aaaaee !important;
        text-decoration: underline !important;
    }

    /* Right panel empty state */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        color: #333;
        text-align: center;
    }

    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }

    .empty-text {
        font-size: 1rem;
        color: #444;
    }

    /* Input */
    .stTextArea textarea {
        background-color: #1a1a2e !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        font-size: 1rem !important;
        font-family: 'EB Garamond', Georgia, serif !important;
    }

    .stTextArea textarea:focus {
        border-color: #5555cc !important;
        box-shadow: 0 0 0 2px rgba(85,85,204,0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: #5555cc !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'EB Garamond', Georgia, serif !important;
        font-size: 1rem !important;
        transition: background 0.2s !important;
    }

    .stButton > button:hover {
        background: #4444bb !important;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.72rem;
        color: #333;
        margin-top: 2rem;
        text-align: center;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CHECK DATABASE
# ─────────────────────────────────────────────────────────

db_exists = Path("db").exists() and any(Path("db").iterdir())
if not db_exists:
    st.error("Database not found. Run `python3 ingest.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────
# LOAD RAG
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_rag():
    import rag
    return rag

rag = load_rag()

# ─────────────────────────────────────────────────────────
# PUB MAP
# ─────────────────────────────────────────────────────────

pub_names = {
    "pub17.pdf":   ("Publication 17 — Your Federal Income Tax",         "https://www.irs.gov/pub/irs-pdf/p17.pdf"),
    "pub946.pdf":  ("Publication 946 — How To Depreciate Property",     "https://www.irs.gov/pub/irs-pdf/p946.pdf"),
    "pub535.pdf":  ("Publication 535 — Business Expenses",              "https://www.irs.gov/pub/irs-pdf/p535.pdf"),
    "pub587.pdf":  ("Publication 587 — Business Use of Your Home",      "https://www.irs.gov/pub/irs-pdf/p587.pdf"),
    "pub936.pdf":  ("Publication 936 — Home Mortgage Interest",         "https://www.irs.gov/pub/irs-pdf/p936.pdf"),
    "pub502.pdf":  ("Publication 502 — Medical and Dental Expenses",    "https://www.irs.gov/pub/irs-pdf/p502.pdf"),
    "pub970.pdf":  ("Publication 970 — Tax Benefits for Education",     "https://www.irs.gov/pub/irs-pdf/p970.pdf"),
    "pub590a.pdf": ("Publication 590-A — Contributions to IRAs",        "https://www.irs.gov/pub/irs-pdf/p590a.pdf"),
    "pub590b.pdf": ("Publication 590-B — Distributions from IRAs",      "https://www.irs.gov/pub/irs-pdf/p590b.pdf"),
    "pub463.pdf":  ("Publication 463 — Travel, Gift, and Car Expenses", "https://www.irs.gov/pub/irs-pdf/p463.pdf"),
}

# ─────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────

st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-title">⚖️ TaxAI Research Assistant</div>
        <div class="topbar-subtitle">Instant answers from official IRS publications</div>
    </div>
    <div class="badge">Powered by Claude · RAG · ChromaDB</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)

    examples = [
        "Can I deduct my home office?",
        "What is the standard deduction?",
        "How much can I contribute to my IRA?",
        "Can I deduct medical expenses?",
        "How does business depreciation work?",
        "Can I deduct student loan interest?",
    ]

    for i, example in enumerate(examples):
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state["question_input"] = example

    st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your question</div>', unsafe_allow_html=True)

    question = st.text_area(
        "question",
        height=120,
        key="question_input",
        placeholder="Ask any U.S. federal tax question...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        submit = st.button("Ask TaxAI", type="primary", use_container_width=True)
    with col2:
        show_sources = st.checkbox("Show excerpts", value=False)

    st.markdown('<div class="disclaimer">For research purposes only.<br>Not a substitute for professional tax advice.</div>', unsafe_allow_html=True)

with right:
    if submit and question.strip():
        with st.spinner("Searching IRS publications..."):
            result = rag.answer_question(question)

        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        if result["sources"]:
            source_items = "".join([
                f'<div class="source-item">📄 <a href="{pub_names[s][1]}" target="_blank" style="color:#8888cc;text-decoration:none;">{pub_names[s][0]}</a></div>'
                if s in pub_names else f'<div class="source-item">📄 {s}</div>'
                for s in result["sources"]
            ])
            st.markdown(f"""
            <div class="sources-box">
                <div class="sources-title">Sources</div>
                {source_items}
            </div>
            """, unsafe_allow_html=True)

        if show_sources and result["chunks"]:
            st.markdown("<div style='margin-top:1.5rem; font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:1px;'>Retrieved Excerpts</div>", unsafe_allow_html=True)
            for i, chunk in enumerate(result["chunks"], 1):
                label = pub_names[chunk["source"]][0] if chunk["source"] in pub_names else chunk["source"]
                with st.expander(f"Excerpt {i} — {label} ({int(chunk['relevance_score']*100)}% match)"):
                    st.text(chunk["text"])

    elif submit:
        st.warning("Please enter a question.")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⚖️</div>
            <div class="empty-text">Ask a tax question to get a cited answer<br>sourced directly from IRS publications.</div>
        </div>
        """, unsafe_allow_html=True)
