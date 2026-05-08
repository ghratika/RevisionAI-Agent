import streamlit as st
import os
import tempfile
import time

from tools import (
    clean_transcript, chunk_transcript,
    generate_all_notes,
    store_memory, retrieve_memory_with_groq, list_subjects,
    export_to_pdf,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lecture Revision Agent",
    page_icon="📚",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stTextArea textarea {
        background-color: #1a1a2e;
        color: #e0e0e0;
        border: 1px solid #333366;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-1px);
    }
    .note-card {
        background: #1a1a2e;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    .note-card h3 {
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📚 Lecture Revision Agent")
st.caption("Paste any lecture transcript → get exam-ready revision notes instantly.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    subject = st.text_input("📖 Notes Subject", placeholder="e.g. Thermodynamics")

    st.markdown("**🎓 Difficulty Level**")
    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"],
        index=1,
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("🔒 Powered by Groq + Llama 3 · Free · No API key needed")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📝 Generate Notes", "🔍 Ask Your Notes"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate Notes
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    transcript = st.text_area(
        "Paste your lecture transcript here",
        height=300,
        placeholder="Paste the full transcript of your lecture here..."
    )

    process_btn = st.button("🚀 Generate Revision Notes", type="primary")

    if process_btn:
        if not subject.strip():
            st.warning("⚠️ Please enter a subject name in the sidebar.")
        elif not transcript.strip():
            st.warning("⚠️ Please paste a transcript first.")
        else:
            with st.spinner("Cleaning and chunking transcript..."):
                cleaned = clean_transcript(transcript)
                chunks = chunk_transcript(cleaned)

            with st.spinner(f"Generating revision notes using Llama 3... ({len(chunks)} chunks)"):
                notes = generate_all_notes(chunks, subject, difficulty)

            # Store to memory
            store_memory(subject, "\n\n".join(chunks), {"type": "transcript"})

            notes["subject"] = subject

            st.success("✅ Notes generated!")
            st.markdown(f"**Subject:** {subject} &nbsp;|&nbsp; **Level:** {difficulty}")
            st.divider()

            col1, col2 = st.columns(2)

            def render_card(icon, title, content):
                if not content:
                    return
                # Check if content is an error from Groq
                if content.startswith("Error:"):
                    st.markdown(f"### {icon} {title}")
                    st.warning("⚠️ Could not generate this section. Your transcript may be too long or there was a temporary issue. Try again or use a shorter transcript.")
                    st.markdown("---")
                    return
                st.markdown(f"### {icon} {title}")
                # Render markdown properly so **bold** works
                st.markdown(content, unsafe_allow_html=False)
                st.markdown("---")

            with col1:
                render_card("📋", "Summary", notes.get("summary"))
                render_card("📌", "Key Concepts", notes.get("concepts"))

            with col2:
                render_card("📖", "Important Terms", notes.get("terms"))
                render_card("⚡", "Revision Bullets", notes.get("revision_bullets"))

            if notes.get("exam_questions"):
                render_card("🎯", "Exam Questions", notes.get("exam_questions"))



            # ── PDF Export ─────────────────────────────────────────

            st.divider()
            safe_subject = subject.replace(" ", "_")
            pdf_path = os.path.join(
                tempfile.gettempdir(),
                f"{safe_subject}_{int(time.time())}.pdf"
            )

            try:
                export_to_pdf(notes, pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download PDF",
                        f,
                        file_name=f"{safe_subject}_revision.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"PDF export failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask Your Notes (Memory Query)
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("Ask any question about lectures you've already processed.")

    mem_subject = st.text_input("📖 Subject", placeholder="e.g. Thermodynamics", key="mem_subject")
    query = st.text_input("❓ Your Question", placeholder="e.g. What is entropy?")

    if st.button("🔍 Search My Notes", key="search_btn"):
        if not mem_subject.strip():
            st.warning("Enter a subject.")
        elif not query.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Searching your notes..."):
                answer = retrieve_memory_with_groq(mem_subject, query)

            if answer:
                st.markdown("### 💡 Answer")
                st.markdown(answer)
            else:
                st.info(f"No notes found for '{mem_subject}'. Process a transcript first.")