"""
Streamlit UI for the JEE AI Tutor.

Run locally:
    streamlit run app.py

Requires GEMINI_API_KEY (env var, a local .env file, or Streamlit secrets)
and a prebuilt vector store (python embeddings/build_concept_store.py).
"""

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _has_api_key():
    return any(os.environ.get(v) for v in API_KEY_VARS)


def _load_env():
    """Load .env (all vars) and pull the API key from Streamlit secrets if unset."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    if not _has_api_key():
        for var in API_KEY_VARS:
            try:
                if var in st.secrets:
                    os.environ[var] = st.secrets[var]
                    break
            except Exception:
                pass


@st.cache_resource(show_spinner="Preparing the tutor (first boot builds the vector store)…")
def get_tutor():
    """
    Import the tutor once per server process (loads the embedding model).

    Builds the vector store first if it is missing, so a fresh deployment works
    without committing the store. The retriever connects to the collections at
    import time, so the build must happen before importing the tutor.
    """
    from embeddings.build_concept_store import ensure_store
    ensure_store()
    from tutor.tutor_inference import tutor
    return tutor


EXAMPLES = [
    "Why does Gibbs free energy determine if a reaction is spontaneous?",
    "Why does the equilibrium constant decrease with temperature for exothermic reactions?",
    "How does torque relate to angular momentum for a rigid body?",
    "When is a function guaranteed to have a root in an interval?",
]

st.set_page_config(page_title="JEE AI Tutor", page_icon="🧠", layout="centered")
_load_env()

st.title("🧠 AI Tutor for IIT-JEE")
st.caption(
    "Concept-aware tutoring grounded in JEE Advanced past-year questions. "
    "Explains the *why*, flags common exam traps, and walks you through real "
    "problems step by step — after you've tried them."
)

with st.sidebar:
    st.header("Options")
    subject = st.selectbox(
        "Subject",
        ["All subjects", "Physics", "Chemistry", "Mathematics"],
        help="Restrict retrieval to one subject, or search across all three.",
    )
    st.markdown("**Try an example:**")
    for ex in EXAMPLES:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query"] = ex
    st.divider()
    if not _has_api_key():
        st.warning("GEMINI_API_KEY not set — see .env.example.")

query = st.text_area(
    "Ask a concept question",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Why does entropy increase in the free expansion of a gas?",
    height=100,
)

if st.button("Explain", type="primary"):
    if not query.strip():
        st.info("Type a question first.")
    elif not _has_api_key():
        st.error("GEMINI_API_KEY is not set. Add it to a .env file or Streamlit secrets.")
    else:
        subject_filter = None if subject == "All subjects" else subject
        try:
            tutor = get_tutor()
            with st.spinner("Thinking through the concept…"):
                result = tutor(query.strip(), subject=subject_filter)
        except Exception as exc:  # surface config/API errors to the student
            st.error(f"Something went wrong: {exc}")
            st.stop()

        if "error" in result:
            st.warning(result["error"])
            st.stop()

        concept = result["primary_concept"]
        st.subheader("📚 Concept")
        st.markdown(
            f"**{concept['concept'].replace('_', ' ').title()}** "
            f"· {concept['subject']} › {concept['chapter']}"
        )
        st.markdown(result["concept_explanation"])

        if result["common_traps"]:
            st.subheader("⚠️ Common traps")
            for trap in result["common_traps"]:
                with st.container(border=True):
                    st.markdown(f"**{trap['trap']}**")
                    st.markdown(f"*Why students fall for it:* {trap['why_students_fall_for_it']}")
                    st.markdown(f"*Correction:* {trap['correction']}")

        if result["worked_examples"]:
            st.subheader("📝 Practice — try first, then reveal")
            st.caption(
                "Attempt each question yourself before opening the hints. The "
                "struggle is where the learning happens."
            )
            # Each problem is a bordered container. Question is visible; the
            # approach hint and full walkthrough are behind separate expanders so
            # the student attempts it first. (Streamlit forbids nesting expanders,
            # so both sit as siblings inside the container.)
            for ex in result["worked_examples"]:
                q = ex.get("question")
                with st.container(border=True):
                    st.markdown(f"**{ex['qid']}** — {ex['what_it_tests']}")
                    if q:
                        meta = " · ".join(
                            filter(None, [q.get("year"), q.get("difficulty_level"), q.get("chapter")])
                        )
                        if meta:
                            st.caption(meta)
                        st.markdown(f"**Question.** {q['question_text']}")
                        if q.get("options"):
                            for i, opt in enumerate(q["options"]):
                                st.markdown(f"- ({chr(97 + i)}) {opt}")

                    # Reveal 1: a nudge to get started, not the full solution.
                    with st.expander("💡 Stuck? Show the approach"):
                        st.markdown(ex["approach"])

                    # Reveal 2: the full step-by-step, plus the verified solution.
                    with st.expander("✅ Show the full step-by-step walkthrough"):
                        for i, step in enumerate(ex["steps"], 1):
                            st.markdown(f"**{i}.** {step}")
                        if ex.get("trap"):
                            st.warning(f"**Watch out:** {ex['trap']}")
                        if q and q.get("correct_answer"):
                            st.markdown(f"**Answer:** {q['correct_answer']}")
                        if q and q.get("solution_text"):
                            st.divider()
                            st.caption("Official verified solution (ground truth):")
                            st.markdown(q["solution_text"])
