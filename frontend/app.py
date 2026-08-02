"""Phase 5.2: Streamlit query dashboard. Talks to the FastAPI service.

Run: streamlit run frontend/app.py
(Assumes the API is running at API_BASE, default http://localhost:8000.)
"""
import os

import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Internal Docs RAG", layout="wide")
st.title("Internal Docs — Hybrid Search RAG")

with st.sidebar:
    st.header("Settings")
    use_reranker = st.checkbox("Use reranker", value=True)
    compare_mode = st.checkbox("Compare hybrid vs. dense-only", value=False)
    st.caption(f"API: {API_BASE}")
    if st.button("List indexed documents"):
        try:
            docs = requests.get(f"{API_BASE}/v1/documents", timeout=10).json()
            st.json(docs)
        except Exception as e:
            st.error(f"Could not reach API: {e}")

question = st.text_input("Ask a question about the indexed docs", "")
ask_clicked = st.button("Ask", type="primary")


def render_result(label: str, result: dict):
    st.subheader(label)
    if result.get("is_fallback"):
        st.warning(result["answer"])
        st.caption(result.get("what_was_found", ""))
        return

    st.write(result["answer"])

    conf = result.get("confidence", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite confidence", conf.get("composite_confidence", "-"))
    c2.metric("Retrieval confidence", conf.get("retrieval_confidence", "-"))
    c3.metric("Citation coverage", conf.get("citation_coverage", "-"))
    c4.metric("Completeness", conf.get("answer_completeness", "-"))

    if conf.get("below_threshold"):
        st.error("Confidence below threshold — treat this answer with caution.")

    with st.expander("Retrieved chunks (final, after fusion + rerank)"):
        for i, c in enumerate(result["retrieval"]["final_chunks"], start=1):
            meta = c.get("metadata", {})
            st.markdown(f"**[{i}] {meta.get('source_file')}** — {meta.get('section_heading') or ''}")
            st.text(c["text"][:400])

    with st.expander("Citation verification report"):
        st.json(result.get("citation_report", {}))


if ask_clicked and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            if compare_mode:
                col1, col2 = st.columns(2)
                hybrid = requests.post(f"{API_BASE}/v1/ask", json={
                    "question": question, "use_reranker": use_reranker, "apply_hybrid": True,
                }, timeout=60).json()
                dense = requests.post(f"{API_BASE}/v1/ask", json={
                    "question": question, "use_reranker": use_reranker, "apply_hybrid": False,
                }, timeout=60).json()
                with col1:
                    render_result("Hybrid (dense + sparse)", hybrid)
                with col2:
                    render_result("Dense-only", dense)
            else:
                result = requests.post(f"{API_BASE}/v1/ask", json={
                    "question": question, "use_reranker": use_reranker, "apply_hybrid": True,
                }, timeout=60).json()
                render_result("Answer", result)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API at {API_BASE}: {e}")
