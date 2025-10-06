import base64
from pathlib import Path

import streamlit as st

from src.app.settings import get_settings
from src.generation.generator import get_generator
from src.ingestion.indexer import build_indexes
from src.retrieval.hybrid import HybridRetriever

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

settings = get_settings()
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(settings.index_dir, settings.duckdb_path)
if "backend" not in st.session_state:
    st.session_state.backend = settings.generator_backend

st.sidebar.title("Controls")
if st.sidebar.button("Rebuild Index"):
    with st.spinner("Indexing data..."):
        build_indexes(Path("data"), settings.index_dir, settings.duckdb_path)
        st.session_state.retriever = HybridRetriever(settings.index_dir, settings.duckdb_path)
        st.success("Index rebuilt")

st.session_state.backend = st.sidebar.selectbox("Generator backend", ["local", "openai"], index=0)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)

st.title("🔎 Multimodal Retrieval-Augmented Generation")
query = st.text_area("Ask a question about the sample corpus")
image_file = st.file_uploader("Optional image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([2, 1])
with col2:
    if image_file:
        st.image(image_file, caption="Uploaded image", use_column_width=True)

if st.button("Ask") and query:
    retriever: HybridRetriever = st.session_state.retriever
    generator = get_generator(st.session_state.backend)
    image_b64 = None
    if image_file:\n        image_bytes = image_file.getvalue()\n        image_b64 = base64.b64encode(image_bytes).decode('utf-8')\n    with st.spinner("Running hybrid retrieval..."):
        result = retriever.retrieve(query_text=query, top_k=top_k, image_b64=image_b64)
        contexts = result["contexts"]
    with st.spinner("Generating grounded answer..."):
        gen_result = generator.generate(query, contexts)
    st.subheader("Answer")
    st.write(gen_result.answer)
    st.caption(f"Citations: {', '.join(gen_result.citations)}")

    st.subheader("Evidence")
    for ctx in contexts:
        with st.expander(f"{ctx.source_id} ({ctx.modality})"):
            st.write(ctx.content)
            st.json(ctx.metadata)

    with st.expander("Raw Response JSON"):
        st.json(
            {
                "answer": gen_result.answer,
                "citations": gen_result.citations,
                "contexts": [
                    {
                        "source_id": ctx.source_id,
                        "modality": ctx.modality,
                        "score": ctx.score,
                        "metadata": ctx.metadata,
                    }
                    for ctx in contexts
                ],
                "modality_breakdown": result["modality_breakdown"],
            }
        )

st.sidebar.markdown("---")
if st.sidebar.button("Run Evaluation"):
    with st.spinner("Evaluating..."):
        from src.evaluation.run_ragas import run_eval

        metrics = run_eval(Path("docs/eval_report.md"))
        st.success("Evaluation completed")
        st.sidebar.write(
            f"Faithfulness: {metrics.faithfulness:.2f}\n\n"
            f"Answer relevancy: {metrics.answer_relevancy:.2f}\n\n"
            f"Context precision: {metrics.context_precision:.2f}\n\n"
            f"Context recall: {metrics.context_recall:.2f}"
        )
        st.sidebar.markdown("[View full report](docs/eval_report.md)")


