"""
Streamlit RAG — Select documents to query (fallback: all docs)
- Upload/manage PDFs, TXT, MD
- Vector store: Chroma (persisted locally)
- Embeddings: OpenAI (text-embedding-3-small)
- LLM: via `init_chat_model` (provider-agnostic name; e.g., gpt-4o-mini)
- Streaming answers + concise source citations
"""

from __future__ import annotations
import json
from typing import List, Dict

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from rag_store import RAGStore, format_docs_for_context, DocMeta

APP_TITLE = "Streamlit RAG — Select Docs or Search All"
# DEFAULT_SYSTEM_PROMPT = (
#     "You are a concise assistant. Use ONLY the provided context. "
#     "If the answer is not contained in the context, respond exactly with: I don't know. "
#     "Cite sources inline as [S1], [S2], ... corresponding to the provided snippets."
# )
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant for a document QA app. Use ONLY the provided context for factual claims. "
    "Do not invent information. "
    "\n\nBehavior:"
    "\n• Small talk: If the user greets, thanks, or makes casual chit-chat, reply briefly and warmly; do not include citations; do not say 'I don't know'."
    "\n• Answering: If the answer is contained in the context, respond in 1–3 sentences and cite sources inline as [S1], [S2], ... matching the snippets used."
    "\n• Out-of-scope: If the user asks for information not contained in the context, respond exactly with: I don't know."
    "\n• Clarify: If the request is ambiguous, ask one concise clarifying question."
    "\n• Style: Be direct, avoid filler, no disclaimers, no moralizing, no mention of being an AI."
)
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 5120

# -----------------------------
# Session State (Chat Memory)
# -----------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history: List[BaseMessage] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids: List[str] = []

def add_user_message(content: str): st.session_state.history.append(HumanMessage(content=content))
def add_ai_message(content: str): st.session_state.history.append(AIMessage(content=content))
def clear_history(): st.session_state.history = []

# -----------------------------
# Prompt / Chain
# -----------------------------
def build_chain(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder("history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ]
    )
    return prompt | model | StrOutputParser()

# -----------------------------
# Sidebar (Docs & Settings)
# -----------------------------
def sidebar(store: RAGStore):
    with st.sidebar:
        st.header("📁 Documents")

        # Upload
        uploaded = st.file_uploader(
            "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
        if uploaded:
            with st.status("Indexing…", expanded=True):
                for uf in uploaded:
                    doc_id, meta = store.index_streamlit_upload(uf)
                    st.write(f"Indexed: {meta['name']}  ({doc_id})")

        # Registry + selection
        docs: List[DocMeta] = store.list_documents()
        by_label: Dict[str, str] = {
            f"{d['name']}  •  {d['doc_id']}": d["doc_id"] for d in docs
        }
        selection = st.multiselect(
            "Restrict retrieval to selected docs (leave empty to use all):",
            options=list(by_label.keys()),
            default=[k for k in by_label if by_label[k] in st.session_state.selected_doc_ids],
            max_selections=None,
        )
        st.session_state.selected_doc_ids = [by_label[k] for k in selection]

        # Delete
        options = ["—"] + list(by_label.keys())   # the list of names + "—"
        if docs:
            # del_choice = st.selectbox("Delete a document", ["—"] + [d["doc_id"] for d in docs])
            del_choice = st.selectbox("Delete a document", options)
            if st.button("🗑️ Delete", use_container_width=True) and del_choice != "—":
                store.delete_document(by_label[del_choice])
                st.rerun()

        st.divider()
        st.header("⚙️ Settings")

        st.session_state.system_prompt = st.text_area(
            "System Prompt", value=st.session_state.system_prompt, height=120
        )
        st.session_state.model_name = st.text_input("Model Name", value=st.session_state.model_name)
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05)
        st.session_state.max_tokens = st.number_input("Max Tokens", 500, 120000, int(st.session_state.max_tokens), 100)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Clear chat", use_container_width=True):
                clear_history(); st.rerun()
        with c2:
            transcript = [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in st.session_state.history
            ]
            st.download_button(
                "⬇️ Export chat (.json)",
                data=json.dumps(transcript, ensure_ascii=False, indent=2),
                file_name="chat_transcript.json",
                mime="application/json",
                use_container_width=True,
            )

# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.title("📚 RAG — Ask Your Documents")
    st.caption("Select docs to target; no selection means search across all.")

    init_session_state()
    store = RAGStore()  # local ./data + ./chroma + registry.json
    sidebar(store)

    # Render history
    for msg in st.session_state.history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

    # Guard: show provider hint for OpenAI models
    if st.session_state.model_name.startswith(("gpt-", "o")):
        if "OPENAI_API_KEY" not in st.secrets and not st.session_state.get("_openai_env_hint_shown", False):
            st.info("Set OPENAI_API_KEY in env or `.streamlit/secrets.toml` to use OpenAI models.", icon="🔑")
            st.session_state["_openai_env_hint_shown"] = True

    # Chat input
    question = st.chat_input("Ask a question…")
    if not question:
        return

    # Persist user message
    add_user_message(question)
    with st.chat_message("user"): st.markdown(question)

    # Build model/chain
    try:
        model = init_chat_model(
            model=st.session_state.model_name,
            model_provider="openai",
            temperature=float(st.session_state.temperature) if st.session_state.model_name.startswith(("gpt-",)) else 1,
            max_tokens=int(st.session_state.max_tokens),
            streaming=True,
        )
        chain = build_chain(model)

        # Retrieve
        retriever = store.get_retriever(selected_doc_ids=st.session_state.selected_doc_ids, k=6)
        docs = retriever.invoke(question) #retriever.get_relevant_documents(question) # 
        ctx_text, numbered_sources = format_docs_for_context(docs)  # context string + [{sid, name, page, score, doc_id}]

        # Stream answer
        with st.chat_message("assistant"):
            chunks = chain.stream({
                "system_prompt": st.session_state.system_prompt,
                "history": st.session_state.history[:-1],  # exclude the just-added user msg
                "context": ctx_text,
                "question": question,
            })
            full = st.write_stream(chunks)

            # Show sources
            with st.expander("Sources used"):
                if not numbered_sources:
                    st.write("No matching passages retrieved.")
                else:
                    for s in numbered_sources:
                        # print(f"{s=}")
                        # st.markdown(f"- **[{s['sid']}]** {s['name']} (p. {s['page']}) — `doc_id={s['doc_id']}`  • score={s['score']:.3f}")
                        st.markdown(f"- **[{s['sid']}]** {s['name']} (p. {s['page']}) — `doc_id={s['doc_id']}`")

        add_ai_message(full)

    except Exception as e:
        st.error(f"RAG failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
