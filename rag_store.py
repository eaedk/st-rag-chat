from __future__ import annotations
import os, io, json, time, hashlib, shutil
from typing import List, Dict, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

DocMeta = Dict[str, str]

DATA_DIR = "./data"
PERSIST_DIR = "./chroma"
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def load_text_bytes(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def openai_embeddings():
    # Swap here if you want a different embedding provider
    return OpenAIEmbeddings(model="text-embedding-3-small")

def _new_vectorstore():
    return Chroma(
        collection_name="rag_store",
        embedding_function=openai_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def _save_registry(reg: List[DocMeta]):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

def _load_registry() -> List[DocMeta]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def _loader_for(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path)
    elif ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

class RAGStore:
    def __init__(self):
        ensure_dirs()
        self.vs = _new_vectorstore()

    # ---------- Document Registry ----------
    def list_documents(self) -> List[DocMeta]:
        reg = _load_registry()
        # sort by created_at desc
        return sorted(reg, key=lambda m: m.get("created_at", 0), reverse=True)

    def _register_doc(self, meta: DocMeta):
        reg = _load_registry()
        # overwrite if exists
        reg = [m for m in reg if m["doc_id"] != meta["doc_id"]]
        reg.append(meta)
        _save_registry(reg)

    def _unregister_doc(self, doc_id: str):
        reg = _load_registry()
        reg = [m for m in reg if m["doc_id"] != doc_id]
        _save_registry(reg)

    # ---------- Ingestion ----------
    def index_file(self, file_path: str, *, doc_id: str | None = None, name: str | None = None) -> Tuple[str, DocMeta]:
        loader = _loader_for(file_path)
        raw_docs = loader.load()
        # annotate metadata
        for d in raw_docs:
            d.metadata.setdefault("source", file_path)
            d.metadata.setdefault("page", d.metadata.get("page", 1))
        chunks = _split_docs(raw_docs)

        # derivations
        if doc_id is None:
            with open(file_path, "rb") as f:
                doc_id = sha1_bytes(f.read())[:16]
        if name is None:
            name = os.path.basename(file_path)

        # delete old vectors if same doc_id exists
        try:
            self.vs.delete(where={"doc_id": doc_id})
        except Exception:
            pass

        # add to vector store
        metadatas = []
        for c in chunks:
            md = dict(c.metadata)
            md.update({"doc_id": doc_id, "name": name})
            metadatas.append(md)
        self.vs.add_texts([c.page_content for c in chunks], metadatas=metadatas)
        # self.vs.persist()

        meta: DocMeta = {
            "doc_id": doc_id,
            "name": name,
            "path": file_path,
            "num_chunks": str(len(chunks)),
            "created_at": str(int(time.time())),
        }
        self._register_doc(meta)
        return doc_id, meta

    def index_streamlit_upload(self, uploaded_file) -> Tuple[str, DocMeta]:
        # Persist file bytes then index
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in (".pdf", ".txt", ".md"):
            raise ValueError("Only .pdf, .txt, .md are supported.")
        b = uploaded_file.read()
        doc_id = sha1_bytes(b)[:16]
        dst_path = os.path.join(DATA_DIR, f"{doc_id}{ext}")
        with open(dst_path, "wb") as f: f.write(b)
        return self.index_file(dst_path, doc_id=doc_id, name=uploaded_file.name)

    # ---------- Delete ----------
    def delete_document(self, doc_id: str):
        try:
            self.vs.delete(where={"doc_id": doc_id})
            self.vs.persist()
        except Exception:
            pass
        # remove file if present
        reg = _load_registry()
        for m in reg:
            if m["doc_id"] == doc_id and os.path.exists(m["path"]):
                try: os.remove(m["path"])
                except Exception: pass
        self._unregister_doc(doc_id)

    # ---------- Retrieval ----------
    def get_retriever(self, *, selected_doc_ids: List[str] | None, k: int = 4):
        flt = None
        if selected_doc_ids:
            flt = {"doc_id": {"$in": selected_doc_ids}}
        return self.vs.as_retriever(search_kwargs={"k": k, "filter": flt})

# ---------- Helpers ----------
def format_docs_for_context(docs: List[Document]) -> Tuple[str, List[Dict]]:
    """
    Return:
      - context text joining top-k docs, each prefixed with [S{n}]
      - list of {sid, name, page, score, doc_id}
    """
    out_lines, sources = [], []
    for i, d in enumerate(docs):
        sid = f"S{i+1}"
        name = d.metadata.get("name", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", 1)
        score = d.metadata.get("score", 0.0) or d.metadata.get("relevance_score", 0.0) or 0.0
        doc_id = d.metadata.get("doc_id", "?")

        # print(f"{d.metadata=}")
        # print(f"{d=}")
        # Attach sid into metadata for display & instruct model to cite [S#]
        out_lines.append(f"[{sid}] {d.page_content.strip()}\n(Source: {name}, page {page})")
        sources.append({"sid": sid, "name": name, "page": page, "score": float(score), "doc_id": doc_id})
    return "\n\n".join(out_lines), sources
