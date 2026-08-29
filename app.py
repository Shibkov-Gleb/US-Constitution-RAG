from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).parent
DOCUMENT_PATH = BASE_DIR / "data" / "us_constitution.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "us_constitution"

# Start with this value. Raise it if unrelated questions receive answers; lower it
# if valid Constitution questions are rejected too often.
MIN_RELEVANCE_SCORE = 0.45
TOP_K = 3


def create_index() -> VectorStoreIndex:
    """Open the persistent Chroma collection as a LlamaIndex vector index."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def has_built_index() -> bool:
    """Return True only when a collection with saved Constitution chunks exists."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        return False
    return collection.count() > 0


def build_index() -> None:
    """Split the Constitution, create embeddings, and save them in Chroma."""
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"Missing source document: {DOCUMENT_PATH}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Rebuilding starts cleanly, so changed source text cannot leave stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass  # The collection has not been created yet.

    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    document = Document(text=DOCUMENT_PATH.read_text(encoding="utf-8"))
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=80)
    nodes = splitter.get_nodes_from_documents([document])
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"Indexed {len(nodes)} chunks in {CHROMA_PATH.name}/")


def retrieve_context(index: VectorStoreIndex, question: str):
    """RAG step 1: retrieve only the most relevant Constitution chunks."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(question)
    return [node for node in nodes if node.score is not None and node.score >= MIN_RELEVANCE_SCORE]


def answer_question(question: str, source_nodes) -> str:
    """RAG step 2: answer using the retrieved text, or decline."""
    if not source_nodes:
        return "I can't answer that from the U.S. Constitution document."

    context = "\n\n---\n\n".join(
        f"Constitution excerpt {number}:\n{node.node.get_content()}"
        for number, node in enumerate(source_nodes, start=1)
    )
    prompt = f"""You answer questions using only the U.S. Constitution excerpts below.

Rules:
- Answer only if the excerpts directly support the answer.
- If the excerpts do not contain enough information, say exactly:
  I can't answer that from the U.S. Constitution document.
- Do not use outside knowledge, legal interpretations, or current events.
- Keep the answer short and clear. Mention an article or amendment only when it appears in an excerpt.

Excerpts:
{context}

Question: {question}
Answer:"""
    llm = OpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), temperature=0)
    return str(llm.complete(prompt)).strip()


def print_sources(source_nodes) -> None:
    print("\nRetrieved excerpts:")
    for number, node in enumerate(source_nodes, start=1):
        preview = node.node.get_content().replace("\n", " ")
        print(f"  [{number}] score={node.score:.2f}  {preview[:180]}...")