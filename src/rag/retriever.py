"""Retrieval module: vector similarity search."""

from langchain_core.documents import Document

from src.config import settings
from src.pipeline.vectorstore import get_vectorstore


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Retrieve relevant documents using vector similarity search."""
    top_k = top_k or settings.retriever_top_k
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=top_k)
