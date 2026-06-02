"""Chroma vector store initialization."""

from langchain_chroma import Chroma

from src.config import settings
from src.pipeline.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    """
    Get the Chroma vector store client.

    Connects to the configured Chroma persistent directory.
    Uses Bedrock Titan embeddings for vector encoding.

    Returns:
        Chroma: LangChain-wrapped Chroma client ready
            for similarity search and document insertion.
    """
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )
