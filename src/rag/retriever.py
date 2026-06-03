"""Retrieval module: vector search + Cohere reranking."""

import json
import logging

import boto3
from langchain_core.documents import Document

from src.config import settings
from src.pipeline.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Module-level boto3 client for reranking (reused across requests)
_bedrock_rerank_client = boto3.client(
    "bedrock-runtime", region_name=settings.rerank_region
)


def _rerank(query: str, documents: list[Document], top_k: int) -> list[Document]:
    """
    Rerank candidate documents using Cohere Rerank 3.5 on Bedrock.

    Sends the query and all candidate document texts to the Cohere
    cross-encoder model, which scores each document's actual relevance
    to the query (not just semantic similarity).

    Falls back to original document order if reranking fails.

    Args:
        query (str): User's search query.
        documents (list[Document]): Candidate documents from vector
            similarity search.
        top_k (int): Number of top-ranked documents to return.

    Returns:
        list[Document]: Top-k documents reordered by relevance score,
            most relevant first. Returns original order (truncated to top_k)
            if reranking fails.
    """
    if not documents:
        return documents

    try:
        response = _bedrock_rerank_client.invoke_model(
            modelId=settings.rerank_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "api_version": 2,
                    "query": query,
                    "documents": [doc.page_content for doc in documents],
                    "top_n": top_k,
                }
            ),
        )

        result = json.loads(response["body"].read())

        # Validate response structure
        if "results" not in result or not isinstance(result["results"], list):
            logger.warning("Rerank API returned unexpected format, falling back to original order")
            return documents[:top_k]

        # Extract and validate indices
        ranked_indices = []
        for item in result["results"]:
            if "index" in item and 0 <= item["index"] < len(documents):
                ranked_indices.append(item["index"])
            else:
                logger.warning(f"Invalid index in rerank response: {item.get('index')}")

        if not ranked_indices:
            logger.warning("No valid indices in rerank response, falling back to original order")
            return documents[:top_k]

        return [documents[i] for i in ranked_indices]

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to original vector search order.")
        return documents[:top_k]


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """
    Retrieve relevant documents using vector search + cross-encoder reranking.

    Two-stage retrieval pipeline:
        1. Broad vector similarity search fetches fetch_k candidates (fast).
        2. Cohere Rerank cross-encoder re-scores and selects top_k (precise).

    Args:
        query (str): User's natural language question.
        top_k (int | None): Number of final documents to return.
            Defaults to settings.retriever_top_k.

    Returns:
        list[Document]: Top-k most relevant Documents with metadata
            (source filename and page number).
    """
    top_k = top_k or settings.retriever_top_k
    fetch_k = settings.retriever_fetch_k
    vectorstore = get_vectorstore()
    candidates = vectorstore.similarity_search(query, k=fetch_k)
    return _rerank(query, candidates, top_k)
