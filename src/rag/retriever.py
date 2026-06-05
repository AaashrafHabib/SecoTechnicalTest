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
    """Rerank candidate documents using Cohere Rerank 3.5 on Bedrock with error handling."""
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
            logger.warning("Rerank API returned unexpected format, falling back")
            return documents[:top_k]

        # Extract and validate indices
        ranked_indices = []
        for item in result["results"]:
            if "index" in item and 0 <= item["index"] < len(documents):
                ranked_indices.append(item["index"])

        if not ranked_indices:
            logger.warning("No valid indices in rerank response")
            return documents[:top_k]

        return [documents[i] for i in ranked_indices]

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to vector search order.")
        return documents[:top_k]


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Retrieve relevant documents using vector search + reranking."""
    top_k = top_k or settings.retriever_top_k
    fetch_k = settings.retriever_fetch_k
    vectorstore = get_vectorstore()
    candidates = vectorstore.similarity_search(query, k=fetch_k)
    return _rerank(query, candidates, top_k)
