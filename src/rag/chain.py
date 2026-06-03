"""RAG generation chain: context formatting + streaming LLM response."""

import logging
from collections.abc import AsyncGenerator

import boto3
from langchain_core.documents import Document

from src.config import settings
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# Module-level boto3 client for LLM generation (reused across requests)
_bedrock_client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

SYSTEM_PROMPT = """\
<role>
You are a Building Intelligence assistant for construction inspectors at SECO Luxembourg.
Your role is to help inspectors find relevant information from EU construction standards (Eurocodes)
regarding defects, compliance requirements, and remediation guidance.
</role>

<rules>
- Answer based ONLY on the provided context from EU construction standards.
- If the context doesn't contain enough information, say so clearly — never fabricate information.
- Always cite the source document and page number for each claim using [Source N] format.
- Use technical language appropriate for construction professionals.
- Structure your answers with clear headings when the answer covers multiple topics.
- When discussing defects, mention relevant standard clauses and acceptance criteria.
</rules>

<output_format>
- Use markdown headings (##) to organize multi-part answers.
- Cite every factual claim with [Source N] referencing the provided context.
- End with a brief summary if the answer exceeds 3 paragraphs.
</output_format>
"""


def format_context(documents: list["Document"]) -> str:
    """
    Format retrieved documents into a numbered, citable context string.

    Each document is labeled with a source index, filename, and page number
    so the LLM can cite them using [Source N] format in its response.

    Args:
        documents (list[Document]): LangChain Documents with 'source' and
            'page' metadata fields.

    Returns:
        str: Formatted context string with documents separated by '---'
            delimiters, each prefixed with [Source N: filename, Page P].
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}\n"
        )
    return "\n---\n".join(context_parts)


async def generate_response(
    query: str, documents: list[Document] | None = None
) -> AsyncGenerator[str, None]:
    """
    Stream a RAG-augmented response from Bedrock Claude.

    End-to-end pipeline:
        1. Retrieves and reranks relevant document chunks (or uses provided documents).
        2. Formats context with source citations into structured XML prompt.
        3. Streams response tokens from Claude via Converse Stream API.

    Args:
        query (str): Inspector's natural language question about EU
            construction standards.
        documents (list[Document] | None): Pre-retrieved documents to use.
            If None, retrieval is performed automatically.

    Yields:
        str: Text tokens as they are streamed from the LLM.
    """
    # Retrieve documents if not provided
    if documents is None:
        documents = retrieve(query)

    # Guard against empty context - don't invoke LLM
    if not documents:
        yield "I don't have any relevant information in my knowledge base to answer this question. "
        yield "Please ensure the document ingestion has completed successfully, or try rephrasing your question."
        return

    context = format_context(documents)

    user_message = f"""\
<context>
{context}
</context>

<user_question>
{query}
</user_question>

<instructions>
Answer the inspector's question using ONLY the information in <context>.
Cite sources with [Source N] for every claim. If the context is insufficient, state what is missing.
</instructions>"""

    try:
        response = _bedrock_client.converse_stream(
            modelId=settings.bedrock_model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 2048},
        )

        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"]["delta"].get("text", "")
                    if text:
                        yield text
    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        yield f"\n\n⚠️ Error generating response: {str(e)}"


def get_sources_display(documents: list[Document]) -> list[dict[str, str]]:
    """
    Get source metadata for display in the Chainlit UI sidebar.

    Formats retrieved documents as preview cards showing the source
    filename, page number, and a 500-char content snippet.

    Args:
        documents (list[Document]): Pre-retrieved documents to format.

    Returns:
        list[dict[str, str]]: List of source dicts with keys:
            - name: Display label formatted as "filename (p.N)".
            - content: First 500 characters of the chunk text.
    """
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        sources.append(
            {
                "name": f"{source} (p.{page})",
                "content": doc.page_content[:500],
            }
        )
    return sources
