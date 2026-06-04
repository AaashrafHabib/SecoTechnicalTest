"""Chainlit chat application for EU Construction Standards RAG assistant."""

import chainlit as cl

from src.rag.chain import generate_response, get_sources_display


@cl.on_chat_start
async def start() -> None:
    """
    Send welcome message when a new chat session starts.

    Displays an introduction explaining the assistant's capabilities
    and suggested question topics for the inspector.
    """
    await cl.Message(
        content=(
            "Welcome to **Building Intelligence** — your EU Construction Standards assistant.\n\n"
            "I help construction inspectors quickly find relevant Eurocode clauses, "
            "defect criteria, and compliance requirements.\n\n"
            "**Ask me anything about:**\n"
            "- Structural requirements (concrete, steel, timber, masonry)\n"
            "- Acceptance criteria and tolerances\n"
            "- Defect classification and remediation\n"
            "- Fire resistance and safety requirements\n\n"
            "Type your question below to get started."
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message) -> None:
    """
    Handle incoming user messages.

    Streams the RAG-augmented LLM response token-by-token, then attaches
    source documents as side-panel elements for reference.

    Args:
        message (cl.Message): Incoming Chainlit message containing the
            inspector's question in message.content.
    """
    from src.rag.retriever import retrieve

    msg = cl.Message(content="")
    await msg.send()

    # Retrieve documents once and reuse for both generation and sources display
    documents = retrieve(message.content)

    full_response = ""
    async for token in generate_response(message.content, documents=documents):
        full_response += token
        await msg.stream_token(token)

    await msg.update()

    # Use same documents for sources display (no duplicate retrieval)
    sources = get_sources_display(documents)
    elements = []
    for source in sources:
        elements.append(
            cl.Text(
                name=source["name"],
                content=source["content"],
                display="side",
            )
        )

    msg.elements = elements  # type: ignore[assignment]
    await msg.update()
