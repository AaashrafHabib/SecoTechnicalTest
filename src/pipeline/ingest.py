"""PDF ingestion pipeline: extract, chunk, and store in Chroma."""

import os
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.pipeline.vectorstore import get_vectorstore


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, str]]:
    """Extract text content from each page of a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append(
                    {
                        "text": text,
                        "source": os.path.basename(pdf_path),
                        "page": str(page_num),
                    }
                )
    finally:
        doc.close()
    return pages


def chunk_documents(
    pages: list[dict[str, str]],
) -> list[tuple[str, dict[str, str]]]:
    """Split extracted pages into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            metadata = {"source": page["source"], "page": page["page"]}
            chunks.append((split, metadata))
    return chunks


def ingest_pdfs(data_dir: str | None = None) -> int:
    """Ingest all PDFs from a directory into the Chroma vector store."""
    data_dir = data_dir or settings.data_raw_dir
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return 0

    all_chunks: list[tuple[str, dict[str, str]]] = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    vectorstore = get_vectorstore()
    texts = [chunk[0] for chunk in all_chunks]
    metadatas = [chunk[1] for chunk in all_chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print(f"Ingested {len(texts)} chunks into Chroma")
    return len(texts)


if __name__ == "__main__":
    ingest_pdfs()
