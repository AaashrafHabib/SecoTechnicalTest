"""PDF ingestion pipeline: extract, chunk, and store in Chroma."""

import os
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.pipeline.vectorstore import get_vectorstore


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, str]]:
    """
    Extract text content from each page of a PDF file.

    Opens the PDF with PyMuPDF (fitz) and extracts text from each page.
    Empty pages are skipped.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.

    Returns:
        list[dict[str, str]]: List of page dicts with keys:
            - text: Extracted page content.
            - source: PDF filename (basename only).
            - page: 1-indexed page number as string.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
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
    return pages


def chunk_documents(
    pages: list[dict[str, str]],
) -> list[tuple[str, dict[str, str]]]:
    """
    Split extracted pages into overlapping text chunks.

    Uses RecursiveCharacterTextSplitter with configurable chunk size and
    overlap from settings. Each chunk retains the source and page metadata
    of its origin page.

    Args:
        pages (list[dict[str, str]]): Page dicts from extract_text_from_pdf().

    Returns:
        list[tuple[str, dict[str, str]]]: List of (chunk_text, metadata) tuples
            ready for vector store insertion.
    """
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
    """
    Ingest all PDFs from a directory into the Chroma vector store.

    Scans the target directory for .pdf files, extracts text page-by-page,
    splits into overlapping chunks, and bulk-inserts into Chroma with
    source metadata and vector embeddings.

    Args:
        data_dir (str | None): Directory containing PDF files.
            Defaults to settings.data_raw_dir if not provided.

    Returns:
        int: Total number of chunks ingested into the vector store.
    """
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
