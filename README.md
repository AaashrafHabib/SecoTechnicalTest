# Building Intelligence: Inspector Defect Search

A RAG-powered chatbot that helps construction inspectors quickly find relevant EU construction standard clauses (Eurocodes), defect criteria, and remediation guidance.

## What problem are you solving, and for whom?

**User:** Construction inspectors at SECO Luxembourg

**Problem:** Inspectors spend significant time manually searching through hundreds of pages of Eurocodes and EU standards during site inspections. When they encounter a potential defect (cracking in concrete, corrosion on steel, foundation settlement), they need to quickly identify the relevant acceptance criteria, tolerance limits, and remediation requirements from the applicable standard.

**Solution:** A conversational AI interface where inspectors type natural-language questions and receive cited answers directly from EU construction standards, with source references for traceability.

## Why is this relevant to SECO?

SECO's core business is technical control and risk prevention in construction. Their inspectors assess compliance against standards daily. This tool:
- Reduces time spent searching through documentation during inspections
- Ensures inspectors reference the correct standard clauses
- Provides traceability through source citations (important for official reports)
- Could scale to include SECO's own internal inspection templates and historical findings

## Data sources

- **EU Construction Standards (Eurocodes):** EN 1990 (Basis of Design), EN 1991 (Actions), EN 1992 (Concrete), EN 1993 (Steel), EN 1997 (Geotechnical)
- Generated as structured PDFs from publicly available Eurocode summaries and guidance documents (real Eurocodes are copyrighted by CEN, so we use public domain summaries)
- The pipeline is designed to ingest any PDF — in production, SECO would add their licensed standard documents

## Technical decisions and trade-offs

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **Chainlit** for UI | Real-time streaming, chat-native, Python-native (matches our stack). Production-ready with auth support. | Not React (SECO's main stack), but justified by speed-to-MVP and Chainlit's maturity for AI chat apps |
| **AWS Bedrock (Claude)** | Enterprise-grade, no API key management in prod, IAM-native, data stays in AWS | Vendor lock-in to AWS, but SECO likely already uses AWS |
| **Amazon Titan Embeddings** | Fully AWS-native, no external API calls for embeddings, good multilingual support (FR/DE/EN) | Less community benchmarks than OpenAI embeddings, but sufficient for technical documents |
| **ChromaDB** | Zero-ops for MVP, file-based persistence, good LangChain integration | Won't scale to millions of documents — would migrate to OpenSearch/pgvector in production |
| **PyMuPDF** | Fast PDF extraction, handles complex layouts, good for technical documents | Doesn't handle scanned PDFs — would add AWS Textract for OCR in production |
| **UV** | Fast, reproducible builds, lockfile for exact dependency pinning | Newer tool, less community knowledge |

## What would you put in production tomorrow vs. what would you throw away?

**Keep for production:**
- RAG architecture (retrieve → augment → generate)
- Chainlit as the UI layer (it supports auth, multi-tenancy, conversation history)
- AWS Bedrock integration (Claude for generation, Titan for embeddings)
- Pre-commit hooks and linting setup
- Docker/ECS deployment pattern

**Throw away / replace:**
- ChromaDB → Amazon OpenSearch or PostgreSQL + pgvector (scalability, managed service)
- Sample PDFs → real licensed Eurocode documents + SECO internal standards
- Simple similarity search → hybrid search (keyword + semantic) with reranking
- Single-user setup → add Chainlit auth + per-user conversation history

## If you had 3 more months, what would the product look like?

1. **Multi-modal ingestion:** AWS Textract for scanned documents, support for architectural plans (image understanding)
2. **Multilingual support:** French/German/English (Luxembourg's working languages), with language detection and cross-language retrieval
3. **Inspection workflow integration:** Generate pre-filled inspection checklists based on building type and applicable standards
4. **Historical defect database:** RAG over SECO's past inspection reports to find similar defects and their resolutions
5. **Mobile-first interface:** Inspectors work on-site — optimize for tablet/phone with offline capability
6. **Compliance scoring:** Automated assessment of whether a reported defect exceeds standard tolerances
7. **Report generation:** Auto-generate the relevant standards section for inspection reports

## Setup

### Prerequisites
- Python 3.10-3.12
- [UV](https://docs.astral.sh/uv/) package manager
- AWS account with Bedrock access (Claude 3 Sonnet + Titan Embeddings enabled)

### Installation

```bash
# Clone and install
git clone https://github.com/AaashrafHabib/Seco-technical-test.git
cd Seco-technical-test
uv sync

# Configure AWS credentials
cp .env.example .env
# Edit .env with your AWS credentials

# Generate sample data
uv run python scripts/download_standards.py

# Ingest documents into vector store
uv run python -m src.pipeline.ingest

# Run the app
uv run chainlit run src/app.py --port 8000
```

### Docker

```bash
docker compose up --build
```

### Deploy to AWS ECS

See `infra/ecs-task-definition.json` for the Fargate task definition template.

## Project Structure

```
src/
├── config.py              # Settings (env vars, model IDs)
├── pipeline/
│   ├── ingest.py          # PDF loading, chunking, embedding
│   ├── embeddings.py      # Bedrock Titan embeddings
│   └── vectorstore.py     # ChromaDB operations
├── rag/
│   ├── retriever.py       # Similarity search
│   └── chain.py           # RAG chain (retrieve → prompt → generate)
└── app.py                 # Chainlit entry point

scripts/
├── download_standards.py  # Generate sample EU standard PDFs
└── lint.py                # Linting script (ruff + mypy)

infra/
└── ecs-task-definition.json  # AWS ECS Fargate config
```

## Tech Stack

- **LLM:** Claude 3 Sonnet via AWS Bedrock
- **Embeddings:** Amazon Titan Embed Text v2
- **Vector Store:** ChromaDB (persistent, file-based)
- **Document Processing:** PyMuPDF + LangChain text splitters
- **UI:** Chainlit
- **Deployment:** Docker → AWS ECS (Fargate)
- **Package Manager:** UV
- **Linting:** Ruff + Mypy (pre-commit hooks)
