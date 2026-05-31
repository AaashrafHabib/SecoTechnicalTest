"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the RAG pipeline."""
    aws_region: str = "eu-west-1"
    bedrock_model_id: str = "eu.anthropic.claude-sonnet-4-6"
    bedrock_embeddings_model_id: str = "amazon.titan-embed-text-v2:0"
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "eu-construction-standards"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 5
    retriever_fetch_k: int = 20
    rerank_model_id: str = "cohere.rerank-v3-5:0"
    rerank_region: str = "us-east-1"
    data_raw_dir: str = "data/raw"

    model_config = {"env_file": ".env", "env_prefix": "SECO_", "extra": "ignore"}


settings = Settings()
