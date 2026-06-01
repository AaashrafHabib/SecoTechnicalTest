"""Embedding model factory for AWS Bedrock."""

from langchain_aws import BedrockEmbeddings

from src.config import settings


def get_embeddings() -> BedrockEmbeddings:
    """
    Create a BedrockEmbeddings instance using Amazon Titan Embed v2.

    Initializes the embedding client configured with the model ID and
    AWS region from application settings.

    Returns:
        BedrockEmbeddings: Configured embedding model client ready for
            text-to-vector encoding.
    """
    return BedrockEmbeddings(
        model_id=settings.bedrock_embeddings_model_id,
        region_name=settings.aws_region,
    )
