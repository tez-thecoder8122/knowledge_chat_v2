from openai import OpenAI
from typing import List
import numpy as np
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Service for handling embeddings operations."""

    MODEL = "text-embedding-3-small"

    @staticmethod
    def generate_embedding(text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Args:
            text: Text to embed
        Returns:
            Embedding vector
        """
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.embeddings.create(
                input=text,
                model=EmbeddingService.MODEL
            )
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        Args:
            texts: List of texts to embed
        Returns:
            List of embedding vectors
        """
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.embeddings.create(
                input=texts,
                model=EmbeddingService.MODEL
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Args:
            vec1: First vector
            vec2: Second vector
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)
        return dot_product / (norm_1 * norm_2)
