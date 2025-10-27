import faiss
import numpy as np
from typing import List, Tuple
from pathlib import Path
from sqlalchemy.orm import Session
from openai import OpenAI

from app.models.database import Document
from app.services.embedding_service import EmbeddingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryService:
    """Service for querying the knowledge base."""
    
    @staticmethod
    def load_user_documents(db: Session, user_id: int) -> List[Document]:
        """
        Load all documents for a user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            List of user documents
        """
        documents = db.query(Document).filter(Document.user_id == user_id).all()
        return documents
    
    @staticmethod
    def load_chunks_from_file(chunks_path: str) -> List[str]:
        """
        Load text chunks from file.
        
        Args:
            chunks_path: Path to chunks file
            
        Returns:
            List of text chunks
        """
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = content.split("\n---CHUNK---\n")
            chunks = [c.strip() for c in chunks if c.strip()]
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks from {chunks_path}: {e}")
            return []
    
    @staticmethod
    def search_similar_chunks(
        query_embedding: List[float],
        index_path: str,
        chunks_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            index_path: Path to FAISS index
            chunks_path: Path to chunks file
            top_k: Number of results to return
            
        Returns:
            List of (chunk_text, distance) tuples
        """
        try:
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load chunks
            chunks = QueryService.load_chunks_from_file(chunks_path)
            
            if not chunks:
                return []
            
            # Search
            query_vector = np.array([query_embedding]).astype('float32')
            distances, indices = index.search(query_vector, min(top_k, len(chunks)))
            
            # Get results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    results.append((chunks[idx], float(dist)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in {index_path}: {e}")
            return []
    
    @staticmethod
    def generate_answer(question: str, context_chunks: List[str]) -> str:
        """
        Generate answer using GPT-4 based on context.
        
        Args:
            question: User question
            context_chunks: Relevant context chunks
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer that question."
Always base your answers strictly on the provided context."""
            
            user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
            
            # Use modern OpenAI client
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    @staticmethod
    def query_knowledge_base(
        db: Session, 
        user_id: int, 
        question: str, 
        top_k: int = 3
    ) -> Tuple[str, List[dict], List[str]]:
        """
        Query the knowledge base and generate answer.

        Args:
            db: Database session
            user_id: User ID
            question: User question
            top_k: Number of context chunks to retrieve

        Returns:
            Tuple of (answer, sources, context_chunks)
        """
        # Load user documents
        documents = QueryService.load_user_documents(db, user_id)
        if not documents:
            return "You haven't uploaded any documents yet.", [], []

        # Generate query embedding
        query_embedding = EmbeddingService.generate_embedding(question)

        # Search across all documents
        all_results = []
        for doc in documents:
            if not doc.faiss_index_path:
                continue

            index_path = doc.faiss_index_path
            chunks_path = index_path.replace(".index", "_chunks.txt")

            results = QueryService.search_similar_chunks(
                query_embedding,
                index_path,
                chunks_path,
                top_k
            )

            for chunk, distance in results:
                all_results.append((chunk, distance, doc.original_filename))

        if not all_results:
            return "No relevant information found in your documents.", [], []

        # Sort by distance and get top results
        all_results.sort(key=lambda x: x[1])
        top_results = all_results[:top_k]

        # Extract chunks and sources
        context_chunks = [r[0] for r in top_results]
        sources = [
            {
                "document": r[2],
                "chunk": r[0],
                "distance": r[1]
            }
            for r in top_results
        ]

        # Generate answer using context
        answer = QueryService.generate_answer(question, context_chunks)

        return answer, sources, context_chunks
