import faiss
import numpy as np
import re
import base64
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from openai import OpenAI

from app.models.database import Document, DocumentMedia, MediaType
from app.services.embedding_service import EmbeddingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryService:
    """Enhanced service for querying knowledge base with media support"""
    
    @staticmethod
    def load_user_documents(db: Session, user_id: int) -> List[Document]:
        """Load all documents for a user."""
        documents = db.query(Document).filter(Document.user_id == user_id).all()
        return documents
    
    @staticmethod
    def clean_retrieved_chunk(chunk: str) -> str:
        """Clean retrieved chunk text for better presentation."""
        chunk = re.sub(r'\n\s*\n+', ' ', chunk)
        chunk = re.sub(r'\n\s+', ' ', chunk)
        chunk = re.sub(r' +', ' ', chunk)
        return chunk.strip()
    
    @staticmethod
    def load_chunks_from_file(chunks_path: str) -> List[str]:
        """Load text chunks from file."""
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = content.split("\n---CHUNK---\n")
            chunks = [c.strip() for c in chunks if c.strip()]
            chunks = [QueryService.clean_retrieved_chunk(c) for c in chunks]
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []
    
    @staticmethod
    def search_similar_chunks(
        query_embedding: List[float],
        index_path: str,
        chunks_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Search for similar chunks using FAISS."""
        try:
            index = faiss.read_index(index_path)
            chunks = QueryService.load_chunks_from_file(chunks_path)
            
            if not chunks:
                return []
            
            query_vector = np.array([query_embedding]).astype('float32')
            distances, indices = index.search(query_vector, min(top_k, len(chunks)))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    results.append((chunks[idx], float(dist)))
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []
    
    @staticmethod
    def retrieve_related_media(
        db: Session,
        document_id: int,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve images and tables related to a query.
        
        Args:
            db: Database session
            document_id: Document ID
            query: User query
            top_k: Number of media items to return
        
        Returns:
            List of media metadata with base64 encoded data
        """
        try:
            media_items = db.query(DocumentMedia).filter(
                DocumentMedia.document_id == document_id
            ).all()
            
            if not media_items:
                return []
            
            # Score media relevance to query using simple keyword matching
            # Can be enhanced with embeddings
            scored_media = []
            query_lower = query.lower()
            
            for media in media_items:
                # Get description/content for scoring
                content = (media.description or "") + " " + (media.associated_text or "")
                content_lower = content.lower()
                
                # Calculate relevance score
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                
                overlap = len(query_words & content_words)
                score = overlap / len(query_words) if query_words else 0.0
                
                if score > 0.1:  # Threshold for relevance
                    scored_media.append((media, score))
            
            # Sort by score and get top results
            scored_media.sort(key=lambda x: x[1], reverse=True)
            top_media = scored_media[:top_k]
            
            # Format results
            results = []
            for media, score in top_media:
                if media.media_type == MediaType.IMAGE:
                    result = {
                        "id": media.id,
                        "type": "image",
                        "page_number": media.page_number,
                        "description": media.description,
                        "image_base64": base64.b64encode(media.image_data).decode('utf-8') if media.image_data else "",
                        "image_format": media.image_format,
                        "relevance_score": float(score)
                    }
                
                elif media.media_type == MediaType.TABLE:
                    result = {
                        "id": media.id,
                        "type": "table",
                        "page_number": media.page_number,
                        "description": media.description,
                        "table_csv": media.table_data,
                        "table_html": media.table_html,
                        "relevance_score": float(score)
                    }
                
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving related media: {e}")
            return []
    
    @staticmethod
    def generate_answer(question: str, context_chunks: List[str]) -> str:
        """Generate answer using GPT-4 based on context, with improved fallback."""
        try:
            context = "\n\n".join(context_chunks)

            system_prompt = (
                "You are a helpful AI assistant that answers questions based only on the provided context.\n"
                "If you cannot find a direct answer in the context, provide a brief summary or highlight related details from the context.\n"
                "If nothing is relevant, reply 'No relevant information found in the uploaded documents.'\n"
                "Never invent facts beyond the context. Always cite phrases from the context when possible."
            )

            user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )

            answer = response.choices[0].message.content.strip()

            # Guardrail: If the answer is too generic, append relevant context
            generic_responses = [
                "I don't have enough information to answer that question.",
                "No relevant information found in the uploaded documents.",
                "No answer found."
            ]
            if answer in generic_responses:
                fallback = (
                    "The document does not provide a direct answer to your question. "
                    "Here are the most relevant sections:\n\n"
                )
                for chunk in context_chunks:
                    fallback += f"- {chunk}\n"
                answer = fallback

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
        top_k: int = 3,
        include_media: bool = True
    ) -> Tuple[str, List[dict], List[str], List[dict]]:
        """
        Query the knowledge base and generate answer with media.
        
        Args:
            db: Database session
            user_id: User ID
            question: User question
            top_k: Number of context chunks to retrieve
            include_media: Whether to include related media
        
        Returns:
            Tuple of (answer, text_sources, context_chunks, media_items)
        """
        try:
            # Load user documents
            documents = QueryService.load_user_documents(db, user_id)
            
            if not documents:
                return "You haven't uploaded any documents yet.", [], [], []
            
            # Generate query embedding
            query_embedding = EmbeddingService.generate_embedding(question)
            
            # Search across all documents
            all_results = []
            document_map = {}  # Map document ID to document object
            
            for doc in documents:
                if not doc.faiss_index_path:
                    continue
                
                document_map[doc.id] = doc
                
                index_path = doc.faiss_index_path
                chunks_path = index_path.replace(".index", "_chunks.txt")
                
                results = QueryService.search_similar_chunks(
                    query_embedding,
                    index_path,
                    chunks_path,
                    top_k
                )
                
                for chunk, distance in results:
                    all_results.append((chunk, distance, doc.id, doc.original_filename))
            
            if not all_results:
                return "No relevant information found in your documents.", [], [], []
            
            # Sort by distance and get top results
            all_results.sort(key=lambda x: x[1])
            top_results = all_results[:top_k]
            
            # Extract chunks and sources
            context_chunks = [r[0] for r in top_results]
            
            sources = [
                {
                    "document": r[3],
                    "chunk": r[0],
                    "distance": r[1]
                }
                for r in top_results
            ]
            
            # Generate text answer
            answer = QueryService.generate_answer(question, context_chunks)
            
            # Retrieve related media
            media_items = []
            if include_media:
                for doc_id in set([r[2] for r in top_results]):
                    doc_media = QueryService.retrieve_related_media(
                        db, doc_id, question, top_k=2
                    )
                    media_items.extend(doc_media)
            
            return answer, sources, context_chunks, media_items
        
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}", exc_info=True)
            raise
    
    @staticmethod
    def get_media_for_document(db: Session, document_id: int) -> list:
        """Retrieve media items (images, tables) for a document."""
        media_items = []
        
        media_records = db.query(DocumentMedia).filter(
            DocumentMedia.document_id == document_id
        ).all()
        
        for media in media_records:
            media_obj = {
                "id": media.id,
                "document_id": media.document_id,
                "media_type": media.media_type,
                "page_number": media.page_number,
                "description": media.description or "",
            }
            
            if media.media_type == MediaType.IMAGE:
                # Convert binary image to base64
                if media.image_data:
                    media_obj["image_base64"] = base64.b64encode(media.image_data).decode('utf-8')
                    media_obj["image_format"] = media.image_format or "png"
            
            elif media.media_type == MediaType.TABLE:
                # Include table as HTML and CSV
                media_obj["table_html"] = media.table_html or ""
                media_obj["table_csv"] = media.table_data or ""
            
            media_items.append(media_obj)
        
        return media_items