import os
import uuid
from pathlib import Path
from typing import List, Tuple
import PyPDF2
import faiss
import numpy as np
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException, status

from app.models.database import Document
from app.services.embedding_service import EmbeddingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentService:
    """Service for handling document operations."""

    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 50  # Overlap between chunks

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate uploaded file type."""
        file_ext = Path(file.filename).suffix.lower()
        allowed_exts = settings.ALLOWED_EXTENSIONS.split(',')

        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {allowed_exts}"
            )

    @staticmethod
    def save_uploaded_file(file: UploadFile, user_id: int) -> Tuple[str, str, int]:
        """Save uploaded file to disk."""
        upload_dir = Path(settings.UPLOAD_DIR) / str(user_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_ext = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = upload_dir / unique_filename

        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)

        file_size = len(content)
        logger.info(f"File saved: {file_path} ({file_size} bytes)")

        return str(file_path), unique_filename, file_size

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            text = ""
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process PDF file"
            )

    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text content from a TXT file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Extracted {len(text)} characters from TXT")
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading TXT file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process TXT file"
            )

    @staticmethod
    def extract_text(file_path: str, file_type: str) -> str:
        """Extract text from file depending on extension."""
        if file_type == ".pdf":
            return DocumentService.extract_text_from_pdf(file_path)
        elif file_type == ".txt":
            return DocumentService.extract_text_from_txt(file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type"
            )

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + DocumentService.CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += DocumentService.CHUNK_SIZE - DocumentService.CHUNK_OVERLAP

        logger.info(f"Generated {len(chunks)} chunks from text")
        return chunks

    @staticmethod
    def process_document(file: UploadFile, user_id: int, db: Session) -> Document:
        """Handles full document processing, embedding, and indexing."""
        try:
            # Step 1: Validate
            DocumentService.validate_file(file)

            # Step 2: Save
            file_path, filename, file_size = DocumentService.save_uploaded_file(file, user_id)
            file_type = Path(file.filename).suffix.lower()

            # Step 3: Extract text
            text = DocumentService.extract_text(file_path, file_type)
            content_preview = text[:500] if len(text) > 500 else text

            # Step 4: Create DB entry
            document = Document(
                filename=filename,
                original_filename=file.filename,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                content_preview=content_preview,
                user_id=user_id,
                chunk_count=0
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            # Step 5: Chunking
            chunks = DocumentService.chunk_text(text)
            if not chunks:
                raise HTTPException(status_code=400, detail="Document has no readable text content")

            # Step 6: Embeddings
            embeddings = EmbeddingService.generate_embeddings_batch(chunks)
            embeddings_array = np.array(embeddings).astype("float32")

            # Step 7: FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            index_dir = Path(settings.FAISS_INDEX_DIR) / str(user_id)
            index_dir.mkdir(parents=True, exist_ok=True)

            index_path = index_dir / f"doc_{document.id}.index"
            chunks_path = index_dir / f"doc_{document.id}_chunks.txt"

            faiss.write_index(index, str(index_path))

            with open(chunks_path, "w", encoding="utf-8") as f:
                f.write("\n---CHUNK---\n".join(chunks))

            # Step 8: Update DB
            document.faiss_index_path = str(index_path)
            document.chunk_count = len(chunks)

            db.commit()
            db.refresh(document)

            logger.info(f"Document processed successfully: {document.original_filename}")
            return document

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
