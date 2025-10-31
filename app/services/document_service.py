import os
import uuid
import re
import json
import base64
from pathlib import Path
from typing import List, Tuple, Dict
import PyPDF2
import faiss
import numpy as np
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException, status

from app.models.database import Document, DocumentMedia, MediaType
from app.services.embedding_service import EmbeddingService
from app.services.vision_service import VisionService
from app.config import settings
from app.utils.logger import setup_logger
from app.models.database import DocumentChunk

logger = setup_logger(__name__)


class DocumentService:
    """Enhanced service for handling document operations with media support"""
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
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
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n\s+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r'[\u00A0\u2000-\u200B]', ' ', text)
        text = re.sub(r'(\S)\s+(\S)', r'\1 \2', text)
        return text.strip()
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text content from PDF with proper cleaning."""
        try:
            text = ""
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            
            text = DocumentService.clean_text(text)
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process PDF file"
            )
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text content from TXT file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            text = DocumentService.clean_text(text)
            logger.info(f"Extracted {len(text)} characters from TXT")
            return text
        
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
    def chunk_text_by_sentences(text: str) -> List[str]:
        """Split text into chunks based on sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= DocumentService.CHUNK_SIZE:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-DocumentService.CHUNK_OVERLAP:] if len(prev_chunk) > DocumentService.CHUNK_OVERLAP else prev_chunk
                    combined = overlap_text + " " + chunk
                    overlapped_chunks.append(combined.strip())
            chunks = overlapped_chunks
        
        logger.info(f"Generated {len(chunks)} chunks from text")
        return chunks
    
    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """Split text into overlapping chunks."""
        try:
            chunks = DocumentService.chunk_text_by_sentences(text)
            
            if not chunks or len(chunks) == 1 and len(text) > DocumentService.CHUNK_SIZE * 2:
                logger.info("Falling back to character-based chunking")
                chunks = []
                start = 0
                text_length = len(text)
                
                while start < text_length:
                    end = start + DocumentService.CHUNK_SIZE
                    chunk = text[start:end]
                    chunk = DocumentService.clean_text(chunk.strip())
                    if chunk:
                        chunks.append(chunk)
                    start += DocumentService.CHUNK_SIZE - DocumentService.CHUNK_OVERLAP
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    @staticmethod
    def extract_and_store_media(file_path: str, document_id: int, user_id: int, db: Session) -> Tuple[List[DocumentMedia], int]:
        """
        Extract images and tables from PDF and store in database.
        
        Args:
            file_path: Path to PDF file
            document_id: Document ID
            user_id: User ID
            db: Database session
        
        Returns:
            Tuple of (media_list, total_media_count)
        """
        try:
            vision_service = VisionService()
            media_list = []
            
            # Create media directory
            media_dir = Path(settings.UPLOAD_DIR) / str(user_id) / f"doc_{document_id}_media"
            
            # Extract images
            logger.info("Extracting images from PDF...")
            images_data = VisionService.extract_images_from_pdf(file_path, media_dir / "images")
            
            for img_info in images_data:
                # Analyze image with vision
                description = vision_service.analyze_image_with_vision(img_info["filepath"])
                
                # Read image as binary
                with open(img_info["filepath"], "rb") as f:
                    image_binary = f.read()
                
                # Create media record
                media = DocumentMedia(
                    document_id=document_id,
                    media_type=MediaType.IMAGE,
                    image_data=image_binary,
                    image_format=img_info["format"],
                    page_number=img_info["page_number"],
                    description=description,
                    associated_text=""
                )
                db.add(media)
                media_list.append(media)
            
            db.flush()  # Flush to get IDs
            
            # Extract tables
            logger.info("Extracting tables from PDF...")
            tables_data = VisionService.extract_tables_from_pdf(file_path)
            
            for table_info in tables_data:
                media = DocumentMedia(
                    document_id=document_id,
                    media_type=MediaType.TABLE,
                    table_data=table_info["csv"],
                    table_html=table_info["html"],
                    page_number=table_info["page_number"],
                    description=f"Table with {table_info['rows']} rows and {table_info['cols']} columns",
                    associated_text=""
                )
                db.add(media)
                media_list.append(media)
            
            db.commit()
            logger.info(f"Extracted and stored {len(media_list)} media items")
            return media_list, len(media_list)
        
        except Exception as e:
            logger.error(f"Error extracting media: {e}", exc_info=True)
            db.rollback()
            return [], 0
    
    @staticmethod
    def link_media_to_chunks(db: Session, document_id: int) -> None:
        """
        Link media items to relevant text chunks based on context.
        
        Args:
            db: Database session
            document_id: Document ID
        """
        try:
            from app.models.database import DocumentChunk
            
            media_items = db.query(DocumentMedia).filter(
                DocumentMedia.document_id == document_id
            ).all()
            
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            # Link chunks to media based on page proximity
            for chunk in chunks:
                related_media = []
                for media in media_items:
                    related_media.append(str(media.id))
                
                if related_media:
                    chunk.related_media_ids = ",".join(related_media[:3])  # limits to 3 media files
            
            db.commit()
            logger.info("Linked media to chunks")
        
        except Exception as e:
            logger.error(f"Error linking media to chunks: {e}")
            db.rollback()
    
    @staticmethod
    def process_document(file: UploadFile, user_id: int, db: Session) -> Document:
        """Handles full document processing with media extraction."""
        try:
            # Step 1: Validate
            DocumentService.validate_file(file)
            
            # Step 2: Save file
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
            
            # Step 5: Extract media (images, tables)
            if file_type == ".pdf":
                logger.info("Processing media content...")
                media_dir = Path(settings.UPLOAD_DIR) / str(user_id) / f"doc_{document.id}_media"
                media_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Try extracting embedded images (PyMuPDF)
                embedded_images = VisionService.extract_images_from_pdf(file_path, str(media_dir))
                # 2. Also render every PDF page as image (guaranteed fallback)
                page_images = VisionService.render_pdf_pages_as_images(file_path, str(media_dir))
                
                # Option 1: Only embed page images if no embedded images are found
                images_data = embedded_images if embedded_images else page_images
                # Option 2: (Uncomment for both!)
                # images_data = embedded_images + page_images

                tables_data = VisionService.extract_tables_from_pdf(file_path)
                
                DocumentService.save_images_to_db(db, document.id, images_data)
                DocumentService.save_tables_to_db(db, document.id, tables_data)

            # Step 6: Chunking
            chunks = DocumentService.chunk_text(text)
            if not chunks:
                raise HTTPException(status_code=400, detail="Document has no readable text content")

            DocumentService.save_chunks_to_db(db, document.id, chunks)
            
            # Step 7: Embeddings
            embeddings = EmbeddingService.generate_embeddings_batch(chunks)
            embeddings_array = np.array(embeddings).astype("float32")
            
            # Step 8: FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            index_dir = Path(settings.FAISS_INDEX_DIR) / str(user_id)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            index_path = index_dir / f"doc_{document.id}.index"
            chunks_path = index_dir / f"doc_{document.id}_chunks.txt"
            
            faiss.write_index(index, str(index_path))
            
            # Save chunks to text file
            with open(chunks_path, "w", encoding="utf-8") as f:
                f.write("\n---CHUNK---\n".join(chunks))
            
            # Step 9: Update DB
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




    def save_chunks_to_db(db: Session, document_id: int, chunks: list):
        for idx, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=idx,
                chunk_text=chunk_text
            )
            db.add(chunk)
        db.commit()

    def save_images_to_db(db: Session, document_id: int, images_data: list):
        for img_info in images_data:
            with open(img_info["filepath"], "rb") as f:
                image_binary = f.read()
            media = DocumentMedia(
                document_id=document_id,
                media_type=MediaType.IMAGE,
                image_data=image_binary,
                image_format=img_info["format"],
                page_number=img_info["page_number"],
                description=img_info.get("description", ""),
                associated_text=""
            )
            db.add(media)
        db.commit()

    def save_tables_to_db(db: Session, document_id: int, tables_data: list):
        for table_info in tables_data:
            media = DocumentMedia(
                document_id=document_id,
                media_type=MediaType.TABLE,
                table_data=table_info["csv"],
                table_html=table_info["html"],
                page_number=table_info["page_number"],
                description=f"Table with {table_info['rows']} rows and {table_info['cols']} columns",
                associated_text=""
            )
            db.add(media)
        db.commit()
