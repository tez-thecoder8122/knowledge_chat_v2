from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.schemas import DocumentUploadResponse, DocumentListResponse
from app.services.document_service import DocumentService
from app.db.session import get_db
from app.utils.helpers import get_current_user
from app.models.database import User, Document
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.
    
    Args:
        file: Document file (PDF or TXT)
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Document upload information
    """
    try:
        document = DocumentService.process_document(file, current_user.id, db)
        
        logger.info(f"Document uploaded by user {current_user.username}: {document.original_filename}")
        
        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size=document.file_size,
            chunk_count=document.chunk_count,
            uploaded_at=document.uploaded_at,
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/", response_model=List[DocumentListResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents for the current user.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of user documents
    """
    documents = db.query(Document).filter(Document.user_id == current_user.id).all()
    logger.info(f"User {current_user.username} retrieved {len(documents)} documents")
    return documents


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document.
    
    Args:
        document_id: Document ID to delete
        current_user: Authenticated user
        db: Database session
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete file and index
    import os
    
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        if document.faiss_index_path and os.path.exists(document.faiss_index_path):
            os.remove(document.faiss_index_path)
            chunks_path = document.faiss_index_path.replace(".index", "_chunks.txt")
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
    except Exception as e:
        logger.warning(f"Error deleting files: {e}")
    
    db.delete(document)
    db.commit()
    
    logger.info(f"Document deleted: {document.original_filename}")
