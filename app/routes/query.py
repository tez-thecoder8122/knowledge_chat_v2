from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.db.session import get_db
from app.services.query_service import QueryService
from app.services.auth_service import get_current_user
from app.models.schemas import QueryRequest, QueryResponse, TextSource, ImageMedia, TableMedia



router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("/ask", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Query the knowledge base and get answers with related media.
    - answer: Generated answer from LLM
    - text_sources: Text chunks used for answer
    - media_items: Related images and tables 
    - context_used: Clean context chunks
    """
    try:
        answer, sources, context_chunks, media_items = QueryService.query_knowledge_base(
            db=db,
            user_id=current_user.id,
            question=request.question,
            top_k=request.top_k,
            include_media=True
        )

        
        text_sources = [
            TextSource(
                document=source["document"],
                chunk=source["chunk"],
                distance=source["distance"]
            )
            for source in sources
        ]

        return QueryResponse(
            question=request.question,
            answer=answer,
            text_sources=text_sources,
            media_items=media_items,
            context_used=context_chunks,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/media/{media_id}")
async def get_media_details(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific media item.
    """
    try:
        from app.models.database import DocumentMedia
        
        media = db.query(DocumentMedia).filter(
            DocumentMedia.id == media_id
        ).first()
        
        if not media:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found"
            )
        
        # Check user authorization (user owns the document)
        doc = media.document
        if doc.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return {
            "id": media.id,
            "type": media.media_type,
            "page_number": media.page_number,
            "description": media.description,
            "created_at": media.created_at,
            "document": {
                "id": doc.id,
                "filename": doc.original_filename
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/media/{media_id}/data")
async def get_media_data(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Download/retrieve media data (image or table).
    """
    try:
        from app.models.database import DocumentMedia, MediaType
        import base64
        
        media = db.query(DocumentMedia).filter(
            DocumentMedia.id == media_id
        ).first()
        
        if not media:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found"
            )
        
        # Check authorization
        if media.document.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if media.media_type == MediaType.IMAGE:
            return {
                "type": "image",
                "format": media.image_format,
                "data": base64.b64encode(media.image_data).decode('utf-8') if media.image_data else ""
            }
        
        elif media.media_type == MediaType.TABLE:
            return {
                "type": "table",
                "csv": media.table_data,
                "html": media.table_html,
                "description": media.description
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )