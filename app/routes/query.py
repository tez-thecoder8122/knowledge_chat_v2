from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_service import QueryService
from app.db.session import get_db
from app.utils.helpers import get_current_user
from app.models.database import User
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    query_data: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Query the knowledge base and get AI-generated answer.
    
    Args:
        query_data: Query request with question
        current_user: Authenticated user
        db: Database session
        
    Returns:
        AI-generated answer with sources and context
    """
    try:
        answer, sources, context_chunks = QueryService.query_knowledge_base(
            db=db,
            user_id=current_user.id,
            question=query_data.question,
            top_k=query_data.top_k
        )
        
        logger.info(f"Query processed for user {current_user.username}: {query_data.question[:50]}...")
        
        return QueryResponse(
            question=query_data.question,
            answer=answer,
            sources=sources,
            context_used=context_chunks,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )
