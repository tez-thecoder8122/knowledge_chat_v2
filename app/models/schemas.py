from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List


# User Schemas
class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class SourceInfo(BaseModel):
    document: str
    chunk: str
    distance: float


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# Document Schemas
class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: datetime
    message: str
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for listing user documents."""
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


# Query Schemas
class QueryRequest(BaseModel):
    """Schema for knowledge base query."""
    question: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    """Schema for query response."""
    question: str
    answer: str
    sources: List[SourceInfo]
    context_used: List[str]
    timestamp: datetime


# Health Check Schema
class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    database: str
    openai: str
    timestamp: datetime

class ImageMedia(BaseModel):
    id: int
    type: str
    page_number: Optional[int]
    description: str
    image_base64: str
    image_format: str
    relevance_score: float

class TableMedia(BaseModel):
    id: int
    type: str
    page_number: Optional[int]
    description: str
    table_csv: str
    table_html: str
    relevance_score: float

class TextSource(BaseModel):
    document: str
    chunk: str
    distance: float

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    include_media: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    text_sources: List[TextSource]
    media_items: List[dict]  # Can be Image or Media
    context_used: List[str]
    timestamp: str