from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, LargeBinary, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum


Base = declarative_base()

class MediaType(str, enum.Enum):
        """Types of media extracted from documents"""
        IMAGE = "image"
        TABLE = "table"
        TEXT = "text"


class User(Base):
    """User model for authentication and document ownership."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with documents
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    """Document model for storing uploaded file metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_preview = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    faiss_index_path = Column(String(500))
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with user
    owner = relationship("User", back_populates="documents")
    
    media = relationship("DocumentMedia", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    


class DocumentMedia(Base):
    """Store extracted media (images, tables) from documents"""
    __tablename__ = "document_media"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    media_type = Column(String, default=MediaType.IMAGE)
    image_data = Column(LargeBinary, nullable=True)  #BLOB
    image_format = Column(String, default="png")  
    table_data = Column(Text, nullable=True)  
    table_html = Column(Text, nullable=True)  
    page_number = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)  
    embedding = Column(String, nullable=True)  
    embedding_vector = Column(LargeBinary, nullable=True)  
    associated_text = Column(Text, nullable=True)  
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="media")


class DocumentChunk(Base):
    """Enhanced chunk model with media references"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer)
    chunk_text = Column(Text)
    related_media_ids = Column(String, nullable=True)  #Comma Seperated ID's
    embedding_vector = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chunks")
    
    
