from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import openai

from app.config import settings
from app.db.session import init_db
from app.routes import auth, documents, query
from app.models.schemas import HealthCheck
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Knowledge Chat System API...")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Verify OpenAI API key
    try:
        openai.api_key = settings.OPENAI_API_KEY
        logger.info("OpenAI API key configured")
    except Exception as e:
        logger.warning(f"OpenAI configuration issue: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Knowledge Chat System API...")


# Create FastAPI application
app = FastAPI(
    title="Knowledge Chat System API",
    description="Production-grade backend for Generative AI Knowledge Chat System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Knowledge Chat System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    # Check database
    try:
        from app.db.session import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # Check OpenAI
    try:
        openai.api_key = settings.OPENAI_API_KEY
        if settings.OPENAI_API_KEY and len(settings.OPENAI_API_KEY) > 0:
            openai_status = "configured"
        else:
            openai_status = "not_configured"
    except Exception:
        openai_status = "error"
    
    return HealthCheck(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        openai=openai_status,
        timestamp=datetime.utcnow()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
