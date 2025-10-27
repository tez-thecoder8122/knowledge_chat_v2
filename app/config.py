from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    
    # Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str
    
    # JWT Configuration
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    
    # Application Configuration
    UPLOAD_DIR: str = "./uploads"
    FAISS_INDEX_DIR: str = "./faiss_indexes"
    MAX_FILE_SIZE: int = 10485760  #10mb
    ALLOWED_EXTENSIONS: str = ".pdf,.txt"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
