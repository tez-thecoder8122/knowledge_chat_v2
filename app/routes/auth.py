from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.models.schemas import UserCreate, UserLogin, Token, UserResponse
from app.services.auth_service import AuthService
from app.db.session import get_db
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Created user information
    """
    try:
        user = AuthService.create_user(db, user_data)
        logger.info(f"User registered successfully: {user.username}")
        return user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Login and receive JWT token.
    
    Args:
        user_data: User login credentials
        db: Database session
        
    Returns:
        JWT access token and user information
    """
    # Authenticate user
    user = AuthService.authenticate_user(db, user_data.username, user_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in successfully: {user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }
