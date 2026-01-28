from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict
from ..schemas.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    TokenRefresh,
    MessageResponse,
)
from ..core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory user storage (for demo purposes)
# In production, use a proper database
users_db: Dict[str, dict] = {}
user_id_counter = 0


def get_user_by_username(username: str) -> dict | None:
    """Get user by username."""
    return users_db.get(username)


def get_user_by_email(email: str) -> dict | None:
    """Get user by email."""
    for user in users_db.values():
        if user["email"] == email:
            return user
    return None


@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user."""
    global user_id_counter

    # Check if username already exists
    if get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user_id_counter += 1
    hashed_password = get_password_hash(user_data.password)

    users_db[user_data.username] = {
        "id": user_id_counter,
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
    }

    return MessageResponse(
        message=f"User {user_data.username} registered successfully",
        success=True,
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Authenticate user and return tokens."""
    user = get_user_by_username(credentials.username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    token_data = {"sub": user["username"], "email": user["email"]}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh):
    """Refresh access token using refresh token."""
    payload = decode_token(token_data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    user = get_user_by_username(username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new tokens
    new_token_data = {"sub": user["username"], "email": user["email"]}
    access_token = create_access_token(new_token_data)
    new_refresh_token = create_refresh_token(new_token_data)

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.get("/me", response_model=dict)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    user = get_user_by_username(current_user["username"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
    }


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should discard tokens)."""
    return MessageResponse(
        message="Successfully logged out",
        success=True,
    )
