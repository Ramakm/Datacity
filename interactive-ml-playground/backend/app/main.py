from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.routers import ml, auth
from app.core.database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Check database connection
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("Database connection established successfully")
    except Exception as e:
        print(f"Warning: Database connection failed: {e}")
        print("Make sure PostgreSQL is running and the database exists")
    yield
    # Shutdown: Dispose engine
    await engine.dispose()


app = FastAPI(
    title="Interactive ML Playground API",
    description="Backend API for the Interactive ML Playground - Learn ML by doing!",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(ml.router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Interactive ML Playground API is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "available_models": ["linear-regression", "logistic-regression", "knn", "kmeans"],
        "auth_enabled": True,
    }
