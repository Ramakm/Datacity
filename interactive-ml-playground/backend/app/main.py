from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ml, auth

app = FastAPI(
    title="Interactive ML Playground API",
    description="Backend API for the Interactive ML Playground - Learn ML by doing!",
    version="1.0.0",
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
