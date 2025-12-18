from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import detection_router, websocket_router

# Create FastAPI app
app = FastAPI(
    title="🔥 Forest Fire Detection API",
    description="API for detecting forest fires using YOLO11 classification model with WebSocket streaming",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "🔥 Forest Fire Detection API",
        "docs": "/docs",
        "health": "/api/health"
    }
