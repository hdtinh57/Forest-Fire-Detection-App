# Routes package
from routes.detection import router as detection_router
from routes.websocket import router as websocket_router

__all__ = ["detection_router", "websocket_router"]
