from pydantic import BaseModel
from typing import List, Optional


class DetectionResponse(BaseModel):
    """Response schema for single image detection."""
    prediction: str  # "FIRE" or "NON-FIRE"
    confidence: float
    processing_time: float  # in seconds


class VideoFrameResult(BaseModel):
    """Result for a single video frame."""
    frame_number: int
    prediction: str
    confidence: float


class VideoDetectionResponse(BaseModel):
    """Response schema for video detection."""
    total_frames: int
    fire_frames: int
    non_fire_frames: int
    fire_percentage: float
    overall_prediction: str  # "FIRE" or "NON-FIRE" based on majority
    processing_time: float
    frame_results: Optional[List[VideoFrameResult]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
