import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from schemas import DetectionResponse, VideoDetectionResponse, HealthResponse
from services import fire_detection_service

router = APIRouter(prefix="/api", tags=["detection"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=fire_detection_service.is_model_loaded
    )


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    enable_preprocessing: bool = True
):
    """
    Detect fire in an uploaded image.
    
    Args:
        file: Image file (jpg, png, jpeg)
        enable_preprocessing: Bật/tắt tiền xử lý ảnh (khử nhiễu, CLAHE)
        
    Returns:
        Detection result with prediction, confidence, and processing time
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run detection with optional preprocessing
        prediction, confidence, processing_time = fire_detection_service.detect_image(
            image_bytes, 
            enable_preprocessing=enable_preprocessing
        )
        
        return DetectionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            processing_time=round(processing_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(file: UploadFile = File(...)):
    """
    Detect fire in an uploaded video.
    
    Args:
        file: Video file (mp4, avi, mov)
        
    Returns:
        Detection results with frame analysis and overall prediction
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: mp4, avi, mov"
        )
    
    try:
        # Save video to temp file
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Run detection
            result = fire_detection_service.detect_video(tmp_path)
            
            return VideoDetectionResponse(**result)
        
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
