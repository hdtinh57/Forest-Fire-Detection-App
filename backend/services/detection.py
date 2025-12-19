import os
import time
import tempfile
from pathlib import Path
from typing import List, Tuple
import cv2
from ultralytics import YOLO

# Get the backend directory (where this service is located)
BACKEND_DIR = Path(__file__).parent.parent  # backend/
APP_DIR = BACKEND_DIR.parent  # app/

# Check multiple possible model locations (relative paths)
MODEL_PATHS = [
    APP_DIR / "weights" / "best.pt",  # app/weights/best.pt (recommended)
    BACKEND_DIR / "weights" / "best.pt",  # backend/weights/best.pt
    Path("weights") / "best.pt",  # current working directory
]

# Find the first existing model path
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = path
        break





class FireDetectionService:
    """Service for fire detection using YOLO11 classification model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the YOLO11 classification model."""
        if MODEL_PATH is None:
            raise FileNotFoundError(
                f"Model not found in any of these locations:\n" + 
                "\n".join(str(p) for p in MODEL_PATHS)
            )
        self._model = YOLO(str(MODEL_PATH))
        print(f"✅ Model loaded from {MODEL_PATH}")
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def _map_class_name(self, class_name: str) -> str:
        """Map model class names to output labels."""
        class_mapping = {
            "Fire": "FIRE",
            "fire": "FIRE",
            "Non_Fire": "NON-FIRE",
            "non_fire": "NON-FIRE",
            "NonFire": "NON-FIRE",
        }
        return class_mapping.get(class_name, class_name.upper())
    
    def detect_image(
        self, 
        image_bytes: bytes, 
        enable_preprocessing: bool = True
    ) -> Tuple[str, float, float]:
        """
        Detect fire in an image.
        
        Args:
            image_bytes: Image data as bytes
            enable_preprocessing: Bật/tắt tiền xử lý ảnh
            
        Returns:
            Tuple of (prediction, confidence, processing_time)
        """
        start_time = time.time()
        
        # Convert bytes to numpy array for preprocessing
        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Apply preprocessing if enabled
        if enable_preprocessing and image is not None:
            from services.preprocessing import image_preprocessor
            image = image_preprocessor.process_for_fire_detection(image)
        
        # Save preprocessed image to temp file for YOLO processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if image is not None:
                cv2.imwrite(tmp.name, image)
            else:
                tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            # Run inference
            results = self._model(tmp_path, verbose=False)
            result = results[0]
            
            # Get top prediction
            probs = result.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = result.names[top1_idx]
            
            prediction = self._map_class_name(class_name)
            processing_time = time.time() - start_time
            
            return prediction, top1_conf, processing_time
            
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    def detect_video(self, video_path: str, sample_rate: int = 5) -> dict:
        """
        Detect fire in a video by sampling frames.
        
        Args:
            video_path: Path to the video file
            sample_rate: Process every Nth frame (default: 5)
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_results = []
        fire_count = 0
        non_fire_count = 0
        frame_number = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Sample frames
                if frame_number % sample_rate != 0:
                    continue
                
                processed_frames += 1
                
                # Save frame to temp file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name
                
                try:
                    # Run inference on frame
                    results = self._model(tmp_path, verbose=False)
                    result = results[0]
                    
                    probs = result.probs
                    top1_idx = probs.top1
                    top1_conf = probs.top1conf.item()
                    class_name = result.names[top1_idx]
                    
                    prediction = self._map_class_name(class_name)
                    
                    if prediction == "FIRE":
                        fire_count += 1
                    else:
                        non_fire_count += 1
                    
                    frame_results.append({
                        "frame_number": frame_number,
                        "prediction": prediction,
                        "confidence": top1_conf
                    })
                finally:
                    os.unlink(tmp_path)
        
        finally:
            cap.release()
        
        processing_time = time.time() - start_time
        total_frames = processed_frames
        
        # Determine overall prediction based on majority
        if fire_count > non_fire_count:
            overall_prediction = "FIRE"
        else:
            overall_prediction = "NON-FIRE"
        
        fire_percentage = (fire_count / total_frames * 100) if total_frames > 0 else 0
        
        return {
            "total_frames": total_frames,
            "fire_frames": fire_count,
            "non_fire_frames": non_fire_count,
            "fire_percentage": round(fire_percentage, 2),
            "overall_prediction": overall_prediction,
            "processing_time": round(processing_time, 3),
            "frame_results": frame_results
        }


# Create singleton instance
fire_detection_service = FireDetectionService()
