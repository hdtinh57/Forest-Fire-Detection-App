# WebSocket Service for Real-time Video Streaming
import asyncio
import base64
import json
import time
from typing import Optional
import cv2
import numpy as np
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Model paths
MODEL_PATHS = [
    PROJECT_ROOT / "app" / "weights" / "best.pt",
    PROJECT_ROOT / "MyFireProject" / "yolo11n_fire_run5" / "weights" / "best.pt",
]


class WebSocketStreamService:
    """Service for handling real-time video streaming via WebSocket."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the YOLO model for inference."""
        try:
            from ultralytics import YOLO
            
            for path in MODEL_PATHS:
                if path.exists():
                    self._model = YOLO(str(path))
                    print(f"✅ WebSocket Service: Model loaded from {path}")
                    return
            
            print("⚠️ WebSocket Service: No model found, using mock detection")
        except Exception as e:
            print(f"⚠️ WebSocket Service: Failed to load model: {e}")
    
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
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return detection results.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            dict with prediction, confidence, and processed frame
        """
        start_time = time.time()
        h, w = frame.shape[:2]
        
        prediction = "NON-FIRE"
        confidence = 0.0
        
        if self._model is not None:
            try:
                # Run YOLO inference
                results = self._model(frame, verbose=False)
                result = results[0]
                
                probs = result.probs
                if probs is not None:
                    top1_idx = probs.top1
                    confidence = float(probs.top1conf.item())
                    class_name = result.names[top1_idx]
                    prediction = self._map_class_name(class_name)
            except Exception as e:
                print(f"Inference error: {e}")
        else:
            # Mock detection based on color (for testing without model)
            prediction, confidence = self._mock_detect(frame)
        
        # Draw result on frame
        processed_frame = self._draw_result(frame, prediction, confidence)
        
        processing_time = time.time() - start_time
        
        # Encode frame to base64 for WebSocket transmission
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time * 1000, 2),
            "frame": frame_base64,
            "width": w,
            "height": h
        }
    
    def _mock_detect(self, frame: np.ndarray) -> tuple:
        """Mock detection using color analysis."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire-like colors (red/orange)
        lower_fire1 = np.array([0, 100, 100])
        upper_fire1 = np.array([15, 255, 255])
        lower_fire2 = np.array([160, 100, 100])
        upper_fire2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_ratio = fire_pixels / total_pixels
        
        if fire_ratio > 0.05:  # More than 5% fire-like pixels
            return "FIRE", min(0.9, 0.5 + fire_ratio * 2)
        else:
            return "NON-FIRE", 0.85
    
    def _draw_result(self, frame: np.ndarray, prediction: str, confidence: float) -> np.ndarray:
        """Draw detection result on frame."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Colors
        if prediction == "FIRE":
            color = (0, 0, 255)  # Red
            text = f"🔥 FIRE: {confidence*100:.1f}%"
        else:
            color = (0, 255, 0)  # Green
            text = f"✓ SAFE: {confidence*100:.1f}%"
        
        # Draw status bar at top
        cv2.rectangle(output, (0, 0), (w, 50), color, -1)
        cv2.putText(output, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return output
    
    async def process_video_stream(self, video_path: str, websocket, frame_skip: int = 2):
        """
        Process video file and stream results via WebSocket.
        
        Args:
            video_path: Path to video file
            websocket: WebSocket connection
            frame_skip: Process every Nth frame (default: 2)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open video"})
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        fire_frames = 0
        
        try:
            await websocket.send_json({
                "type": "video_info",
                "total_frames": total_frames,
                "fps": fps
            })
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    continue
                
                # Process frame
                result = self.process_frame(frame)
                
                if result["prediction"] == "FIRE":
                    fire_frames += 1
                
                # Send result
                await websocket.send_json({
                    "type": "frame",
                    "frame_number": frame_count,
                    "total_frames": total_frames,
                    "progress": round(frame_count / total_frames * 100, 1),
                    **result
                })
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
            
            # Send final summary
            await websocket.send_json({
                "type": "complete",
                "total_frames": frame_count,
                "fire_frames": fire_frames,
                "fire_percentage": round(fire_frames / (frame_count // frame_skip) * 100, 1) if frame_count > 0 else 0
            })
            
        finally:
            cap.release()


# Singleton instance
websocket_stream_service = WebSocketStreamService()
