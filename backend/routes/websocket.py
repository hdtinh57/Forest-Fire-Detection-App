# WebSocket Routes for Real-time Streaming
import os
import tempfile
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse

from services.websocket_stream import websocket_stream_service

router = APIRouter(prefix="/api/ws", tags=["websocket"])


@router.websocket("/stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing.
    
    Client sends:
    - {"type": "start", "video_path": "path/to/video.mp4"} to start processing
    - {"type": "frame", "data": "base64_image_data"} for individual frames
    - {"type": "stop"} to stop processing
    
    Server sends:
    - {"type": "video_info", "total_frames": N, "fps": X}
    - {"type": "frame", "frame_number": N, "prediction": "FIRE/NON-FIRE", "confidence": 0.95, "frame": "base64"}
    - {"type": "complete", "total_frames": N, "fire_frames": M}
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "start":
                video_path = data.get("video_path")
                frame_skip = data.get("frame_skip", 2)
                
                if video_path and os.path.exists(video_path):
                    await websocket_stream_service.process_video_stream(
                        video_path, websocket, frame_skip
                    )
                else:
                    await websocket.send_json({"error": "Video path not found"})
            
            elif msg_type == "frame":
                # Process single frame sent as base64
                import base64
                import numpy as np
                import cv2
                
                frame_data = data.get("data")
                if frame_data:
                    # Decode base64 to image
                    img_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        result = websocket_stream_service.process_frame(frame)
                        await websocket.send_json({
                            "type": "frame_result",
                            **result
                        })
                    else:
                        await websocket.send_json({"error": "Invalid frame data"})
            
            elif msg_type == "stop":
                await websocket.send_json({"type": "stopped"})
                break
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@router.post("/upload-and-stream")
async def upload_video_for_streaming(file: UploadFile = File(...)):
    """
    Upload a video file and get a path for WebSocket streaming.
    
    Returns:
        video_path: Path to use with WebSocket streaming
    """
    allowed_types = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid file type. Allowed: mp4, avi, mov"}
        )
    
    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    return {
        "video_path": tmp_path,
        "filename": file.filename,
        "message": "Video uploaded. Connect to WebSocket and send {'type': 'start', 'video_path': '<path>'}"
    }
