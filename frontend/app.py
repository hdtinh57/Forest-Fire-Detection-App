import streamlit as st
import requests
from PIL import Image
import io
import time
import base64
import asyncio
import json
import tempfile
import os

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

# Page config
st.set_page_config(
    page_title="🔥 Forest Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .result-fire {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(255, 68, 68, 0.3);
    }
    .result-safe {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(0, 200, 81, 0.3);
    }
    .confidence-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #f7931e, #ff6b35);
    }
    .stream-status {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stream-active {
        background: #28a745;
        color: white;
    }
    .stream-processing {
        background: #ffc107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def detect_image(image_bytes: bytes, enable_preprocessing: bool = True) -> dict:
    """Send image to API for detection."""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    params = {"enable_preprocessing": enable_preprocessing}
    response = requests.post(f"{API_BASE_URL}/api/detect/image", files=files, params=params)
    response.raise_for_status()
    return response.json()


def upload_video_for_streaming(video_bytes: bytes, filename: str) -> dict:
    """Upload video and get path for WebSocket streaming."""
    content_type = "video/mp4"
    if filename.endswith(".avi"):
        content_type = "video/avi"
    elif filename.endswith(".mov"):
        content_type = "video/quicktime"
    
    files = {"file": (filename, video_bytes, content_type)}
    response = requests.post(f"{API_BASE_URL}/api/ws/upload-and-stream", files=files)
    response.raise_for_status()
    return response.json()


def display_result(prediction: str, confidence: float, processing_time: float):
    """Display detection result with styling."""
    if prediction == "FIRE":
        st.markdown(f"""
            <div class="result-fire">
                🔥 FIRE DETECTED 🔥
                <div class="confidence-box">
                    Confidence: {confidence*100:.1f}%<br>
                    Processing Time: {processing_time:.3f}s
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.error("⚠️ **CẢNH BÁO: Phát hiện có lửa trong ảnh!**")
    else:
        st.markdown(f"""
            <div class="result-safe">
                ✅ NON-FIRE ✅
                <div class="confidence-box">
                    Confidence: {confidence*100:.1f}%<br>
                    Processing Time: {processing_time:.3f}s
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.success("✅ **AN TOÀN: Không phát hiện lửa trong ảnh.**")


def process_video_with_websocket(video_path: str, frame_placeholder, stats_placeholder, progress_placeholder):
    """Process video using WebSocket for real-time streaming."""
    import websocket
    import threading
    
    results = {
        "frames_processed": 0,
        "fire_frames": 0,
        "total_frames": 0,
        "current_frame": None,
        "current_prediction": None,
        "current_confidence": 0,
        "processing_time_ms": 0,
        "is_complete": False,
        "error": None
    }
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "video_info":
                results["total_frames"] = data.get("total_frames", 0)
            
            elif msg_type == "frame":
                results["frames_processed"] = data.get("frame_number", 0)
                results["current_prediction"] = data.get("prediction")
                results["current_confidence"] = data.get("confidence", 0)
                results["processing_time_ms"] = data.get("processing_time_ms", 0)
                results["current_frame"] = data.get("frame")
                
                if data.get("prediction") == "FIRE":
                    results["fire_frames"] += 1
            
            elif msg_type == "complete":
                results["is_complete"] = True
                results["fire_percentage"] = data.get("fire_percentage", 0)
            
            elif "error" in data:
                results["error"] = data["error"]
                
        except Exception as e:
            results["error"] = str(e)
    
    def on_error(ws, error):
        results["error"] = str(error)
    
    def on_close(ws, close_status_code, close_msg):
        pass
    
    def on_open(ws):
        # Send start command
        ws.send(json.dumps({
            "type": "start",
            "video_path": video_path,
            "frame_skip": 2
        }))
    
    # Connect to WebSocket
    ws_url = f"{WS_BASE_URL}/api/ws/stream"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket in thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Update UI while processing
    while not results["is_complete"] and results["error"] is None:
        time.sleep(0.1)
        
        # Update frame display
        if results["current_frame"]:
            try:
                frame_bytes = base64.b64decode(results["current_frame"])
                frame_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)
            except:
                pass
        
        # Update stats
        pred_emoji = "🔥" if results["current_prediction"] == "FIRE" else "✅"
        stats_placeholder.markdown(f"""
        **Current Frame:** {results['frames_processed']} / {results['total_frames']}
        
        **Prediction:** {pred_emoji} {results['current_prediction'] or 'Processing...'}
        
        **Confidence:** {results['current_confidence']*100:.1f}%
        
        **Processing Time:** {results['processing_time_ms']:.1f}ms
        
        **🔥 Fire Frames:** {results['fire_frames']}
        """)
        
        # Update progress
        if results["total_frames"] > 0:
            progress = results["frames_processed"] / results["total_frames"]
            progress_placeholder.progress(progress)
    
    ws.close()
    
    return results


def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 Forest Fire Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
            **Forest Fire Detection** sử dụng mô hình YOLO11 Classification để phát hiện cháy rừng.
            
            📸 **Input:** Ảnh hoặc Video  
            📊 **Output:** FIRE hoặc NON-FIRE
        """)
        
        st.header("🔌 API Status")
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.warning("Hãy chạy backend server:\n```\ncd app/backend\nuvicorn main:app --reload\n```")
        
        st.header("📋 Instructions")
        st.markdown("""
            1. Chọn tab **Image** hoặc **Video**
            2. Upload file của bạn
            3. Nhấn **Detect** để phân tích
            4. Xem kết quả!
        """)
    
    # Main content
    tab_image, tab_video = st.tabs(["📸 Image Detection", "🎬 Video Detection"])
    
    # Image Detection Tab
    with tab_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_image = st.file_uploader(
                "Chọn ảnh (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                key="image_uploader"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Ảnh đã upload", use_container_width=True)
                
                # Preprocessing toggle
                enable_preprocess = st.checkbox(
                    "🔧 Bật tiền xử lý ảnh (Denoise, CLAHE, White Balance)",
                    value=True,
                    key="preprocess_toggle",
                    help="Áp dụng các kỹ thuật xử lý ảnh trước khi detect để cải thiện độ chính xác"
                )
                
                if st.button("🔍 Detect Fire", key="detect_image_btn"):
                    if not check_api_health():
                        st.error("❌ API không khả dụng. Hãy chạy backend server!")
                    else:
                        with st.spinner("🔄 Đang phân tích..."):
                            try:
                                img_bytes = io.BytesIO()
                                if image.mode == 'RGBA':
                                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                                    rgb_image.paste(image, mask=image.split()[3])
                                    rgb_image.save(img_bytes, format="JPEG")
                                else:
                                    image.save(img_bytes, format="JPEG")
                                img_bytes = img_bytes.getvalue()
                                
                                result = detect_image(img_bytes, enable_preprocessing=enable_preprocess)
                                st.session_state["image_result"] = result
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.subheader("📊 Detection Result")
            if "image_result" in st.session_state:
                result = st.session_state["image_result"]
                display_result(
                    result["prediction"],
                    result["confidence"],
                    result["processing_time"]
                )
            else:
                st.info("👆 Upload ảnh và nhấn Detect để xem kết quả")
    
    # Video Detection Tab (WebSocket Streaming)
    with tab_video:
        st.subheader("🎬 Video Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_video = st.file_uploader(
                "Chọn video (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                key="video_uploader"
            )
            
            if uploaded_video:
                st.video(uploaded_video)
                
                if st.button("🚀 Stream & Detect", key="stream_video_btn", type="primary"):
                    if not check_api_health():
                        st.error("❌ API không khả dụng!")
                    else:
                        try:
                            # Save video temporarily
                            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                                tmp.write(uploaded_video.read())
                                tmp_path = tmp.name
                            
                            # Upload for streaming
                            with st.spinner("📤 Uploading video..."):
                                video_bytes = open(tmp_path, "rb").read()
                                upload_result = upload_video_for_streaming(video_bytes, uploaded_video.name)
                            
                            video_path = upload_result.get("video_path")
                            
                            if video_path:
                                st.markdown('<div class="stream-status stream-active">🔴 STREAMING...</div>', unsafe_allow_html=True)
                                
                                # Create placeholders
                                frame_placeholder = st.empty()
                                progress_placeholder = st.empty()
                                stats_placeholder = st.empty()
                                
                                # Process with WebSocket
                                results = process_video_with_websocket(
                                    video_path, 
                                    frame_placeholder, 
                                    stats_placeholder,
                                    progress_placeholder
                                )
                                
                                if results.get("error"):
                                    st.error(f"❌ Error: {results['error']}")
                                else:
                                    st.success("✅ Processing complete!")
                                    st.session_state["video_result"] = results
                            
                            # Cleanup
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.subheader("📊 Video Statistics")
            if "video_result" in st.session_state:
                result = st.session_state["video_result"]
                
                fire_pct = result.get("fire_percentage", 0)
                
                if fire_pct > 50:
                    st.markdown("""
                    <div class="result-fire" style="padding: 1rem; font-size: 1.5rem;">
                        🔥 FIRE DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-safe" style="padding: 1rem; font-size: 1.5rem;">
                        ✅ MOSTLY SAFE
                    </div>
                    """, unsafe_allow_html=True)
                
                st.metric("Total Frames", result.get("frames_processed", 0))
                st.metric("🔥 Fire Frames", result.get("fire_frames", 0))
                st.metric("Fire Percentage", f"{fire_pct:.1f}%")
            else:
                st.info("👆 Upload video và nhấn Stream để xem kết quả")



if __name__ == "__main__":
    main()

