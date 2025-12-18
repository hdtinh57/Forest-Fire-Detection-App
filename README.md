# 🔥 Forest Fire Detection App

Ứng dụng phát hiện cháy rừng sử dụng model YOLO11 Classification với WebSocket streaming cho xử lý video real-time.

## ✨ Features

- **🖼️ Image Detection**: Upload ảnh và phát hiện lửa với độ chính xác 99%
- **🎬 Video Detection**: Phân tích video qua HTTP hoặc WebSocket
- **⚡ WebSocket Streaming**: Xử lý video real-time với độ trễ thấp
- **🔧 Image Preprocessing**: Khử nhiễu, CLAHE, cân bằng trắng

## 📂 Project Structure

```
├── backend/
│   ├── routes/
│   │   ├── detection.py         # REST API endpoints
│   │   └── websocket.py         # WebSocket streaming
│   ├── schemas/
│   │   └── detection.py         # Pydantic models
│   ├── services/
│   │   ├── detection.py         # YOLO11 inference
│   │   ├── preprocessing.py     # Image preprocessing
│   │   └── websocket_stream.py  # WebSocket service
│   ├── main.py                  # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit UI
│   └── requirements.txt
├── src/                         # Source scripts
└── weights/
    └── best.pt                  # Trained YOLO11 model
```


## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 2. Run Backend (FastAPI)

```bash
cd ../backend
uvicorn main:app --reload --port 8000
```

API sẽ chạy tại: http://localhost:8000

Swagger Docs: http://localhost:8000/docs

### 3. Run Frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

Frontend sẽ mở tại: http://localhost:8501

## 📸 Usage

1. **Image Detection**: Upload ảnh JPG/PNG → Nhận kết quả FIRE/NON-FIRE
2. **Video Detection**: Upload video MP4/AVI → Phân tích từng frame

## 🔌 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect/image` | Detect fire in image |
| POST | `/api/detect/video` | Detect fire in video (sync) |

### WebSocket API

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/api/ws/stream` | Real-time video streaming |

#### WebSocket Protocol

**Upload video trước:**
```http
POST /api/ws/upload-and-stream
Content-Type: multipart/form-data

Response: {"video_path": "/tmp/video123.mp4"}
```

**Kết nối WebSocket và gửi lệnh:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/stream');

// 1. Bắt đầu xử lý video
ws.send(JSON.stringify({
    type: 'start',
    video_path: '/tmp/video123.mp4',
    frame_skip: 2  // Xử lý mỗi 2 frame
}));

// 2. Nhận kết quả từng frame
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'video_info') {
        console.log('Total frames:', data.total_frames);
    }
    
    if (data.type === 'frame') {
        console.log('Frame:', data.frame_number);
        console.log('Prediction:', data.prediction);  // FIRE or NON-FIRE
        console.log('Confidence:', data.confidence);
        // data.frame = base64 encoded image
    }
    
    if (data.type === 'complete') {
        console.log('Fire percentage:', data.fire_percentage);
    }
};

// 3. Dừng xử lý
ws.send(JSON.stringify({ type: 'stop' }));
```

#### Xử lý frame đơn lẻ

```javascript
// Gửi 1 frame để detect
ws.send(JSON.stringify({
    type: 'frame',
    data: base64ImageData  // Base64 encoded image
}));

// Nhận kết quả
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    // result.prediction: "FIRE" or "NON-FIRE"
    // result.confidence: 0.0 - 1.0
    // result.processing_time_ms: thời gian xử lý (ms)
    // result.frame: base64 processed image
};
```

## 🖼️ Image Preprocessing

Hệ thống áp dụng các kỹ thuật xử lý ảnh trước khi đưa vào model:

| Technique | Description |
|-----------|-------------|
| **Bilateral Filter** | Khử nhiễu giữ cạnh |
| **Saturation Boost** | Tăng độ bão hòa màu lửa |
| **CLAHE** | Cải thiện contrast cục bộ |

Có thể tắt preprocessing qua API:
```http
POST /api/detect/image?enable_preprocessing=false
```

## 📊 Output

- **FIRE** 🔥: Phát hiện có lửa (confidence: 0-100%)
- **NON-FIRE** ✅: Không có lửa

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **ML Model** | YOLO11 Classification |
| **WebSocket** | FastAPI WebSocket |
| **Image Processing** | OpenCV, NumPy |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Model Accuracy | 99.2% |
| Inference Time | ~10ms/image |
| WebSocket Latency | ~100ms/frame |
