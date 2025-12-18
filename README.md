# 🔥 Forest Fire Detection App

Ứng dụng phát hiện cháy rừng sử dụng model YOLO11 Classification.

## 📂 Project Structure

```
app/
├── backend/
│   ├── routes/detection.py      # API endpoints
│   ├── schemas/detection.py     # Pydantic models
│   ├── services/detection.py    # YOLO11 inference
│   ├── main.py                  # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit UI
│   └── requirements.txt
└── weights/
    └── best.pt                  # Trained YOLO11 model
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Backend
cd app/backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 2. Run Backend (FastAPI)

```bash
cd app/backend
uvicorn main:app --reload --port 8000
```

API sẽ chạy tại: http://localhost:8000

### 3. Run Frontend (Streamlit)

```bash
cd app/frontend
streamlit run app.py
```

Frontend sẽ mở tại: http://localhost:8501

## 📸 Usage

1. **Image Detection**: Upload ảnh JPG/PNG → Nhận kết quả FIRE/NON-FIRE
2. **Video Detection**: Upload video MP4/AVI → Phân tích từng frame

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect/image` | Detect fire in image |
| POST | `/api/detect/video` | Detect fire in video |

## 📊 Output

- **FIRE** 🔥: Phát hiện có lửa
- **NON-FIRE** ✅: Không có lửa
