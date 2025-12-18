# 🔥 BÁO CÁO ĐỀ TÀI: HỆ THỐNG PHÁT HIỆN CHÁY RỪNG
## Môn học: Xử lý Ảnh và Video (IVP501)

---

## 📋 THÔNG TIN ĐỀ TÀI

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên đề tài** | Hệ thống phát hiện cháy rừng thời gian thực sử dụng Deep Learning |
| **Mô hình** | YOLOv11 Classification |
| **Framework** | Ultralytics, FastAPI, Streamlit |
| **Ngôn ngữ** | Python 3.11 |

---

## 📖 MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Xử lý ảnh đầu vào (Preprocessing)](#3-xử-lý-ảnh-đầu-vào-preprocessing)
4. [Data Augmentation trong Training](#4-data-augmentation-trong-training)
5. [Mô hình YOLO11 Classification](#5-mô-hình-yolo11-classification)
6. [Kết quả Training](#6-kết-quả-training)
7. [Ứng dụng Web](#7-ứng-dụng-web)
8. [Kết luận](#8-kết-luận)

---

## 1. GIỚI THIỆU

### 1.1 Bài toán
Phát hiện cháy rừng sớm là một vấn đề quan trọng trong việc bảo vệ môi trường và tài sản. Hệ thống này sử dụng **Computer Vision** và **Deep Learning** để phân loại ảnh/video thành hai lớp:
- **FIRE**: Có lửa/cháy
- **NON-FIRE**: Không có lửa

### 1.2 Mục tiêu
- Xây dựng pipeline xử lý ảnh hoàn chỉnh
- Train mô hình YOLO11 Classification đạt accuracy > 95%
- Phát triển ứng dụng web demo real-time
- Tối ưu hiệu năng với WebSocket streaming

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│    │  Image   │    │  Video   │    │  Camera  │                │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘                │
│         └───────────────┼───────────────┘                       │
│                         ▼                                        │
│    ┌─────────────────────────────────────────────────────┐      │
│    │           PREPROCESSING MODULE                       │      │
│    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │      │
│    │  │ Denoise     │→ │ White       │→ │ CLAHE       │ │      │
│    │  │ (NLMeans)   │  │ Balance     │  │ Contrast    │ │      │
│    │  └─────────────┘  └─────────────┘  └─────────────┘ │      │
│    └─────────────────────────┬───────────────────────────┘      │
│                              ▼                                   │
│    ┌─────────────────────────────────────────────────────┐      │
│    │              YOLO11 CLASSIFICATION                    │      │
│    │                                                       │      │
│    │  Input: 224x224 RGB                                  │      │
│    │  Architecture: 47 layers, 1.5M params               │      │
│    │  Output: [FIRE, NON-FIRE] + Confidence              │      │
│    └─────────────────────────┬───────────────────────────┘      │
│                              ▼                                   │
│    ┌─────────────────────────────────────────────────────┐      │
│    │                    OUTPUT                             │      │
│    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │      │
│    │  │ Prediction  │  │ Confidence  │  │ Alert       │ │      │
│    │  │ FIRE/NOFIRE │  │ 0.0 - 1.0   │  │ System      │ │      │
│    │  └─────────────┘  └─────────────┘  └─────────────┘ │      │
│    └─────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục

```
project/
├── app/
│   ├── backend/                 # FastAPI Server
│   │   ├── routes/
│   │   │   ├── detection.py     # API phát hiện ảnh/video
│   │   │   └── websocket.py     # WebSocket streaming
│   │   ├── services/
│   │   │   ├── detection.py     # YOLO inference
│   │   │   ├── preprocessing.py # Tiền xử lý ảnh
│   │   │   └── websocket_stream.py
│   │   └── main.py
│   ├── frontend/                # Streamlit UI
│   │   └── app.py
│   └── weights/
│       └── best.pt              # Model weights
├── MyFireProject/               # Training results
│   └── yolo11n_fire_run5/
│       ├── args.yaml            # Training config
│       ├── weights/best.pt      # Best model
│       └── results.csv          # Training metrics
├── processed_dataset/           # Dataset
│   ├── train/                   # 4000 images
│   ├── val/                     # 1000 images
│   └── test/                    # 50 images
└── REPORT.md                    # File này
```

---

## 3. XỬ LÝ ẢNH ĐẦU VÀO (PREPROCESSING)

### 3.1 Pipeline Tiền Xử Lý

```
Input Image → White Balance → Denoise → CLAHE → Output
```

### 3.2 Các kỹ thuật được sử dụng

#### 3.2.1 Khử nhiễu - Non-Local Means Denoising

**Công thức toán học:**

$$\hat{I}(x) = \frac{1}{C(x)} \sum_{y \in \Omega} w(x,y) \cdot I(y)$$

Trong đó:
- $\hat{I}(x)$: Giá trị pixel sau khi khử nhiễu
- $I(y)$: Giá trị pixel gốc
- $w(x,y)$: Trọng số dựa trên độ tương đồng của patch
- $C(x)$: Hệ số chuẩn hóa

**Trọng số được tính:**

$$w(x,y) = e^{-\frac{\|P(x) - P(y)\|^2}{h^2}}$$

- $P(x), P(y)$: Các patch xung quanh pixel x và y
- $h$: Tham số điều khiển cường độ khử nhiễu

**Code implementation:**
```python
def denoise(self, image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=self.denoise_strength,           # h = 10 (default)
        hForColorComponents=self.denoise_strength,
        templateWindowSize=7,               # Kích thước patch
        searchWindowSize=21                 # Vùng tìm kiếm
    )
```

**Ưu điểm:**
- Giữ được cạnh sắc nét
- Hiệu quả với nhiễu Gaussian
- Phù hợp cho ảnh tự nhiên

---

#### 3.2.2 Cân bằng trắng - Gray World Algorithm

**Giả thiết:** Trung bình của tất cả màu trong ảnh tự nhiên nên là màu xám trung tính.

**Thuật toán:**

1. Tính trung bình của mỗi kênh màu:
   - $\mu_R = \frac{1}{N}\sum R_i$
   - $\mu_G = \frac{1}{N}\sum G_i$
   - $\mu_B = \frac{1}{N}\sum B_i$

2. Tính trung bình tổng: $\mu = \frac{\mu_R + \mu_G + \mu_B}{3}$

3. Scale mỗi kênh:
   - $R' = R \times \frac{\mu}{\mu_R}$
   - $G' = G \times \frac{\mu}{\mu_G}$
   - $B' = B \times \frac{\mu}{\mu_B}$

**Code implementation:**
```python
def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
    result = image.copy().astype(np.float32)
    
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    avg = (avg_b + avg_g + avg_r) / 3
    
    result[:, :, 0] = result[:, :, 0] * (avg / avg_b)
    result[:, :, 1] = result[:, :, 1] * (avg / avg_g)
    result[:, :, 2] = result[:, :, 2] * (avg / avg_r)
    
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

#### 3.2.3 CLAHE - Contrast Limited Adaptive Histogram Equalization

**Vấn đề với Histogram Equalization thông thường:**
- Tăng nhiễu trong vùng đồng nhất
- Không thích ứng với các vùng khác nhau của ảnh

**Giải pháp CLAHE:**

1. **Chia ảnh thành các tile** (8x8 grid mặc định)
2. **Áp dụng histogram equalization** cho từng tile
3. **Giới hạn contrast** (clip limit) để tránh over-amplification
4. **Bilinear interpolation** để loại bỏ artifacts ở biên

**Công thức Histogram Equalization:**

$$s_k = (L-1) \sum_{j=0}^{k} p_r(r_j)$$

Trong đó:
- $s_k$: Giá trị output
- $L$: Số mức xám (256)
- $p_r(r_j)$: Xác suất của mức xám $r_j$

**Code implementation:**
```python
def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
    # Chuyển sang không gian LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Tách các kênh L, A, B
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE chỉ trên kênh L (Lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Ghép lại và chuyển về BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
```

**Tham số:**
- `clipLimit = 2.0`: Giới hạn tăng contrast
- `tileGridSize = (8, 8)`: Số tile theo mỗi chiều

---

#### 3.2.4 Bilateral Filter (Khử nhiễu giữ cạnh)

**Công thức:**

$$I^{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \cdot f_r(\|I(x_i) - I(x)\|) \cdot g_s(\|x_i - x\|)$$

Trong đó:
- $f_r$: Range filter (domain of intensity)
- $g_s$: Spatial filter (domain of space)
- $W_p$: Normalization factor

**Đặc điểm:**
- Kết hợp **domain filter** và **range filter**
- Làm mịn vùng đồng nhất
- Giữ nguyên cạnh sắc

---

## 4. DATA AUGMENTATION TRONG TRAINING

### 4.1 Cấu hình Training (args.yaml)

Dựa trên file `MyFireProject/yolo11n_fire_run5/args.yaml`:

```yaml
# === CẤU HÌNH CƠ BẢN ===
task: classify
model: yolo11n-cls.pt
epochs: 30
batch: 32
imgsz: 224                # Kích thước ảnh đầu vào

# === AUGMENTATION PARAMETERS ===
hsv_h: 0.015              # Biến đổi Hue (±1.5%)
hsv_s: 0.7                # Biến đổi Saturation (±70%)
hsv_v: 0.4                # Biến đổi Value (±40%)
degrees: 0.0              # Xoay ảnh (độ)
translate: 0.1            # Dịch chuyển (±10%)
scale: 0.5                # Scale (±50%)
shear: 0.0                # Biến dạng shear
perspective: 0.0          # Biến dạng phối cảnh
flipud: 0.0               # Lật dọc (0%)
fliplr: 0.5               # Lật ngang (50%)
mosaic: 1.0               # Mosaic augmentation (100%)
mixup: 0.0                # MixUp augmentation
auto_augment: randaugment # Auto augmentation strategy
erasing: 0.4              # Random erasing (40%)
```

### 4.2 Chi tiết các kỹ thuật Augmentation

#### 4.2.1 HSV Augmentation

**Mục đích:** Thay đổi màu sắc để model robust với điều kiện ánh sáng khác nhau.

**Công thức:**
- **Hue shift:** $H' = (H + \Delta h \times 180) \mod 180$, với $\Delta h \in [-0.015, 0.015]$
- **Saturation:** $S' = S \times (1 + \Delta s)$, với $\Delta s \in [-0.7, 0.7]$
- **Value:** $V' = V \times (1 + \Delta v)$, với $\Delta v \in [-0.4, 0.4]$

**Ý nghĩa cho Fire Detection:**
- Lửa có màu sắc đa dạng (cam, đỏ, vàng)
- Biến đổi HSV giúp model nhận diện các màu lửa khác nhau
- `hsv_v: 0.4` quan trọng vì lửa có thể sáng/tối khác nhau

#### 4.2.2 Geometric Augmentation

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `translate` | 0.1 | Dịch chuyển ±10% theo mỗi chiều |
| `scale` | 0.5 | Scale từ 50% đến 150% |
| `fliplr` | 0.5 | 50% xác suất lật ngang |

**Lý do chọn:**
- `scale: 0.5` quan trọng vì lửa có thể xuất hiện ở nhiều kích thước
- `fliplr: 0.5` tăng đa dạng dữ liệu
- `degrees: 0.0` vì lửa thường hướng lên trên

#### 4.2.3 Mosaic Augmentation

**Cách hoạt động:**
1. Lấy 4 ảnh ngẫu nhiên từ dataset
2. Ghép thành 1 ảnh 2x2
3. Random crop để có kích thước mong muốn

**Ưu điểm:**
- Tăng context đa dạng
- Hiệu quả batch normalization (nhiều object trong 1 ảnh)
- Giảm overfitting

**Cấu hình:** `mosaic: 1.0` → 100% batch sử dụng mosaic

#### 4.2.4 RandAugment

**`auto_augment: randaugment`** áp dụng chuỗi các augmentation ngẫu nhiên.

**Các phép biến đổi có thể:**
- AutoContrast
- Equalize
- Invert
- Rotate
- Posterize
- Solarize
- Color jittering
- Sharpness

#### 4.2.5 Random Erasing

**`erasing: 0.4`** → 40% ảnh bị xóa ngẫu nhiên một phần.

**Công dụng:**
- Giúp model không phụ thuộc vào một vùng cụ thể
- Tăng robustness với occlusion
- Regularization effect

---

## 5. MÔ HÌNH YOLO11 CLASSIFICATION

### 5.1 Kiến trúc

```
YOLO11n-cls Architecture:
──────────────────────────────────────────────────────────────
Layer    From   Params   Module                      Arguments
──────────────────────────────────────────────────────────────
  0        -1      464   Conv                        [3, 16, 3, 2]
  1        -1    4,672   Conv                        [16, 32, 3, 2]
  2        -1    6,640   C3k2                        [32, 64, 1, False, 0.25]
  3        -1   36,992   Conv                        [64, 64, 3, 2]
  4        -1   26,080   C3k2                        [64, 128, 1, False, 0.25]
  5        -1  147,712   Conv                        [128, 128, 3, 2]
  6        -1   87,040   C3k2                        [128, 128, 1, True]
  7        -1  295,424   Conv                        [128, 256, 3, 2]
  8        -1  346,112   C3k2                        [256, 256, 1, True]
  9        -1  249,728   C2PSA                       [256, 256, 1]
 10        -1  332,802   Classify                    [256, 2]
──────────────────────────────────────────────────────────────
Total: 47 layers, 1,533,666 parameters, 3.3 GFLOPs
```

### 5.2 Các thành phần chính

#### Conv Layer
- Convolution + BatchNorm + SiLU activation
- Stride 2 để downsampling

#### C3k2 Block
- CSP Bottleneck với 2 convolutions
- Feature reuse và gradient flow

#### C2PSA (Polarized Self-Attention)
- Self-attention mechanism
- Capture long-range dependencies

#### Classify Head
- Global Average Pooling
- Linear layer: 256 → 2 classes

### 5.3 Hyperparameters Training

```yaml
# Optimizer
optimizer: auto         # Tự động chọn (AdamW)
lr0: 0.01               # Learning rate ban đầu
lrf: 0.01               # Final LR = lr0 * lrf
momentum: 0.937         # Momentum for SGD
weight_decay: 0.0005    # L2 regularization

# Training
epochs: 30
batch: 32
warmup_epochs: 3.0
warmup_momentum: 0.8
amp: true               # Automatic Mixed Precision
```

---

## 6. KẾT QUẢ TRAINING

### 6.1 Dataset

| Tập | Số ảnh | Fire | Non-Fire |
|-----|--------|------|----------|
| Train | 3,680* | ~1,840 | ~1,840 |
| Val | 929* | ~465 | ~465 |
| Test | 50 | 25 | 25 |

*Sau khi loại bỏ ảnh lỗi (320 corrupt trong train, 71 corrupt trong val)

### 6.2 Training Progress

| Epoch | Loss | Top-1 Accuracy |
|-------|------|----------------|
| 1 | 0.225 | 95.9% |
| 5 | 0.144 | 97.4% |
| 10 | 0.090 | 98.1% |
| 15 | 0.057 | 98.4% |
| 20 | 0.040 | 98.5% |
| 23 | 0.035 | **99.2%** |
| 30 | 0.019 | 98.9% |

### 6.3 Kết quả cuối cùng

| Metric | Validation | Test |
|--------|------------|------|
| **Top-1 Accuracy** | 99.2% | 98.0% |
| **Top-5 Accuracy** | 100% | 100% |
| **Inference Time** | 0.6ms | 10.6ms |

### 6.4 Training Time

- **Total:** 0.116 hours (~7 minutes)
- **GPU:** NVIDIA GeForce RTX 3070 Laptop (8GB)
- **Speed:** ~12s per epoch

---

## 7. ỨNG DỤNG WEB

### 7.1 Backend API (FastAPI)

**Endpoints:**

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/health` | Kiểm tra trạng thái |
| POST | `/api/detect/image` | Phát hiện trong ảnh |
| POST | `/api/detect/video` | Phát hiện trong video |
| WS | `/api/ws/stream` | WebSocket streaming |

**Ví dụ Response:**
```json
{
    "prediction": "FIRE",
    "confidence": 0.9856,
    "processing_time": 0.012
}
```

### 7.2 Frontend (Streamlit)

**Chức năng:**
- Upload ảnh/video
- Hiển thị kết quả real-time
- WebSocket streaming cho video
- Dashboard thống kê

### 7.3 WebSocket Streaming

**Lợi ích so với HTTP:**
- Độ trễ thấp (~100ms/frame vs hàng giây)
- Persistent connection
- Bidirectional communication

**Protocol:**
```json
// Client → Server
{"type": "start", "video_path": "path/to/video.mp4"}
{"type": "frame", "data": "base64_image"}
{"type": "stop"}

// Server → Client  
{"type": "frame", "prediction": "FIRE", "confidence": 0.95}
{"type": "complete", "total_frames": 100, "fire_frames": 23}
```

---

## 8. KẾT LUẬN

### 8.1 Đạt được

✅ Xây dựng pipeline xử lý ảnh hoàn chỉnh
- Khử nhiễu với Non-Local Means
- Cân bằng trắng với Gray World
- Tăng contrast với CLAHE

✅ Train model YOLO11 đạt **99.2% accuracy**
- Sử dụng data augmentation hiệu quả
- Training nhanh (~7 phút)

✅ Ứng dụng web real-time
- FastAPI backend
- Streamlit frontend
- WebSocket streaming

### 8.2 Hạn chế

⚠️ Dataset có nhiều ảnh lỗi (~8%)
⚠️ Chưa test với video thực tế
⚠️ Chưa có smoke detection riêng

### 8.3 Hướng phát triển

- Mở rộng sang detection (bounding box)
- Thêm class Smoke
- Tích hợp camera thực
- Deploy lên cloud

---

## 📚 TÀI LIỆU THAM KHẢO

1. Ultralytics YOLO Documentation: https://docs.ultralytics.com
2. OpenCV Documentation: https://docs.opencv.org
3. "Non-Local Means Denoising" - Buades et al., 2005
4. "CLAHE" - Zuiderveld, 1994
5. "RandAugment" - Cubuk et al., 2020

---

*Report được tạo tự động bởi hệ thống*
