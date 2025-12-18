# Image Preprocessing Service
# Các kỹ thuật xử lý ảnh đầu vào để cải thiện chất lượng detection
import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Lớp tiền xử lý ảnh sử dụng các kỹ thuật xử lý ảnh cổ điển.
    Áp dụng trước khi đưa ảnh vào mô hình AI để cải thiện chất lượng.
    """
    
    def __init__(
        self,
        denoise_strength: int = 10,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        brightness_alpha: float = 1.0,
        brightness_beta: int = 0,
        contrast_enhancement: bool = True,
        noise_reduction: bool = True,
        white_balance: bool = True
    ):
        """
        Khởi tạo bộ tiền xử lý ảnh.
        
        Args:
            denoise_strength: Cường độ khử nhiễu (1-30, cao hơn = khử nhiều hơn)
            clahe_clip_limit: Giới hạn clip cho CLAHE (1.0-4.0)
            clahe_grid_size: Kích thước lưới cho CLAHE
            brightness_alpha: Hệ số điều chỉnh độ sáng (1.0 = không đổi)
            brightness_beta: Độ lệch brightness (-100 đến 100)
            contrast_enhancement: Bật/tắt cải thiện contrast
            noise_reduction: Bật/tắt khử nhiễu
            white_balance: Bật/tắt cân bằng trắng
        """
        self.denoise_strength = denoise_strength
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.brightness_alpha = brightness_alpha
        self.brightness_beta = brightness_beta
        self.contrast_enhancement = contrast_enhancement
        self.noise_reduction = noise_reduction
        self.white_balance = white_balance
        
        # Khởi tạo CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Khử nhiễu ảnh sử dụng Non-Local Means Denoising.
        
        Thuật toán:
        - Sử dụng cv2.fastNlMeansDenoisingColored cho ảnh màu
        - Tìm các patch tương tự trong ảnh và lấy trung bình
        - Giữ được chi tiết cạnh tốt hơn so với Gaussian blur
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã khử nhiễu
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.denoise_strength,
            hForColorComponents=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Cải thiện contrast sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Thuật toán:
        - Chuyển ảnh sang không gian màu LAB
        - Áp dụng CLAHE chỉ trên kênh L (Lightness)
        - Giữ nguyên màu sắc (kênh A và B)
        - Chuyển về không gian BGR
        
        Ưu điểm so với Histogram Equalization thông thường:
        - Tránh over-amplification của noise
        - Tăng cường local contrast
        - Phù hợp cho ảnh có vùng sáng/tối không đều
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã cải thiện contrast
        """
        # Chuyển sang không gian LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Tách các kênh
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE trên kênh L
        l_enhanced = self._clahe.apply(l)
        
        # Ghép lại
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Chuyển về BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def adjust_brightness_contrast(
        self, 
        image: np.ndarray,
        alpha: Optional[float] = None,
        beta: Optional[int] = None
    ) -> np.ndarray:
        """
        Điều chỉnh độ sáng và contrast.
        
        Công thức: output = alpha * input + beta
        - alpha > 1: Tăng contrast
        - alpha < 1: Giảm contrast  
        - beta > 0: Tăng độ sáng
        - beta < 0: Giảm độ sáng
        
        Args:
            image: Ảnh BGR đầu vào
            alpha: Hệ số contrast (nếu None, dùng giá trị mặc định)
            beta: Độ lệch brightness (nếu None, dùng giá trị mặc định)
            
        Returns:
            Ảnh đã điều chỉnh
        """
        if alpha is None:
            alpha = self.brightness_alpha
        if beta is None:
            beta = self.brightness_beta
            
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Cân bằng trắng tự động sử dụng thuật toán Gray World.
        
        Thuật toán Gray World:
        - Giả định rằng trung bình của tất cả màu trong ảnh nên là gray
        - Tính trung bình của mỗi kênh màu
        - Scale mỗi kênh để đạt được cân bằng
        
        Ứng dụng:
        - Hiệu chỉnh ảnh chụp dưới ánh sáng không đều
        - Cải thiện màu sắc tự nhiên
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã cân bằng trắng
        """
        result = image.copy().astype(np.float32)
        
        # Tính trung bình của mỗi kênh
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Tính trung bình tổng
        avg = (avg_b + avg_g + avg_r) / 3
        
        # Scale mỗi kênh
        if avg_b > 0:
            result[:, :, 0] = result[:, :, 0] * (avg / avg_b)
        if avg_g > 0:
            result[:, :, 1] = result[:, :, 1] * (avg / avg_g)
        if avg_r > 0:
            result[:, :, 2] = result[:, :, 2] * (avg / avg_r)
        
        # Clip về [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Hiệu chỉnh gamma để điều chỉnh độ sáng phi tuyến.
        
        Công thức: output = 255 * (input/255)^(1/gamma)
        - gamma > 1: Làm tối ảnh
        - gamma < 1: Làm sáng ảnh
        - gamma = 1: Không thay đổi
        
        Ứng dụng:
        - Hiệu chỉnh ảnh quá tối hoặc quá sáng
        - Cải thiện chi tiết trong vùng tối
        
        Args:
            image: Ảnh BGR đầu vào
            gamma: Hệ số gamma (0.1 - 3.0)
            
        Returns:
            Ảnh đã hiệu chỉnh gamma
        """
        # Tạo lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def unsharp_mask(
        self, 
        image: np.ndarray, 
        sigma: float = 1.0, 
        strength: float = 1.5
    ) -> np.ndarray:
        """
        Làm sắc nét ảnh sử dụng Unsharp Masking.
        
        Thuật toán:
        1. Làm mờ ảnh gốc bằng Gaussian blur
        2. Trừ ảnh mờ khỏi ảnh gốc để tạo mask
        3. Cộng mask vào ảnh gốc với hệ số strength
        
        Công thức: sharpened = original + strength * (original - blurred)
        
        Args:
            image: Ảnh BGR đầu vào
            sigma: Độ lệch chuẩn của Gaussian (1.0 - 3.0)
            strength: Cường độ làm sắc (0.5 - 3.0)
            
        Returns:
            Ảnh đã làm sắc nét
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        return sharpened
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Cân bằng histogram để cải thiện contrast toàn cục.
        
        Thuật toán:
        - Chuyển sang YUV (Y = luminance)
        - Cân bằng histogram trên kênh Y
        - Chuyển về BGR
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã cân bằng histogram
        """
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def bilateral_filter(
        self, 
        image: np.ndarray, 
        d: int = 9, 
        sigma_color: float = 75.0,
        sigma_space: float = 75.0
    ) -> np.ndarray:
        """
        Bộ lọc bilateral - khử nhiễu mà giữ cạnh.
        
        Thuật toán:
        - Kết hợp domain filter và range filter
        - Domain: Dựa trên khoảng cách không gian
        - Range: Dựa trên sự khác biệt intensity
        
        Ưu điểm:
        - Giữ được cạnh sắc nét
        - Làm mịn vùng đồng nhất
        
        Args:
            image: Ảnh BGR đầu vào
            d: Đường kính của vùng lân cận
            sigma_color: Sigma cho range filter
            sigma_space: Sigma cho domain filter
            
        Returns:
            Ảnh đã lọc
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline tiền xử lý hoàn chỉnh.
        
        Thứ tự xử lý:
        1. Cân bằng trắng (nếu bật)
        2. Khử nhiễu (nếu bật)
        3. Cải thiện contrast bằng CLAHE (nếu bật)
        4. Điều chỉnh brightness/contrast
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã tiền xử lý
        """
        result = image.copy()
        
        # 1. Cân bằng trắng
        if self.white_balance:
            result = self.auto_white_balance(result)
        
        # 2. Khử nhiễu
        if self.noise_reduction:
            result = self.denoise(result)
        
        # 3. Cải thiện contrast bằng CLAHE
        if self.contrast_enhancement:
            result = self.enhance_contrast_clahe(result)
        
        # 4. Điều chỉnh brightness/contrast
        if self.brightness_alpha != 1.0 or self.brightness_beta != 0:
            result = self.adjust_brightness_contrast(result)
        
        return result
    
    def process_for_fire_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline tối ưu cho phát hiện lửa.
        
        Các bước:
        1. Bilateral filter để khử nhiễu giữ cạnh
        2. Tăng saturation để làm nổi bật màu lửa
        3. CLAHE để cải thiện contrast
        
        Args:
            image: Ảnh BGR đầu vào
            
        Returns:
            Ảnh đã tiền xử lý cho fire detection
        """
        # 1. Bilateral filter
        result = self.bilateral_filter(image)
        
        # 2. Tăng saturation trong không gian HSV
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # Tăng 30% saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. CLAHE
        result = self.enhance_contrast_clahe(result)
        
        return result


# Singleton instance
image_preprocessor = ImagePreprocessor()
