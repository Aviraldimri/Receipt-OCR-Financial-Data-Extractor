"""
preprocessing.py — Image Preprocessing Module
===============================================
Implements noise removal, skew correction, and contrast enhancement
for receipt images before OCR processing.

WHY EACH STEP IS USED:
1. Noise Removal (Gaussian / Median Blur):
   - Receipts often have speckle noise from scanning or camera capture.
   - Median filtering is especially effective for salt-and-pepper noise
     while preserving edges (text boundaries).
   - Gaussian blur smooths the image uniformly, reducing high-frequency noise.

2. Skew Correction (Hough Transform):
   - Receipts photographed by hand are rarely perfectly aligned.
   - Skewed text dramatically reduces OCR accuracy.
   - We detect dominant line angles via the Hough transform and rotate
     the image to correct alignment.

3. Contrast Enhancement (CLAHE):
   - Adaptive histogram equalization enhances local contrast, making
     faded or poorly-lit text more legible.
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) avoids
     over-amplifying noise by clipping the histogram.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Returns BGR image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ─────────────────────────────────────────────
# 1. NOISE REMOVAL
# ─────────────────────────────────────────────

def remove_noise_gaussian(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to remove high-frequency noise.
    Works well for general noise but may slightly blur text edges.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def remove_noise_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter — superior for salt-and-pepper noise.
    Preserves edges better than Gaussian, making it ideal for text images.
    """
    return cv2.medianBlur(image, kernel_size)


def remove_noise_bilateral(image: np.ndarray) -> np.ndarray:
    """
    Bilateral filter — removes noise while keeping edges sharp.
    Computationally heavier but produces best results for receipt text.
    """
    return cv2.bilateralFilter(image, 9, 75, 75)


# ─────────────────────────────────────────────
# 2. SKEW CORRECTION
# ─────────────────────────────────────────────

def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect the skew angle of the image using Hough Line Transform.
    Returns the median angle of detected lines.
    """
    gray = convert_to_grayscale(image)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (likely text lines)
        if abs(angle) < 45:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    return float(np.median(angles))


def correct_skew(image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Correct image skew by rotating around the center.
    Returns the corrected image and the angle applied.
    """
    if angle is None:
        angle = detect_skew_angle(image)
    
    if abs(angle) < 0.5:
        return image, 0.0  # No significant skew
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)
    
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated, angle


# ─────────────────────────────────────────────
# 3. CONTRAST ENHANCEMENT
# ─────────────────────────────────────────────

def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0,
                           tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    WHY CLAHE over standard histogram equalization:
    - Standard HE uses a global transformation → can wash out local details.
    - CLAHE divides the image into tiles and equalizes each independently,
      preserving local contrast variations critical for receipt text.
    - The clip limit prevents over-amplification of noise.
    """
    gray = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def enhance_contrast_histogram(image: np.ndarray) -> np.ndarray:
    """Standard histogram equalization — simpler but less adaptive."""
    gray = convert_to_grayscale(image)
    return cv2.equalizeHist(gray)


# ─────────────────────────────────────────────
# 4. BINARIZATION (Adaptive Thresholding)
# ─────────────────────────────────────────────

def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to create a binary image.
    This is the final step before OCR — it separates text from background.
    """
    gray = convert_to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )


# ─────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess_receipt(image_path: str, 
                       noise_method: str = "median",
                       enhance_method: str = "clahe",
                       do_skew_correction: bool = True,
                       do_binarize: bool = False) -> dict:
    """
    Full preprocessing pipeline for a receipt image.
    
    Args:
        image_path: Path to the receipt image.
        noise_method: 'gaussian', 'median', or 'bilateral'.
        enhance_method: 'clahe' or 'histogram'.
        do_skew_correction: Whether to detect and correct skew.
        do_binarize: Whether to apply adaptive thresholding.
    
    Returns:
        Dictionary with:
        - 'original': Original image
        - 'processed': Final processed image
        - 'grayscale': Grayscale version
        - 'skew_angle': Detected skew angle (if corrected)
        - 'steps': List of processing steps applied
    """
    steps = []
    
    # Load
    original = load_image(image_path)
    current = original.copy()
    steps.append("Loaded image")
    
    # Step 1: Noise removal
    if noise_method == "gaussian":
        current = remove_noise_gaussian(current)
        steps.append("Applied Gaussian blur (noise removal)")
    elif noise_method == "median":
        current = remove_noise_median(current)
        steps.append("Applied Median filter (noise removal)")
    elif noise_method == "bilateral":
        current = remove_noise_bilateral(current)
        steps.append("Applied Bilateral filter (noise removal)")
    
    # Step 2: Skew correction
    skew_angle = 0.0
    if do_skew_correction:
        current, skew_angle = correct_skew(current)
        if abs(skew_angle) > 0.5:
            steps.append(f"Corrected skew: {skew_angle:.2f}°")
        else:
            steps.append("No significant skew detected")
    
    # Step 3: Contrast enhancement
    if enhance_method == "clahe":
        enhanced = enhance_contrast_clahe(current)
        steps.append("Applied CLAHE contrast enhancement")
    elif enhance_method == "histogram":
        enhanced = enhance_contrast_histogram(current)
        steps.append("Applied histogram equalization")
    else:
        enhanced = convert_to_grayscale(current)
    
    # Step 4: Optional binarization
    if do_binarize:
        final = binarize_image(enhanced)
        steps.append("Applied adaptive thresholding (binarization)")
    else:
        final = enhanced
    
    return {
        "original": original,
        "processed": final,
        "color_processed": current,
        "grayscale": convert_to_grayscale(current),
        "skew_angle": skew_angle,
        "steps": steps
    }
