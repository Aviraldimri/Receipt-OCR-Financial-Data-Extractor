"""
ocr.py — OCR Pipeline Module
==============================
Extracts text from preprocessed receipt images using EasyOCR.
Returns raw text with per-word confidence scores and bounding boxes.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


# Initialize EasyOCR reader (singleton to avoid re-loading model)
_reader: Optional[easyocr.Reader] = None


def get_reader(languages: List[str] = None) -> easyocr.Reader:
    """Get or create the EasyOCR reader (cached singleton)."""
    global _reader
    if languages is None:
        languages = ['en']
    if _reader is None:
        _reader = easyocr.Reader(languages, gpu=False)
    return _reader


def extract_text(image: np.ndarray, 
                 detail: int = 1,
                 paragraph: bool = False) -> List[Dict]:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image: Preprocessed image (grayscale or BGR).
        detail: 0 = text only, 1 = text + bounding box + confidence.
        paragraph: If True, merge nearby text into paragraphs.
    
    Returns:
        List of dictionaries, each containing:
        - 'text': Extracted text string
        - 'confidence': OCR confidence score (0–1)
        - 'bbox': Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    reader = get_reader()
    
    results = reader.readtext(
        image,
        detail=detail,
        paragraph=paragraph
    )
    
    extracted = []
    for result in results:
        if detail == 1:
            bbox, text, confidence = result
            extracted.append({
                "text": text.strip(),
                "confidence": float(confidence),
                "bbox": bbox
            })
        else:
            extracted.append({
                "text": result.strip(),
                "confidence": None,
                "bbox": None
            })
    
    return extracted


def get_full_text(ocr_results: List[Dict]) -> str:
    """Concatenate all OCR results into a single text string."""
    lines = [r["text"] for r in ocr_results if r["text"]]
    return "\n".join(lines)


def get_average_confidence(ocr_results: List[Dict]) -> float:
    """Calculate the average OCR confidence across all detected text."""
    confidences = [r["confidence"] for r in ocr_results if r["confidence"] is not None]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def draw_bounding_boxes(image: np.ndarray, 
                        ocr_results: List[Dict],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes around detected text on the image.
    
    Args:
        image: Original/processed image.
        ocr_results: Output from extract_text().
        color: BGR color for bounding boxes.
        thickness: Line thickness.
    
    Returns:
        Image with bounding boxes drawn.
    """
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    for result in ocr_results:
        bbox = result.get("bbox")
        if bbox is None:
            continue
        
        # Convert to integer coordinates
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Add text label with confidence
        text = result["text"]
        conf = result.get("confidence", 0)
        label = f"{text} ({conf:.2f})"
        
        # Position label above the bounding box
        x, y = int(pts[0][0]), int(pts[0][1]) - 5
        font_scale = 0.4
        cv2.putText(
            annotated, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, 1, cv2.LINE_AA
        )
    
    return annotated


def merge_into_lines(ocr_results: List[Dict], y_threshold: int = 15) -> List[Dict]:
    """
    Merge individual word detections into full text lines.
    
    EasyOCR often returns one bounding box per word. For receipt parsing,
    we need full lines (e.g., "Organic Bananas  $2.49" as one line).
    
    Algorithm:
    1. Compute the vertical center (y_center) of each detection's bbox.
    2. Group detections whose y_centers are within y_threshold pixels.
    3. Sort each group left-to-right by x_center.
    4. Concatenate the text within each group to form a line.
    
    Args:
        ocr_results: Raw OCR detections from extract_text().
        y_threshold: Max vertical distance (px) to consider same line.
    
    Returns:
        List of merged line dictionaries with {text, confidence, bbox}.
    """
    if not ocr_results:
        return []
    
    # Calculate center coordinates for each detection
    items = []
    for r in ocr_results:
        bbox = r.get("bbox")
        if bbox is None:
            continue
        pts = np.array(bbox)
        y_center = float(np.mean(pts[:, 1]))
        x_center = float(np.mean(pts[:, 0]))
        items.append({
            "text": r["text"],
            "confidence": r.get("confidence", 0),
            "bbox": bbox,
            "y_center": y_center,
            "x_center": x_center,
            "x_min": float(np.min(pts[:, 0])),
            "y_min": float(np.min(pts[:, 1])),
            "x_max": float(np.max(pts[:, 0])),
            "y_max": float(np.max(pts[:, 1])),
        })
    
    # Sort by y_center (top to bottom)
    items.sort(key=lambda it: it["y_center"])
    
    # Group into lines by y proximity
    lines = []
    current_line = [items[0]]
    
    for item in items[1:]:
        # Check if this item is on the same line as the current group
        avg_y = np.mean([it["y_center"] for it in current_line])
        if abs(item["y_center"] - avg_y) <= y_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)
    
    # Sort each line left-to-right, then merge
    merged = []
    for line_items in lines:
        line_items.sort(key=lambda it: it["x_center"])
        
        # Merge text
        line_text = " ".join(it["text"] for it in line_items)
        
        # Average confidence across words in this line
        confs = [it["confidence"] for it in line_items if it["confidence"] is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        # Compute encompassing bounding box
        all_x_min = min(it["x_min"] for it in line_items)
        all_y_min = min(it["y_min"] for it in line_items)
        all_x_max = max(it["x_max"] for it in line_items)
        all_y_max = max(it["y_max"] for it in line_items)
        merged_bbox = [
            [all_x_min, all_y_min],
            [all_x_max, all_y_min],
            [all_x_max, all_y_max],
            [all_x_min, all_y_max]
        ]
        
        merged.append({
            "text": line_text,
            "confidence": avg_conf,
            "bbox": merged_bbox
        })
    
    return merged


def run_ocr_pipeline(image: np.ndarray) -> Dict:
    """
    Complete OCR pipeline: extract text, merge into lines, compute stats.
    
    Returns:
        Dictionary with:
        - 'results': List of {text, confidence, bbox} (word-level)
        - 'lines': List of {text, confidence, bbox} (merged lines)
        - 'full_text': Concatenated text from merged lines
        - 'average_confidence': Mean confidence
        - 'num_detections': Count of text regions
    """
    results = extract_text(image)
    
    # Merge words into lines for better extraction
    lines = merge_into_lines(results)
    
    return {
        "results": results,
        "lines": lines,
        "full_text": "\n".join(l["text"] for l in lines if l["text"]),
        "average_confidence": get_average_confidence(results),
        "num_detections": len(results)
    }
