"""
confidence.py — Confidence Scoring & Reliability Module
========================================================
Implements multi-level confidence scoring:
  1. OCR-level confidence (from EasyOCR engine)
  2. Field-level confidence (composite of OCR + regex + keyword)
  3. Reliability flags for low-confidence fields
  4. Fallback logic for conflicting/missing data
"""

import re
from typing import Dict, List


# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.85
HIGH_CONFIDENCE_THRESHOLD = 0.95


def compute_ocr_confidence(ocr_results: List[Dict]) -> Dict:
    """
    Compute OCR-level confidence statistics.
    
    Returns:
        Dictionary with:
        - 'average': Mean confidence across all detections
        - 'min': Lowest confidence detection
        - 'max': Highest confidence detection
        - 'low_confidence_count': Number of detections below threshold
        - 'per_word': List of (text, confidence) tuples
    """
    confidences = [r["confidence"] for r in ocr_results if r["confidence"] is not None]
    
    if not confidences:
        return {
            "average": 0.0,
            "min": 0.0,
            "max": 0.0,
            "low_confidence_count": 0,
            "per_word": []
        }
    
    low_conf_words = [
        {"text": r["text"], "confidence": r["confidence"]}
        for r in ocr_results
        if r["confidence"] is not None and r["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]
    
    return {
        "average": round(sum(confidences) / len(confidences), 4),
        "min": round(min(confidences), 4),
        "max": round(max(confidences), 4),
        "low_confidence_count": len(low_conf_words),
        "per_word": [
            {"text": r["text"], "confidence": round(r["confidence"], 4)}
            for r in ocr_results if r["confidence"] is not None
        ]
    }


def compute_field_confidence(extracted_data: Dict, ocr_confidence: Dict) -> Dict:
    """
    Compute field-level composite confidence scores.
    
    Composite score formula:
        field_confidence = 0.4 * ocr_confidence + 0.3 * regex_score + 0.3 * keyword_score
    
    Returns:
        Dictionary with composite confidence for each field.
    """
    ocr_avg = ocr_confidence.get("average", 0.5)
    
    field_scores = {}
    
    for field_name in ["store_name", "date", "items", "total_amount"]:
        field_data = extracted_data.get(field_name, {})
        base_confidence = field_data.get("confidence", 0.0)
        value = field_data.get("value")
        
        # No value → zero confidence
        if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
            field_scores[field_name] = {
                "value": value,
                "confidence": 0.0,
                "reliability": "missing",
                "flagged": True,
                "reason": "Field could not be extracted"
            }
            continue
        
        regex_score = estimate_regex_score(field_name, value)
        keyword_score = estimate_keyword_score(field_name, value)

        # Blend OCR signal + format validation + keyword/semantic plausibility.
        # We still fold in extraction confidence as a stabilizer.
        composite = (
            (0.4 * ocr_avg) +
            (0.3 * regex_score) +
            (0.3 * keyword_score)
        ) * 0.7 + (base_confidence * 0.3)
        composite = round(max(0.0, min(1.0, composite)), 4)
        
        # Determine reliability level
        if composite >= HIGH_CONFIDENCE_THRESHOLD:
            reliability = "high"
        elif composite >= MEDIUM_CONFIDENCE_THRESHOLD:
            reliability = "medium"
        elif composite >= LOW_CONFIDENCE_THRESHOLD:
            reliability = "acceptable"
        else:
            reliability = "low"
        
        flagged = composite < LOW_CONFIDENCE_THRESHOLD
        reason = None
        if flagged:
            reason = f"Confidence {composite:.2f} is below threshold {LOW_CONFIDENCE_THRESHOLD}"
        
        field_scores[field_name] = {
            "value": value,
            "confidence": composite,
            "reliability": reliability,
            "flagged": flagged,
            "reason": reason
        }
    
    return field_scores


def estimate_regex_score(field_name: str, value) -> float:
    """Estimate format-validity score using regex-style validation."""
    if field_name == "date":
        if not isinstance(value, str):
            return 0.0
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", value):
            return 1.0
        if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", value):
            return 1.0
        return 0.4

    if field_name == "total_amount":
        if not isinstance(value, str):
            return 0.0
        return 1.0 if re.fullmatch(r"\d+\.\d{2}", value.strip()) else 0.5

    if field_name == "items":
        if not isinstance(value, list) or not value:
            return 0.0
        valid = 0
        for item in value:
            if (
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and re.fullmatch(r"\d+\.\d{2}", str(item.get("price", "")).strip())
            ):
                valid += 1
        return valid / len(value)

    if field_name == "store_name":
        if not isinstance(value, str):
            return 0.0
        return 1.0 if len(value.strip()) >= 3 else 0.5

    return 0.5


def estimate_keyword_score(field_name: str, value) -> float:
    """Estimate semantic plausibility of extracted field content."""
    if field_name == "store_name":
        if not isinstance(value, str):
            return 0.0
        # Penalize likely non-store tokens.
        lower = value.lower()
        bad_tokens = ["date", "total", "tax", "receipt"]
        return 0.6 if any(tok in lower for tok in bad_tokens) else 1.0

    if field_name == "date":
        return 1.0 if isinstance(value, str) and len(value.strip()) >= 6 else 0.4

    if field_name == "items":
        if not isinstance(value, list):
            return 0.0
        if len(value) == 0:
            return 0.0
        if len(value) <= 2:
            return 0.7
        return 1.0

    if field_name == "total_amount":
        if not isinstance(value, str):
            return 0.0
        try:
            amount = float(value)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if amount > 0 else 0.2

    return 0.5


def apply_reliability_handling(field_scores: Dict) -> Dict:
    """
    Apply reliability handling:
    - Flag fields with confidence < 0.7
    - Add warnings for potentially incorrect data
    - Provide fallback suggestions
    
    Returns:
        Enhanced field scores with reliability metadata.
    """
    warnings = []
    
    for field_name, field_data in field_scores.items():
        if field_data.get("flagged", False):
            warnings.append({
                "field": field_name,
                "confidence": field_data["confidence"],
                "message": f"'{field_name}' has low confidence ({field_data['confidence']:.2f}). "
                          f"Manual verification recommended.",
                "suggestion": get_fallback_suggestion(field_name)
            })
    
    return {
        "fields": field_scores,
        "warnings": warnings,
        "overall_reliability": compute_overall_reliability(field_scores)
    }


def get_fallback_suggestion(field_name: str) -> str:
    """Provide fallback/recovery suggestions for low-confidence fields."""
    suggestions = {
        "store_name": "Try using the address or logo area to identify the store. "
                     "Consider re-scanning with better lighting.",
        "date": "Look for alternative date formats or timestamps on the receipt. "
               "Check for 'Date:' labels in different positions.",
        "items": "Some items may have been misread. Verify quantities and names "
                "against the total amount for consistency.",
        "total_amount": "Cross-check by summing individual items. Look for "
                       "'Amount Due' or 'Balance' as alternative total indicators."
    }
    return suggestions.get(field_name, "Manual verification recommended.")


def compute_overall_reliability(field_scores: Dict) -> Dict:
    """
    Compute overall receipt reliability score.
    
    Returns:
        Dictionary with overall score and assessment.
    """
    confidences = [
        f["confidence"] for f in field_scores.values()
        if f["confidence"] > 0
    ]
    
    if not confidences:
        return {
            "score": 0.0,
            "assessment": "Unable to extract any fields",
            "grade": "F"
        }
    
    avg = sum(confidences) / len(confidences)
    flagged_count = sum(1 for f in field_scores.values() if f.get("flagged", False))
    total_fields = len(field_scores)
    
    # Grade based on average confidence and flagged fields ratio
    if avg >= 0.9 and flagged_count == 0:
        grade = "A"
        assessment = "High confidence — extraction is reliable"
    elif avg >= 0.8 and flagged_count <= 1:
        grade = "B"
        assessment = "Good confidence — minor uncertainties"
    elif avg >= 0.7:
        grade = "C"
        assessment = "Moderate confidence — some fields need verification"
    elif avg >= 0.5:
        grade = "D"
        assessment = "Low confidence — manual review recommended"
    else:
        grade = "F"
        assessment = "Very low confidence — image quality may be too poor"
    
    return {
        "score": round(avg, 4),
        "assessment": assessment,
        "grade": grade,
        "flagged_fields": flagged_count,
        "total_fields": total_fields
    }


def build_confidence_report(extracted_data: Dict, ocr_results: List[Dict]) -> Dict:
    """
    Build a complete confidence report for a receipt.
    
    Args:
        extracted_data: Output from extractor.extract_all()
        ocr_results: Raw OCR results from ocr.py
    
    Returns:
        Complete confidence report dictionary.
    """
    # OCR-level confidence
    ocr_conf = compute_ocr_confidence(ocr_results)
    
    # Field-level confidence
    field_conf = compute_field_confidence(extracted_data, ocr_conf)
    
    # Reliability handling
    reliability = apply_reliability_handling(field_conf)
    
    return {
        "ocr_confidence": ocr_conf,
        "field_confidence": reliability["fields"],
        "warnings": reliability["warnings"],
        "overall_reliability": reliability["overall_reliability"]
    }
