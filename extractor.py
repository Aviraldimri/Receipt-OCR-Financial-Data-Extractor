"""
extractor.py — Key Information Extraction Module
==================================================
Extracts structured fields from raw OCR text using regex patterns,
keyword heuristics, and positional logic.

Fields extracted:
  - Store/Vendor Name
  - Date
  - Items list with prices
  - Total amount
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# Date patterns (handles multiple formats)
DATE_PATTERNS = [
    # MM/DD/YYYY or MM-DD-YYYY
    r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
    # YYYY-MM-DD
    r'(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})',
    # DD Mon YYYY (e.g., 15 Jan 2024)
    r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
    # Mon DD, YYYY (e.g., Jan 15, 2024)
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
]

# Price pattern: captures dollar amounts like $12.99, 12.99, $ 12.99
# Also handles OCR misreads where $ is read as S or 8
PRICE_PATTERN = r'[\$S]?\s*(\d+\.\d{2})'

# Total keywords
TOTAL_KEYWORDS = [
    'total', 'total amount', 'grand total', 'amount due',
    'balance due', 'total due', 'net total', 'sum',
    'amount', 'due', 'pay', 'charge'
]

# Subtotal keywords (to distinguish from total)
SUBTOTAL_KEYWORDS = [
    'subtotal', 'sub total', 'sub-total', 'items total', 'subtora'
]

# Tax keywords
TAX_KEYWORDS = ['tax', 'vat', 'gst', 'hst', 'sales tax']

# Keywords that indicate NON-item lines (to filter out)
NON_ITEM_KEYWORDS = [
    'total', 'subtotal', 'sub total', 'subtora', 'tax', 'vat', 'gst',
    'change', 'cash', 'credit', 'debit', 'visa', 'mastercard',
    'card', 'payment', 'paid', 'paytent', 'receipt', 'thank', 'welcome',
    'date', 'time', 'tel', 'phone', 'fax', 'address',
    'store', 'branch', 'invoice', 'order', 'transaction',
    'discount', 'savings', 'you saved', 'balance',
    'cashier', 'register', 'terminal', 'ref', 'auth',
    'item', 'price', 'qty', 'quantity', 'shopping'
]


def extract_store_name(lines: List[str], ocr_results: List[Dict] = None) -> Tuple[Optional[str], float]:
    """
    Extract store/vendor name from receipt text.
    
    Heuristic: The store name is typically in the first 1–3 lines
    of a receipt, often in larger text (higher position on receipt).
    
    Returns:
        Tuple of (store_name, confidence_score)
    """
    if not lines:
        return None, 0.0
    
    # Take the first few non-empty lines as candidates
    candidates = []
    for line in lines[:5]:
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 2:
            continue
        # Skip lines that are mostly numbers or look like dates/addresses   
        if re.match(r'^[\d\s\-/\.\$]+$', cleaned):
            continue
        # Skip lines with known non-store keywords
        lower = cleaned.lower()
        if any(kw in lower for kw in ['date', 'time', 'tel:', 'phone:', 'address:', 'receipt']):
            continue
        candidates.append(cleaned)
    
    if not candidates:
        return None, 0.0
    
    # First valid candidate is most likely the store name
    store_name = candidates[0]
    
    # Confidence based on position and content
    confidence = 0.85
    if len(store_name) < 3:
        confidence -= 0.2
    if any(char.isdigit() for char in store_name):
        confidence -= 0.15
    
    # Boost confidence if OCR confidence for this line is high
    if ocr_results:
        for r in ocr_results:
            if r["text"].strip() == store_name or store_name in r["text"]:
                ocr_conf = r.get("confidence", 0.5)
                confidence = confidence * 0.6 + ocr_conf * 0.4
                break
    
    return store_name, max(0.0, min(1.0, confidence))


def extract_date(text: str, ocr_results: List[Dict] = None) -> Tuple[Optional[str], float]:
    """
    Extract date from receipt text using regex patterns.
    
    Returns:
        Tuple of (date_string, confidence_score)
    """
    # First try clean date patterns
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            
            # Validate and normalize date
            confidence = 0.9
            
            # Check if it looks like a reasonable date
            try:
                # Try common formats
                for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y',
                           '%m/%d/%y', '%m-%d-%y', '%d-%m-%Y', '%d-%m-%y']:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed.year > 2000 and parsed.year < 2030:
                            confidence = 0.95
                        return date_str, confidence
                    except ValueError:
                        continue
            except Exception:
                pass
            
            return date_str, confidence
    
    # Try OCR-tolerant date patterns (handles partial reads like 03/1*1202)
    ocr_date_patterns = [
        r'(\d{1,2}[/\-]\d{1,2}[/\-\*]\d{2,4})',
        r'[Dd]at[eo]?\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
    ]
    for pattern in ocr_date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Clean up OCR artifacts
            date_str = date_str.replace('*', '/')
            return date_str, 0.7
    
    # Look for date keyword proximity
    lines = text.split('\n')
    for line in lines:
        lower = line.lower()
        if 'dat' in lower:
            # Try to extract a date-like pattern from the same line
            numbers = re.findall(r'\d+[/\-\.\*]\d+[/\-\.\*]\d+', line)
            if numbers:
                return numbers[0].replace('*', '/'), 0.65
    
    return None, 0.0


def extract_items(lines: List[str], ocr_results: List[Dict] = None) -> Tuple[List[Dict], float]:
    """
    Extract line items (product name + price) from receipt text.
    
    Heuristic: Item lines contain both text (item name) and a price
    (dollar amount). We filter out known non-item keywords.
    
    Returns:
        Tuple of (items_list, average_confidence)
    """
    items = []
    item_confidences = []
    
    for i, line in enumerate(lines):
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        
        lower = cleaned.lower()
        
        # Skip non-item lines
        if any(kw in lower for kw in NON_ITEM_KEYWORDS):
            continue
        
        # Look for a price anywhere in the line (handles OCR $ -> S/8)
        # First try standard price at end of line
        price_match = re.search(r'[\$S]?\s*(\d+\.\s*\d{2})\s*$', cleaned)
        if not price_match:
            # Try price anywhere in the line
            price_match = re.search(r'[\$S]\s*(\d+\.\s*\d{2})', cleaned)
        if not price_match:
            # Try bare price pattern
            price_match = re.search(r'(\d+\.\d{2})', cleaned)
        
        if not price_match:
            continue
        
        price_str = price_match.group(1).replace(' ', '')
        try:
            price = float(price_str)
        except ValueError:
            continue
        
        # Extract item name (everything before the price match)
        name_part = cleaned[:price_match.start()].strip()
        
        # Clean up item name - remove leading $ signs, quantity prefixes, etc.
        name_part = re.sub(r'^[\$S]\s*', '', name_part)
        name_part = re.sub(r'^\d+\s*[xX]?\s*', '', name_part)  # Remove quantity prefix
        name_part = re.sub(r'\s+', ' ', name_part).strip()
        
        # Remove trailing special characters
        name_part = name_part.rstrip('$S :.-')
        
        if not name_part or len(name_part) < 2:
            continue
        
        # Skip if price seems unreasonable
        if price <= 0 or price > 10000:
            continue
        
        # Calculate confidence for this item
        conf = 0.85
        if len(name_part) < 3:
            conf -= 0.15
        if price < 0.10:
            conf -= 0.1
        
        # Check OCR confidence for this line
        if ocr_results:
            for r in ocr_results:
                if r["text"].strip() in cleaned or cleaned in r["text"]:
                    ocr_conf = r.get("confidence", 0.5)
                    conf = conf * 0.5 + ocr_conf * 0.5
                    break
        
        items.append({
            "name": name_part,
            "price": f"{price:.2f}"
        })
        item_confidences.append(max(0.0, min(1.0, conf)))
    
    avg_conf = sum(item_confidences) / len(item_confidences) if item_confidences else 0.0
    return items, avg_conf


def extract_total(text: str, lines: List[str], 
                  ocr_results: List[Dict] = None) -> Tuple[Optional[str], float]:
    """
    Extract total amount from receipt text.
    
    Heuristic: Look for lines containing 'total' keyword followed by a price.
    Prefer 'total' over 'subtotal'. Use the last occurrence if multiple found.
    
    Returns:
        Tuple of (total_amount, confidence_score)
    """
    total_candidates = []
    
    for i, line in enumerate(lines):
        lower = line.lower().strip()
        
        # Skip subtotal lines
        if any(kw in lower for kw in SUBTOTAL_KEYWORDS):
            continue
        
        # Check for total keywords
        is_total_line = any(kw in lower for kw in TOTAL_KEYWORDS)
        if not is_total_line:
            continue
        
        # Extract price from this line
        price_match = re.search(PRICE_PATTERN, line)
        if price_match:
            amount = float(price_match.group(1))
            
            # Confidence based on keyword specificity
            conf = 0.8
            if 'total' in lower and 'sub' not in lower:
                conf = 0.92
            if 'grand total' in lower:
                conf = 0.97
            if 'amount due' in lower or 'balance due' in lower:
                conf = 0.95
            
            total_candidates.append({
                "amount": amount,
                "confidence": conf,
                "line_index": i
            })
    
    if not total_candidates:
        # Fallback: try to find the largest price in the last third of the receipt
        last_third = lines[len(lines) * 2 // 3:]
        prices = []
        for line in last_third:
            matches = re.findall(PRICE_PATTERN, line)
            for m in matches:
                prices.append(float(m))
        
        if prices:
            return f"{max(prices):.2f}", 0.5  # Low confidence fallback
        return None, 0.0
    
    # Prefer the last total-keyword match (most likely the grand total)
    best = total_candidates[-1]
    return f"{best['amount']:.2f}", best["confidence"]


def extract_all(ocr_output: dict) -> Dict:
    """
    Extract all structured fields from OCR output.
    
    Args:
        ocr_output: Full output from ocr.run_ocr_pipeline(), containing:
            - 'results': Word-level OCR results
            - 'lines': Merged line-level results
            - 'full_text': Concatenated text
    
    Returns:
        Dictionary with extracted fields and their confidence scores.
    """
    # Use merged lines for extraction (much better than individual words)
    merged_lines = ocr_output.get("lines", [])
    word_results = ocr_output.get("results", [])
    
    # Build text lines from merged line results
    lines = [r["text"] for r in merged_lines if r["text"]]
    full_text = "\n".join(lines)
    
    # Extract each field
    store_name, store_conf = extract_store_name(lines, merged_lines)
    date, date_conf = extract_date(full_text, merged_lines)
    items, items_conf = extract_items(lines, merged_lines)
    total, total_conf = extract_total(full_text, lines, merged_lines)
    
    return {
        "store_name": {
            "value": store_name,
            "confidence": round(store_conf, 4)
        },
        "date": {
            "value": date,
            "confidence": round(date_conf, 4)
        },
        "items": {
            "value": items,
            "confidence": round(items_conf, 4)
        },
        "total_amount": {
            "value": total,
            "confidence": round(total_conf, 4)
        }
    }

