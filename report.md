# AI OCR-Based Receipt Information Extraction System

## Technical Report

### 1. Approach

This project implements a modular, production-level pipeline for extracting structured data from receipt images. The system follows a 5-stage architecture:

**Stage 1 — Image Preprocessing:**
Raw receipt images are rarely clean. They suffer from noise (scanning artifacts), skew (hand-held photos), and poor contrast (faded thermal prints). Our preprocessing pipeline addresses each:
- **Median filtering** removes salt-and-pepper noise while preserving text edges — critical since text is our primary signal.
- **Hough Transform-based skew detection** finds dominant line angles in the image and rotates to correct alignment, improving OCR accuracy by 10–30% on skewed inputs.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** enhances local contrast without over-amplifying noise, making faded text readable.

**Stage 2 — OCR Text Extraction:**
We use **EasyOCR** as the primary OCR engine because it provides per-word confidence scores (0–1) out of the box, supports 80+ languages, and handles noisy images better than Tesseract for typical receipt formats. The OCR output includes bounding box coordinates for each detected text region, enabling spatial analysis.

**Stage 3 — Information Extraction:**
Structured fields are extracted using a combination of:
- **Regex patterns** for dates (handles 8+ formats), prices ($X.XX), and amounts.
- **Keyword heuristics** for totals ("TOTAL", "AMOUNT DUE"), excluding subtotals.
- **Positional logic** — store name is typically the first non-numeric text; items appear between the header and total section.

**Stage 4 — Confidence Scoring:**
A multi-level confidence system provides transparency:
- **OCR-level:** Direct from the EasyOCR engine (per-word).
- **Field-level:** Composite score blending OCR confidence (40%), regex validation success (30%), and keyword match strength (30%).
- **Reliability grading:** A–F scale with automatic flagging of fields below 0.7 threshold.

**Stage 5 — Financial Summary:**
Aggregates results across multiple receipts: total spend, transaction count, and per-store breakdown using pandas.

---

### 2. Tools & Technologies

| Tool | Purpose | Why Chosen |
|------|---------|------------|
| Python 3.8+ | Core language | Industry standard for ML/data |
| OpenCV | Image preprocessing | Most comprehensive CV library |
| EasyOCR | Text extraction | Built-in confidence scores, GPU support |
| NumPy | Array operations | Foundation for image processing |
| Pandas | Data aggregation | Financial summary generation |
| Pillow (PIL) | Receipt generation | Creating test images |
| Streamlit | Web UI | Rapid prototyping, built-in components |
| Plotly | Visualizations | Interactive charts with dark mode |

---

### 3. Challenges & Solutions

**Challenge 1: No Real Dataset Available**
*Solution:* Built a receipt generator (`generate_receipts.py`) that creates realistic receipt images with controllable noise, blur, skew, and contrast conditions. This also serves as a reproducible test suite.

**Challenge 2: Diverse Receipt Formats**
*Solution:* Used flexible regex patterns and keyword-based heuristics rather than rigid templates. The extractor handles multiple date formats, price formats, and total keyword variations.

**Challenge 3: Accurate Total Extraction**
*Solution:* Distinguished between "subtotal," "tax," and "total" using keyword exclusion lists. When multiple total candidates exist, the last matching line is preferred (most likely the grand total).

**Challenge 4: Confidence Scoring Without Ground Truth**
*Solution:* Implemented a composite scoring model that blends OCR engine confidence with validation signals (regex match quality, keyword presence). Fields below 0.7 are automatically flagged with recovery suggestions.

**Challenge 5: Edge Cases (Blurry/Partial Receipts)**
*Solution:* Graceful degradation — missing fields return `null` with zero confidence. The system never crashes on poor input; instead, it reports what it could and couldn't extract.

---

### 4. Possible Improvements

1. **Fine-tuned OCR Model:** Train a LayoutLMv3 or Donut model specifically on receipt datasets for 15–20% accuracy improvement.
2. **Named Entity Recognition:** Use spaCy or a transformer NER model to identify store names and item names more accurately.
3. **Template Matching:** For known store chains, use layout templates to guide extraction.
4. **Multi-language Support:** EasyOCR supports 80+ languages — extend regex patterns for non-English receipts.
5. **Cloud Deployment:** Deploy the Streamlit app on Streamlit Cloud or AWS with GPU-backed EasyOCR for faster processing.
6. **Database Integration:** Store extracted data in PostgreSQL for longitudinal spending analysis.

---

### 5. Architecture Diagram

```
Receipt Image → Preprocessing → OCR Engine → Text Extraction → Confidence Scoring → JSON Output
     │              │                │              │                  │                   │
     │         Denoise/Deskew    EasyOCR        Regex/NLP         Composite Score      Structured
     │         CLAHE Enhance     Bounding Boxes  Heuristics       Reliability Flags    Data + Report
     │                                                                                     │
     └─────────────────────────── Streamlit UI ←───────────────── Financial Summary ←──────┘
```
