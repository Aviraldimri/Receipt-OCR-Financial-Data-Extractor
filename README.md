# 🧾 AI OCR-Based Receipt Information Extraction System

An AI-powered system that extracts structured data from receipt images using OCR, with confidence scoring and a Streamlit web interface.

---

## 📂 Project Structure

```
carbon crunch/
├── main.py                 # CLI entry point — run the full pipeline
├── app.py                  # Streamlit web UI (bonus)
├── preprocessing.py        # Image preprocessing (denoise, deskew, CLAHE)
├── ocr.py                  # EasyOCR pipeline with bounding boxes
├── extractor.py            # Regex + heuristic field extraction
├── confidence.py           # Multi-level confidence scoring
├── summary.py              # Financial summary across receipts
├── generate_receipts.py    # Sample receipt image generator
├── requirements.txt        # Python dependencies
├── report.md               # Technical report (1-2 pages)
├── README.md               # This file
├── sample_receipts/        # Generated sample receipt images
│   ├── receipt_1.png
│   ├── receipt_2.png
│   └── ...
└── output/                 # Extraction results
    ├── receipt_1.json
    ├── receipt_2.json
    ├── receipt_1_annotated.png
    └── summary_report.json
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd "carbon crunch"
pip install -r requirements.txt
```

### Step 2: Generate Sample Receipts

```bash
python generate_receipts.py
```

This creates 5 receipt images in `sample_receipts/` with varying conditions (clean, noisy, blurry, skewed).

### Step 3: Run the Pipeline (CLI)

```bash
# Process all receipts in sample_receipts/ folder
python main.py

# Process a specific folder
python main.py --input my_receipts/ --output my_results/

# Process a single image
python main.py --input receipt.png

# Generate samples and process in one command
python main.py --generate
```

### Step 4: Run the Streamlit UI (Bonus)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload receipt images and see extraction results with interactive visualizations.

---

## 📊 Sample Output JSON

```json
{
  "metadata": {
    "filename": "receipt_1.png",
    "processed_at": "2025-03-15T14:32:05",
    "preprocessing_time_s": 0.15,
    "ocr_time_s": 2.34,
    "skew_angle": 0.0
  },
  "extracted_data": {
    "store_name": "FRESH MART GROCERY",
    "date": "03/15/2025",
    "items": [
      {"name": "Organic Bananas", "price": "2.49"},
      {"name": "Whole Milk 1 Gal", "price": "4.99"},
      {"name": "Sourdough Bread", "price": "5.49"},
      {"name": "Cheddar Cheese", "price": "6.99"},
      {"name": "Chicken Breast", "price": "8.99"}
    ],
    "total_amount": "45.35"
  },
  "confidence": {
    "store_name": {"value": "FRESH MART GROCERY", "confidence": 0.91},
    "date": {"value": "03/15/2025", "confidence": 0.95},
    "items": {"value": [...], "confidence": 0.87},
    "total_amount": {"value": "45.35", "confidence": 0.93}
  },
  "reliability": {
    "overall_grade": "A",
    "overall_score": 0.915,
    "assessment": "High confidence — extraction is reliable",
    "warnings": []
  }
}
```

---

## 🧠 How It Works

### 1. Image Preprocessing
- **Noise Removal:** Median/Gaussian/Bilateral filtering removes scanning artifacts
- **Skew Correction:** Hough transform detects text line angles and auto-rotates
- **Contrast Enhancement:** CLAHE makes faded text readable without amplifying noise

### 2. OCR Pipeline
- **EasyOCR** extracts text with per-word confidence scores (0–1)
- Bounding boxes are drawn for visual verification

### 3. Key Information Extraction
- **Store Name:** First meaningful text line (positional heuristic)
- **Date:** Regex matching 8+ date formats
- **Items:** Lines with text + price pattern, excluding non-item keywords
- **Total:** Keyword matching ("TOTAL", "AMOUNT DUE") with subtotal exclusion

### 4. Confidence Scoring
- **OCR-level:** Per-word confidence from EasyOCR
- **Field-level:** Composite score = 0.4 × OCR + 0.3 × regex + 0.3 × keyword
- **Reliability grading:** A–F scale; fields below 0.7 are flagged

### 5. Edge Case Handling
- Missing fields → `null` with zero confidence
- Blurry images → preprocessing + graceful degradation
- Unusual formats → flexible regex + fallback logic
- Partial receipts → extract what's available, flag the rest

---

## ⚙️ CLI Options

```
python main.py [OPTIONS]

Options:
  --input,  -i    Input directory or image file (default: sample_receipts/)
  --output, -o    Output directory (default: output/)
  --generate, -g  Generate sample receipts before processing
  --quiet, -q     Suppress verbose output
  --no-annotate   Skip saving annotated images
```

---

## 🛠️ Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core language |
| OpenCV | 4.8+ | Image preprocessing |
| EasyOCR | 1.7+ | Text extraction |
| NumPy | 1.24+ | Array operations |
| Pandas | 2.0+ | Data aggregation |
| Streamlit | 1.28+ | Web interface |
| Plotly | 5.18+ | Interactive charts |
| Pillow | 10.0+ | Image generation |

---

## 📄 License

This project is for educational and demonstration purposes.
