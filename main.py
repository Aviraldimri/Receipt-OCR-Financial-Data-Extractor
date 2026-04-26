"""
main.py — CLI Entry Point
===========================
Orchestrates the full OCR receipt extraction pipeline:
1. Generate sample receipts (if needed)
2. Preprocess images
3. Run OCR
4. Extract structured data
5. Score confidence
6. Generate financial summary
7. Save JSON outputs
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Local modules
from preprocessing import preprocess_receipt
from ocr import run_ocr_pipeline, draw_bounding_boxes
from extractor import extract_all
from confidence import build_confidence_report
from summary import generate_financial_summary
import cv2


def process_single_receipt(image_path: str, output_dir: str = "output",
                           save_annotated: bool = True,
                           verbose: bool = True) -> dict:
    """
    Process a single receipt image through the full pipeline.
    
    Args:
        image_path: Path to the receipt image.
        output_dir: Directory to save outputs.
        save_annotated: Whether to save annotated image with bounding boxes.
        verbose: Whether to print progress.
    
    Returns:
        Complete extraction result dictionary.
    """
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
    
    # Step 1: Preprocessing
    if verbose:
        print("\nStep 1: Image Preprocessing...")
    
    start_time = time.time()
    preprocess_result = preprocess_receipt(
        image_path,
        noise_method="median",
        enhance_method="clahe",
        do_skew_correction=True,
        do_binarize=False
    )
    preprocess_time = time.time() - start_time
    
    if verbose:
        for step in preprocess_result["steps"]:
            print(f"   - {step}")
        print(f"   Preprocessing time: {preprocess_time:.2f}s")
    
    # Step 2: OCR
    if verbose:
        print("\nStep 2: Running OCR...")
    
    start_time = time.time()
    # Use the color-processed image (denoised + deskewed) for OCR
    # EasyOCR performs better on color images than CLAHE grayscale
    ocr_output = run_ocr_pipeline(preprocess_result["color_processed"])
    ocr_time = time.time() - start_time
    
    if verbose:
        print(f"   - Detected {ocr_output['num_detections']} text regions")
        print(f"   - Average OCR confidence: {ocr_output['average_confidence']:.2f}")
        print(f"   OCR time: {ocr_time:.2f}s")
    
    # Step 3: Information Extraction
    if verbose:
        print("\nStep 3: Extracting Information...")
    
    extracted = extract_all(ocr_output)
    
    if verbose:
        store = extracted["store_name"]["value"] or "N/A"
        date = extracted["date"]["value"] or "N/A"
        items = extracted["items"]["value"] or []
        total = extracted["total_amount"]["value"] or "N/A"
        print(f"   Store: {store}")
        print(f"   Date:  {date}")
        print(f"   Items: {len(items)}")
        print(f"   Total: ${total}")
    
    # Step 4: Confidence Scoring
    if verbose:
        print("\nStep 4: Computing Confidence Scores...")
    
    confidence_report = build_confidence_report(extracted, ocr_output["results"])
    
    if verbose:
        overall = confidence_report["overall_reliability"]
        print(f"   Grade: {overall['grade']}")
        print(f"   Score: {overall['score']:.2f}")
        print(f"   {overall['assessment']}")
        
        if confidence_report["warnings"]:
            print(f"\n   Warnings:")
            for w in confidence_report["warnings"]:
                print(f"      - {w['message']}")
    
    # Step 5: Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output JSON
    output = {
        "metadata": {
            "filename": filename,
            "processed_at": datetime.now().isoformat(),
            "preprocessing_time_s": round(preprocess_time, 3),
            "ocr_time_s": round(ocr_time, 3),
            "skew_angle": preprocess_result["skew_angle"]
        },
        "extracted_data": {
            "store_name": extracted["store_name"]["value"],
            "date": extracted["date"]["value"],
            "items": extracted["items"]["value"],
            "total_amount": extracted["total_amount"]["value"]
        },
        "confidence": {
            "store_name": {
                "value": extracted["store_name"]["value"],
                "confidence": confidence_report["field_confidence"]["store_name"]["confidence"]
            },
            "date": {
                "value": extracted["date"]["value"],
                "confidence": confidence_report["field_confidence"]["date"]["confidence"]
            },
            "items": {
                "value": extracted["items"]["value"],
                "confidence": confidence_report["field_confidence"]["items"]["confidence"]
            },
            "total_amount": {
                "value": extracted["total_amount"]["value"],
                "confidence": confidence_report["field_confidence"]["total_amount"]["confidence"]
            }
        },
        "reliability": {
            "overall_grade": confidence_report["overall_reliability"]["grade"],
            "overall_score": confidence_report["overall_reliability"]["score"],
            "assessment": confidence_report["overall_reliability"]["assessment"],
            "warnings": [w["message"] for w in confidence_report["warnings"]]
        },
        "ocr_stats": {
            "num_detections": ocr_output["num_detections"],
            "average_confidence": ocr_output["average_confidence"]
        },
        "raw_text": ocr_output["full_text"]
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{name_no_ext}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n   Saved JSON: {json_path}")
    
    # Save annotated image
    if save_annotated:
        annotated = draw_bounding_boxes(
            preprocess_result["color_processed"],
            ocr_output["results"]
        )
        annotated_path = os.path.join(output_dir, f"{name_no_ext}_annotated.png")
        cv2.imwrite(annotated_path, annotated)
        if verbose:
            print(f"   Saved annotated image: {annotated_path}")
    
    return {
        "filename": filename,
        "extracted": extracted,
        "confidence": confidence_report,
        "output": output
    }


def process_receipt_folder(input_dir: str, output_dir: str = "output",
                           verbose: bool = True) -> dict:
    """
    Process all receipt images in a folder.
    
    Args:
        input_dir: Directory containing receipt images.
        output_dir: Directory to save outputs.
        verbose: Whether to print progress.
    
    Returns:
        Full results including financial summary.
    """
    # Find image files
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported_extensions:
            image_files.append(os.path.join(input_dir, fname))
    
    if not image_files:
        print(f"ERROR: No image files found in {input_dir}")
        return {}
    
    print(f"\nFound {len(image_files)} receipt image(s) in '{input_dir}'")
    print(f"Output will be saved to '{output_dir}'")
    
    # Process each receipt
    all_results = []
    for image_path in image_files:
        try:
            result = process_single_receipt(image_path, output_dir, verbose=verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate financial summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"Generating Financial Summary...")
        print(f"{'='*60}")
        
        summary = generate_financial_summary(all_results)
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary_report.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        fs = summary["financial_summary"]
        print(f"\n   Total Spend:         ${fs['total_spend']:.2f}")
        print(f"   Transactions:        {fs['num_transactions']}")
        print(f"   Avg per Transaction: ${fs['average_per_transaction']:.2f}")
        
        if summary["spend_per_store"]:
            print(f"\n   Spend per Store:")
            for store in summary["spend_per_store"]:
                print(f"      - {store['store_name']}: ${store['total_spent']:.2f} ({store['transactions']} txn)")
        
        print(f"\n   Saved summary: {summary_path}")
        
        return {
            "results": all_results,
            "summary": summary
        }
    
    return {"results": all_results, "summary": {}}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI OCR Receipt Information Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Process sample_receipts/
  python main.py --input my_receipts/               # Process custom folder
  python main.py --input receipt.png --output results/  # Single image
  python main.py --generate                         # Generate sample receipts first
        """
    )
    
    parser.add_argument("--input", "-i", default="sample_receipts",
                       help="Input directory or single image file (default: sample_receipts/)")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory for JSON results (default: output/)")
    parser.add_argument("--generate", "-g", action="store_true",
                       help="Generate sample receipt images before processing")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--no-annotate", action="store_true",
                       help="Skip saving annotated images")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  AI OCR Receipt Information Extraction System")
    print("="*60)
    
    # Generate samples if requested or if input dir doesn't exist
    if args.generate or (args.input == "sample_receipts" and not os.path.exists("sample_receipts")):
        print("\nGenerating sample receipt images...")
        from generate_receipts import generate_sample_receipts
        generate_sample_receipts()
    
    # Process
    if os.path.isfile(args.input):
        # Single file
        process_single_receipt(
            args.input, args.output,
            save_annotated=not args.no_annotate,
            verbose=not args.quiet
        )
    elif os.path.isdir(args.input):
        # Directory
        process_receipt_folder(
            args.input, args.output,
            verbose=not args.quiet
        )
    else:
        print(f"ERROR: Input not found: {args.input}")
        print("   Use --generate to create sample receipts first.")
        sys.exit(1)
    
    print(f"\nDone. Check the '{args.output}/' directory for results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
