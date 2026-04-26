"""
summary.py — Financial Summary Module
=======================================
Aggregates extracted data across multiple receipts to produce
financial summaries: total spend, transaction count, per-store breakdown.
"""

import pandas as pd
from typing import List, Dict
from collections import defaultdict


def generate_financial_summary(receipt_results: List[Dict]) -> Dict:
    """
    Generate a financial summary from multiple receipt extraction results.
    
    Args:
        receipt_results: List of dictionaries, each containing:
            - 'filename': Receipt image filename
            - 'extracted': Output from extractor.extract_all()
            - 'confidence': Confidence report from confidence.py
    
    Returns:
        Financial summary dictionary.
    """
    total_spend = 0.0
    num_transactions = len(receipt_results)
    store_spending = defaultdict(float)
    store_transactions = defaultdict(int)
    all_items = []
    receipts_summary = []
    
    for receipt in receipt_results:
        extracted = receipt.get("extracted", {})
        filename = receipt.get("filename", "unknown")
        
        # Get store name
        store_data = extracted.get("store_name", {})
        store_name = store_data.get("value", "Unknown Store") or "Unknown Store"
        
        # Get total amount
        total_data = extracted.get("total_amount", {})
        total_str = total_data.get("value", "0.00") or "0.00"
        try:
            total_amount = float(total_str)
        except (ValueError, TypeError):
            total_amount = 0.0
        
        # Get date
        date_data = extracted.get("date", {})
        date = date_data.get("value", "N/A") or "N/A"
        
        # Get items
        items_data = extracted.get("items", {})
        items = items_data.get("value", []) or []
        
        # Accumulate
        total_spend += total_amount
        store_spending[store_name] += total_amount
        store_transactions[store_name] += 1
        
        for item in items:
            all_items.append({
                "store": store_name,
                "name": item.get("name", "Unknown"),
                "price": float(item.get("price", 0)),
                "receipt_file": filename
            })
        
        receipts_summary.append({
            "filename": filename,
            "store": store_name,
            "date": date,
            "total": total_amount,
            "items_count": len(items),
            "confidence_grade": receipt.get("confidence", {})
                .get("overall_reliability", {}).get("grade", "N/A")
        })
    
    # Per-store breakdown
    per_store = []
    for store, amount in sorted(store_spending.items(), key=lambda x: x[1], reverse=True):
        per_store.append({
            "store_name": store,
            "total_spent": round(amount, 2),
            "transactions": store_transactions[store],
            "average_per_transaction": round(amount / store_transactions[store], 2)
        })
    
    # Top items by price
    sorted_items = sorted(all_items, key=lambda x: x["price"], reverse=True)
    top_items = sorted_items[:10]
    
    return {
        "financial_summary": {
            "total_spend": round(total_spend, 2),
            "num_transactions": num_transactions,
            "average_per_transaction": round(total_spend / max(num_transactions, 1), 2),
            "currency": "USD"
        },
        "spend_per_store": per_store,
        "top_items": top_items,
        "receipts": receipts_summary
    }


def summary_to_dataframe(summary: Dict) -> Dict[str, pd.DataFrame]:
    """
    Convert the financial summary to pandas DataFrames for analysis.
    
    Returns:
        Dictionary of DataFrames:
        - 'overview': Overall financial summary
        - 'per_store': Per-store breakdown
        - 'receipts': Individual receipt details
        - 'top_items': Top items by price
    """
    # Overview
    fs = summary["financial_summary"]
    overview_df = pd.DataFrame([{
        "Metric": k.replace("_", " ").title(),
        "Value": v
    } for k, v in fs.items()])
    
    # Per store
    store_df = pd.DataFrame(summary["spend_per_store"])
    if not store_df.empty:
        store_df.columns = [c.replace("_", " ").title() for c in store_df.columns]
    
    # Receipts
    receipts_df = pd.DataFrame(summary["receipts"])
    if not receipts_df.empty:
        receipts_df.columns = [c.replace("_", " ").title() for c in receipts_df.columns]
    
    # Top items
    items_df = pd.DataFrame(summary["top_items"])
    if not items_df.empty:
        items_df.columns = [c.replace("_", " ").title() for c in items_df.columns]
    
    return {
        "overview": overview_df,
        "per_store": store_df,
        "receipts": receipts_df,
        "top_items": items_df
    }
