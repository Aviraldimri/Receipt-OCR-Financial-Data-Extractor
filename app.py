"""
app.py — Streamlit UI for Receipt OCR Extraction
==================================================
Interactive web interface for uploading receipt images,
running the OCR pipeline, and visualizing results.

Run with:  streamlit run app.py
"""

import streamlit as st
import os
import json
import tempfile
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Local modules
from preprocessing import preprocess_receipt, load_image
from ocr import run_ocr_pipeline, draw_bounding_boxes
from extractor import extract_all
from confidence import build_confidence_report
from summary import generate_financial_summary


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🧾 AI Receipt OCR Extractor",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 8px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Grade badge */
    .grade-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 18px;
        color: white;
    }
    .grade-A { background: linear-gradient(135deg, #00b894, #00cec9); }
    .grade-B { background: linear-gradient(135deg, #0984e3, #74b9ff); }
    .grade-C { background: linear-gradient(135deg, #fdcb6e, #e17055); }
    .grade-D { background: linear-gradient(135deg, #e17055, #d63031); }
    .grade-F { background: linear-gradient(135deg, #d63031, #636e72); }
    
    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 4px 0;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Warning badge */
    .warning-badge {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #e74c3c;
        font-size: 14px;
    }
    
    /* Items table */
    .items-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 8px;
    }
    .items-table th {
        background: rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.8);
        padding: 10px 16px;
        text-align: left;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .items-table td {
        padding: 10px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #ffffff;
        font-size: 15px;
    }
    .items-table tr:hover td {
        background: rgba(255,255,255,0.04);
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 22px;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 20px;
    }
    
    /* JSON display */
    .json-display {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        font-family: 'Consolas', monospace;
        font-size: 13px;
        color: #a8e6cf;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, icon: str = ""):
    """Render a glassmorphism metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(label: str, confidence: float):
    """Render a confidence bar with color coding."""
    if confidence >= 0.85:
        color = "linear-gradient(90deg, #00b894, #00cec9)"
    elif confidence >= 0.7:
        color = "linear-gradient(90deg, #0984e3, #74b9ff)"
    elif confidence >= 0.5:
        color = "linear-gradient(90deg, #fdcb6e, #e17055)"
    else:
        color = "linear-gradient(90deg, #e17055, #d63031)"
    
    width = max(5, confidence * 100)
    
    st.markdown(f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: rgba(255,255,255,0.8); font-size: 14px;">{label}</span>
            <span style="color: rgba(255,255,255,0.6); font-size: 13px;">{confidence:.1%}</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width: {width}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_grade_badge(grade: str):
    """Render a colored grade badge."""
    st.markdown(f"""
    <span class="grade-badge grade-{grade}">{grade}</span>
    """, unsafe_allow_html=True)


def process_uploaded_image(uploaded_file) -> dict:
    """Process an uploaded image through the pipeline."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    
    try:
        # Preprocessing
        preprocess_result = preprocess_receipt(
            tmp_path,
            noise_method=st.session_state.get("noise_method", "median"),
            enhance_method=st.session_state.get("enhance_method", "clahe"),
            do_skew_correction=st.session_state.get("skew_correction", True),
            do_binarize=st.session_state.get("binarize", False)
        )
        
        # OCR
        ocr_output = run_ocr_pipeline(preprocess_result["processed"])
        
        # Extraction
        extracted = extract_all(ocr_output)
        
        # Confidence
        confidence_report = build_confidence_report(extracted, ocr_output["results"])
        
        # Annotated image
        annotated = draw_bounding_boxes(
            preprocess_result["color_processed"],
            ocr_output["results"]
        )
        
        return {
            "preprocess": preprocess_result,
            "ocr": ocr_output,
            "extracted": extracted,
            "confidence": confidence_report,
            "annotated": annotated
        }
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧾 Receipt OCR")
    st.markdown("### AI-Powered Extraction")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Preprocessing Settings")
    
    noise_method = st.selectbox(
        "Noise Removal Method",
        ["median", "gaussian", "bilateral"],
        index=0,
        key="noise_method"
    )
    
    enhance_method = st.selectbox(
        "Contrast Enhancement",
        ["clahe", "histogram"],
        index=0,
        key="enhance_method"
    )
    
    skew_correction = st.checkbox("Skew Correction", value=True, key="skew_correction")
    binarize = st.checkbox("Binarization", value=False, key="binarize")
    
    st.markdown("---")
    st.markdown("#### 📤 Upload Receipts")
    
    uploaded_files = st.file_uploader(
        "Drop receipt images here",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: rgba(255,255,255,0.4); font-size: 12px;'>"
        "Built with EasyOCR + OpenCV<br>© 2025 AI Receipt OCR</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

if not uploaded_files:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="font-size: 48px; color: #ffffff; margin-bottom: 12px;">
            🧾 AI Receipt OCR Extractor
        </h1>
        <p style="font-size: 20px; color: rgba(255,255,255,0.6); max-width: 600px; margin: 0 auto 40px auto;">
            Upload receipt images to automatically extract store names, dates, 
            items, prices, and totals with confidence scoring.
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div class="metric-card" style="min-width: 180px;">
                <div class="metric-label">🔍 OCR Engine</div>
                <div class="metric-value" style="font-size: 18px;">EasyOCR</div>
            </div>
            <div class="metric-card" style="min-width: 180px;">
                <div class="metric-label">🖼️ Preprocessing</div>
                <div class="metric-value" style="font-size: 18px;">OpenCV</div>
            </div>
            <div class="metric-card" style="min-width: 180px;">
                <div class="metric-label">📊 Confidence</div>
                <div class="metric-value" style="font-size: 18px;">Multi-Level</div>
            </div>
        </div>
        <p style="color: rgba(255,255,255,0.4); margin-top: 50px; font-size: 14px;">
            ← Upload receipt images from the sidebar to get started
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Process receipts
    st.markdown('<h1 style="color: #ffffff; margin-bottom: 0;">🧾 Extraction Results</h1>',
               unsafe_allow_html=True)
    
    # Initialize session state for results
    if "all_results" not in st.session_state:
        st.session_state.all_results = []
    
    # Process button
    if st.button("🚀 Process Receipts", type="primary", use_container_width=True):
        st.session_state.all_results = []
        
        progress = st.progress(0, text="Processing receipts...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Processing {uploaded_file.name}..."
            )
            
            try:
                result = process_uploaded_image(uploaded_file)
                result["filename"] = uploaded_file.name
                result["uploaded_file"] = uploaded_file
                st.session_state.all_results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        progress.progress(1.0, text="✅ All receipts processed!")
    
    # Display results
    if st.session_state.all_results:
        results = st.session_state.all_results
        
        # ─── SUMMARY METRICS ───
        st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_spend = 0
        for r in results:
            total_val = r["extracted"]["total_amount"]["value"]
            if total_val:
                try:
                    total_spend += float(total_val)
                except:
                    pass
        
        avg_conf = np.mean([
            r["confidence"]["overall_reliability"]["score"]
            for r in results if r["confidence"]["overall_reliability"]["score"] > 0
        ]) if results else 0
        
        with col1:
            render_metric_card("Receipts Processed", str(len(results)), "📄")
        with col2:
            render_metric_card("Total Spend", f"${total_spend:.2f}", "💰")
        with col3:
            render_metric_card("Avg Confidence", f"{avg_conf:.1%}", "📊")
        with col4:
            avg_grade = results[0]["confidence"]["overall_reliability"]["grade"] if results else "N/A"
            render_metric_card("Quality Grade", avg_grade, "🏆")
        
        st.markdown("---")
        
        # ─── PER-RECEIPT TABS ───
        tab_names = [f"📄 {r['filename']}" for r in results]
        if len(results) > 1:
            tab_names.append("📊 Summary")
        
        tabs = st.tabs(tab_names)
        
        for idx, (tab, result) in enumerate(zip(tabs[:len(results)], results)):
            with tab:
                extracted = result["extracted"]
                confidence = result["confidence"]
                
                col_img, col_data = st.columns([1, 1])
                
                with col_img:
                    st.markdown('<div class="section-header">🖼️ Receipt Image</div>',
                               unsafe_allow_html=True)
                    
                    view_mode = st.radio(
                        "View",
                        ["Original", "Preprocessed", "Annotated (Bounding Boxes)"],
                        horizontal=True,
                        key=f"view_{idx}"
                    )
                    
                    if view_mode == "Original":
                        result["uploaded_file"].seek(0)
                        st.image(result["uploaded_file"], use_container_width=True)
                    elif view_mode == "Preprocessed":
                        st.image(result["preprocess"]["processed"],
                                use_container_width=True, clamp=True)
                    else:
                        annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)
                
                with col_data:
                    st.markdown('<div class="section-header">📋 Extracted Data</div>',
                               unsafe_allow_html=True)
                    
                    # Store name
                    store = extracted["store_name"]["value"] or "N/A"
                    date = extracted["date"]["value"] or "N/A"
                    total = extracted["total_amount"]["value"] or "N/A"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🏪 Store Name</div>
                        <div class="metric-value" style="font-size: 22px;">{store}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        render_metric_card("Date", date, "📅")
                    with sub_col2:
                        render_metric_card("Total", f"${total}" if total != "N/A" else "N/A", "💵")
                    
                    # Items
                    items = extracted["items"]["value"] or []
                    if items:
                        items_html = '<table class="items-table"><thead><tr><th>Item</th><th>Price</th></tr></thead><tbody>'
                        for item in items:
                            items_html += f'<tr><td>{item["name"]}</td><td>${item["price"]}</td></tr>'
                        items_html += '</tbody></table>'
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🛒 Items ({len(items)})</div>
                            {items_html}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No individual items detected")
                
                # ─── CONFIDENCE SECTION ───
                st.markdown('<div class="section-header">🎯 Confidence Scores</div>',
                           unsafe_allow_html=True)
                
                conf_col1, conf_col2 = st.columns([1, 1])
                
                with conf_col1:
                    overall = confidence["overall_reliability"]
                    st.markdown("**Overall Reliability**")
                    render_grade_badge(overall["grade"])
                    st.markdown(f"<p style='color: rgba(255,255,255,0.6); margin-top: 8px;'>"
                              f"{overall['assessment']}</p>", unsafe_allow_html=True)
                    
                    # Field confidence bars
                    st.markdown("**Field Confidence**")
                    for field_name in ["store_name", "date", "items", "total_amount"]:
                        field = confidence["field_confidence"].get(field_name, {})
                        conf = field.get("confidence", 0)
                        label = field_name.replace("_", " ").title()
                        render_confidence_bar(label, conf)
                
                with conf_col2:
                    # Confidence radar chart
                    field_names = ["Store Name", "Date", "Items", "Total Amount"]
                    field_confs = [
                        confidence["field_confidence"].get(f, {}).get("confidence", 0)
                        for f in ["store_name", "date", "items", "total_amount"]
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=field_confs + [field_confs[0]],
                        theta=field_names + [field_names[0]],
                        fill='toself',
                        fillcolor='rgba(99, 110, 250, 0.15)',
                        line=dict(color='#636EFA', width=2),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1],
                                          gridcolor='rgba(255,255,255,0.1)'),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='rgba(255,255,255,0.7)'),
                        height=300,
                        margin=dict(t=30, b=30, l=60, r=60)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Warnings
                if confidence["warnings"]:
                    st.markdown("**⚠️ Warnings**")
                    for w in confidence["warnings"]:
                        st.markdown(f'<div class="warning-badge">{w["message"]}</div>',
                                   unsafe_allow_html=True)
                
                # ─── JSON OUTPUT ───
                with st.expander("📄 View Full JSON Output"):
                    output_json = {
                        "extracted_data": {
                            "store_name": extracted["store_name"]["value"],
                            "date": extracted["date"]["value"],
                            "items": extracted["items"]["value"],
                            "total_amount": extracted["total_amount"]["value"]
                        },
                        "confidence": {
                            field: {
                                "value": confidence["field_confidence"][field]["value"],
                                "confidence": confidence["field_confidence"][field]["confidence"]
                            }
                            for field in ["store_name", "date", "items", "total_amount"]
                        },
                        "reliability": {
                            "grade": confidence["overall_reliability"]["grade"],
                            "score": confidence["overall_reliability"]["score"]
                        }
                    }
                    st.json(output_json)
                    
                    st.download_button(
                        "⬇️ Download JSON",
                        json.dumps(output_json, indent=2),
                        f"{os.path.splitext(result['filename'])[0]}.json",
                        "application/json"
                    )
                
                # ─── RAW TEXT ───
                with st.expander("📝 View Raw OCR Text"):
                    st.code(result["ocr"]["full_text"], language=None)
        
        # ─── SUMMARY TAB ───
        if len(results) > 1:
            with tabs[-1]:
                st.markdown('<div class="section-header">📊 Financial Summary</div>',
                           unsafe_allow_html=True)
                
                # Build summary
                summary_results = []
                for r in results:
                    summary_results.append({
                        "filename": r["filename"],
                        "extracted": r["extracted"],
                        "confidence": r["confidence"]
                    })
                
                summary = generate_financial_summary(summary_results)
                fs = summary["financial_summary"]
                
                s_col1, s_col2, s_col3 = st.columns(3)
                with s_col1:
                    render_metric_card("Total Spend", f"${fs['total_spend']:.2f}", "💰")
                with s_col2:
                    render_metric_card("Transactions", str(fs["num_transactions"]), "📝")
                with s_col3:
                    render_metric_card("Avg per Txn", f"${fs['average_per_transaction']:.2f}", "📊")
                
                # Spend per store chart
                if summary["spend_per_store"]:
                    st.markdown("**💸 Spend per Store**")
                    store_df = pd.DataFrame(summary["spend_per_store"])
                    
                    fig = px.bar(
                        store_df, x="store_name", y="total_spent",
                        color="total_spent",
                        color_continuous_scale="Viridis",
                        labels={"store_name": "Store", "total_spent": "Total Spent ($)"}
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='rgba(255,255,255,0.7)'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download summary
                st.download_button(
                    "⬇️ Download Summary JSON",
                    json.dumps(summary, indent=2),
                    "summary_report.json",
                    "application/json"
                )
