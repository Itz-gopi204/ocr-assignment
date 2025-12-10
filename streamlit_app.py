"""
Streamlit Web Interface for OCR PII Extraction Pipeline
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import io
import time
import os
from datetime import datetime

# Import enhanced pipeline
from enhanced_pipeline import (
    EnhancedOCRPipeline,
    PipelineConfig,
    save_enhanced_results
)

# Import Azure Document Intelligence
try:
    from azure_ocr import process_with_azure, AzureConfig, AZURE_AVAILABLE
    AZURE_READY = AZURE_AVAILABLE and AzureConfig().is_configured()
except ImportError:
    AZURE_READY = False

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="OCR PII Extraction Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .pii-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
# ============================================

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/privacy.png", width=80)
    st.title("Configuration")

    st.markdown("---")

    # OCR Engine Selection
    st.subheader("OCR Engine")

    # Azure option (recommended)
    if AZURE_READY:
        ocr_options = ["Azure Document Intelligence (Recommended)", "EasyOCR", "Both (EasyOCR + Tesseract)"]
        default_idx = 0
    else:
        ocr_options = ["EasyOCR", "Both (EasyOCR + Tesseract)"]
        default_idx = 0
        st.warning("Azure not configured. Using EasyOCR.")

    ocr_engine_choice = st.selectbox(
        "Select OCR Engine",
        ocr_options,
        index=default_idx,
        help="Azure Document Intelligence provides best accuracy for handwritten documents"
    )

    # Map selection to engine
    use_azure = "Azure" in ocr_engine_choice
    ocr_engine = "both" if "Both" in ocr_engine_choice else "easyocr"

    if use_azure:
        st.success("Using Azure Document Intelligence")
    else:
        # Pre-processing options (only for non-Azure)
        st.subheader("Pre-processing")
        apply_morphology = st.checkbox("Apply Morphological Ops", value=False,
                                       help="Helps connect broken characters")
        denoise_strength = st.slider("Denoise Strength", 5, 20, 8)

    # PII Detection options
    st.subheader("PII Detection")
    detect_aadhaar = st.checkbox("Detect Aadhaar", value=True)
    detect_pan = st.checkbox("Detect PAN", value=True)
    detect_phone = st.checkbox("Detect Phone", value=True)
    detect_medical = st.checkbox("Detect Medical IDs", value=True)

    st.markdown("---")

    # Initialize Pipeline Button (only needed for non-Azure)
    if not use_azure:
        if st.button("Initialize EasyOCR Pipeline", type="primary"):
            with st.spinner("Initializing pipeline... (this may take a moment)"):
                try:
                    config = PipelineConfig(
                        ocr_engine=ocr_engine,
                        apply_morphology=apply_morphology,
                        denoise_strength=denoise_strength,
                        detect_aadhaar=detect_aadhaar,
                        detect_pan=detect_pan,
                        detect_phone=detect_phone,
                        detect_medical_ids=detect_medical,
                        color_coded_redaction=True
                    )
                    st.session_state.pipeline = EnhancedOCRPipeline(config)
                    st.success("Pipeline initialized!")
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")

# ============================================
# MAIN CONTENT
# ============================================

# Header
st.markdown('<p class="main-header">🔍 OCR PII Extraction Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Extract and redact Personal Identifiable Information from handwritten medical documents</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📋 Extracted Data", "📈 Metrics"])

# ============================================
# TAB 1: UPLOAD & PROCESS
# ============================================

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a handwritten document image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPEG or PNG image of a handwritten document"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_container_width=True)

            # Process button
            if st.button("Process Document", type="primary"):
                # Read image bytes
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                if use_azure and AZURE_READY:
                    # Use Azure Document Intelligence
                    with st.spinner("Processing with Azure Document Intelligence..."):
                        try:
                            start_time = time.time()
                            azure_result = process_with_azure(image_bytes=image_bytes)
                            processing_time = (time.time() - start_time) * 1000

                            # Convert Azure result to standard format for display
                            results = {
                                'filename': uploaded_file.name,
                                'timestamp': azure_result.get('timestamp', datetime.now().isoformat()),
                                'document_type': azure_result.get('document_type', 'Medical Document'),
                                'pii_extracted': azure_result.get('pii_extracted', {}),
                                'cleaned_text': azure_result.get('cleaned_text', ''),
                                'ocr_confidence': azure_result.get('ocr_confidence', 0),
                                'use_azure': True,
                                'metrics': {
                                    'timing': {'total_ms': processing_time},
                                    'ocr_stats': {'avg_confidence': azure_result.get('ocr_confidence', 0)}
                                },
                                'pii_entities': [],
                                'pii_summary': {}
                            }

                            # Convert pii_extracted to pii_entities format for compatibility
                            pii_extracted = azure_result.get('pii_extracted', {})
                            for category, data in pii_extracted.items():
                                if isinstance(data, dict):
                                    for key, value in data.items():
                                        if value:
                                            results['pii_entities'].append({
                                                'type': key.upper(),
                                                'value': str(value),
                                                'confidence': 0.9,
                                                'category': category,
                                                'detection_method': 'azure'
                                            })
                                elif isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            for k, v in item.items():
                                                results['pii_entities'].append({
                                                    'type': k.upper(),
                                                    'value': str(v),
                                                    'confidence': 0.9,
                                                    'category': category,
                                                    'detection_method': 'azure'
                                                })
                                        else:
                                            results['pii_entities'].append({
                                                'type': category.upper(),
                                                'value': str(item),
                                                'confidence': 0.9,
                                                'category': category,
                                                'detection_method': 'azure'
                                            })

                            st.session_state.results = results
                            st.session_state.processed = True

                            st.success(f"Processing complete in {processing_time:.0f}ms!")
                            st.info(f"Document Type: **{results['document_type']}**")
                            st.info("Check the **Results** tab to see the output")

                        except Exception as e:
                            st.error(f"Error processing with Azure: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                elif st.session_state.pipeline is not None:
                    # Use EasyOCR pipeline
                    with st.spinner("Processing with EasyOCR..."):
                        try:
                            start_time = time.time()
                            results = st.session_state.pipeline.process(
                                image_bytes=image_bytes,
                                generate_redacted=True
                            )
                            results['filename'] = uploaded_file.name
                            results['use_azure'] = False

                            st.session_state.results = results
                            st.session_state.processed = True

                            total_time = results.get('metrics', {}).get('timing', {}).get('total_ms', 0)
                            st.success(f"Processing complete in {total_time:.0f}ms!")
                            st.info("Check the **Results** tab to see the output")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                else:
                    st.warning("Please initialize the EasyOCR pipeline first (click button in sidebar)")

    with col2:
        st.subheader("📖 How it Works")

        st.markdown("""
        **Pipeline Steps:**

        1. **Pre-processing** 🖼️
           - Image resizing and grayscale conversion
           - Deskewing (tilt correction)
           - Contrast enhancement (CLAHE)
           - Noise removal
           - Morphological operations

        2. **OCR** 📝
           - Text extraction using EasyOCR/Tesseract
           - Bounding box detection
           - Confidence scoring

        3. **Text Cleaning** 🧹
           - Whitespace normalization
           - OCR error correction
           - Date format normalization

        4. **PII Detection** 🔍
           - Regex pattern matching
           - Named Entity Recognition (NER)
           - Indian ID patterns (Aadhaar, PAN)
           - Medical IDs (UHID, IPD, MRN)

        5. **Redaction** ⬛
           - Color-coded PII masking
           - Bounding box overlay
        """)

        # Sample images
        st.subheader("📎 Sample Documents")
        sample_col1, sample_col2, sample_col3 = st.columns(3)

        with sample_col1:
            if st.button("Load Sample 1"):
                st.session_state.sample = "samples/sample1.jpg"
                st.rerun()
        with sample_col2:
            if st.button("Load Sample 2"):
                st.session_state.sample = "samples/sample2.jpg"
                st.rerun()
        with sample_col3:
            if st.button("Load Sample 3"):
                st.session_state.sample = "samples/sample3.jpg"
                st.rerun()

# ============================================
# TAB 2: RESULTS
# ============================================

with tab2:
    if not st.session_state.processed or st.session_state.results is None:
        st.info("Upload and process a document to see results here")
    else:
        results = st.session_state.results
        is_azure = results.get('use_azure', False)

        # Show which engine was used
        if is_azure:
            st.success(f"Processed with Azure Document Intelligence | Confidence: {results.get('ocr_confidence', 0):.1f}%")
            st.subheader(f"Document Type: {results.get('document_type', 'Unknown')}")

        # For Azure results, show PII Extracted in a nice format
        if is_azure and 'pii_extracted' in results:
            st.subheader("Extracted PII Information")

            pii_extracted = results['pii_extracted']

            # Patient Info
            if 'patient_info' in pii_extracted:
                with st.expander("Patient Information", expanded=True):
                    for key, value in pii_extracted['patient_info'].items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

            # Organization
            if 'organization' in pii_extracted:
                with st.expander("Organization", expanded=True):
                    for key, value in pii_extracted['organization'].items():
                        clean_value = str(value).replace('\n', ' ')
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {clean_value}")

            # Healthcare Providers
            if 'healthcare_providers' in pii_extracted:
                with st.expander("Healthcare Providers", expanded=True):
                    for provider in pii_extracted['healthcare_providers']:
                        provider_str = ", ".join([f"{k}: {v}" for k, v in provider.items()])
                        st.markdown(f"- {provider_str}")

            # Dates
            if 'dates' in pii_extracted:
                with st.expander("Dates Found", expanded=False):
                    st.write(pii_extracted['dates'])

            # Medications
            if 'medications' in pii_extracted:
                with st.expander("Medications", expanded=False):
                    if isinstance(pii_extracted['medications'], list):
                        st.write(pii_extracted['medications'])
                    else:
                        st.write(pii_extracted['medications'])

            # Diagnosis
            if 'diagnosis' in pii_extracted:
                with st.expander("Diagnosis", expanded=True):
                    st.write(pii_extracted['diagnosis'])

        else:
            # Original display for EasyOCR results
            st.subheader("Image Comparison")

            if 'original_image' in results:
                img_col1, img_col2, img_col3 = st.columns(3)

                with img_col1:
                    st.markdown("**Original Image**")
                    original_rgb = cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, use_container_width=True)

                with img_col2:
                    st.markdown("**Processed Image**")
                    st.image(results['processed_image'], use_container_width=True)

                with img_col3:
                    st.markdown("**Redacted Image**")
                    if 'redacted_image' in results:
                        redacted_rgb = cv2.cvtColor(results['redacted_image'], cv2.COLOR_BGR2RGB)
                        st.image(redacted_rgb, use_container_width=True)

            # Color Legend
            if 'legend' in results and results.get('pii_summary'):
                st.subheader("Redaction Color Legend")
                pii_summary = results['pii_summary']
                if pii_summary:
                    legend_cols = st.columns(min(len(pii_summary), 4))
                    colors = {
                        'NAME': '#FF0000',
                        'DATE': '#FF8000',
                        'PHONE': '#00FF00',
                        'ID_NUMBER': '#FF00FF',
                        'AADHAAR': '#800080',
                        'PAN': '#008080',
                        'AGE': '#FFFF00',
                        'GENDER': '#0080FF',
                        'LOCATION': '#FFA500',
                        'ORGANIZATION': '#808080',
                        'MEDICAL_ID': '#800000',
                    }

                    for i, (pii_type, items) in enumerate(pii_summary.items()):
                        with legend_cols[i % len(legend_cols)]:
                            color = colors.get(pii_type, '#000000')
                            st.markdown(f"""
                            <div style="background-color: {color}; color: white; padding: 5px 10px;
                                        border-radius: 5px; text-align: center; margin: 2px;">
                                {pii_type} ({len(items)})
                            </div>
                            """, unsafe_allow_html=True)

        # PII Detected Table
        st.subheader("🔍 Detected PII Entities")

        if results.get('pii_entities'):
            # Convert to display format
            pii_data = []
            for entity in results['pii_entities']:
                pii_data.append({
                    'Type': entity.get('type', 'Unknown'),
                    'Value': str(entity.get('value', ''))[:50] + ('...' if len(str(entity.get('value', ''))) > 50 else ''),
                    'Confidence': f"{entity.get('confidence', 0)*100:.1f}%",
                    'Category': entity.get('category', 'Unknown'),
                    'Method': entity.get('detection_method', 'Unknown')
                })

            st.dataframe(pii_data, use_container_width=True)
        else:
            st.warning("No PII entities detected in this document")

        # Download buttons
        st.subheader("Download Results")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            # Download redacted image (only for EasyOCR)
            if 'redacted_image' in results:
                _, buffer = cv2.imencode('.jpg', results['redacted_image'])
                st.download_button(
                    label="Download Redacted Image",
                    data=buffer.tobytes(),
                    file_name="redacted_image.jpg",
                    mime="image/jpeg"
                )

        with dl_col2:
            # Download JSON report
            if is_azure:
                report = {
                    'timestamp': results.get('timestamp', ''),
                    'document_type': results.get('document_type', ''),
                    'pii_extracted': results.get('pii_extracted', {}),
                    'ocr_confidence': results.get('ocr_confidence', 0)
                }
            else:
                report = {
                    'timestamp': results.get('timestamp', ''),
                    'metrics': results.get('metrics', {}),
                    'structured_data': results.get('structured_data', {}),
                    'pii_entities': results.get('pii_entities', [])
                }

            st.download_button(
                label="Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name="pii_report.json",
                mime="application/json"
            )

        with dl_col3:
            # Download extracted text
            st.download_button(
                label="Download Extracted Text",
                data=results.get('cleaned_text', ''),
                file_name="extracted_text.txt",
                mime="text/plain"
            )

# ============================================
# TAB 3: EXTRACTED DATA
# ============================================

with tab3:
    if not st.session_state.processed or st.session_state.results is None:
        st.info("Upload and process a document to see extracted data here")
    else:
        results = st.session_state.results
        is_azure = results.get('use_azure', False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Structured Data")

            if is_azure and 'pii_extracted' in results:
                # Azure format
                pii_extracted = results['pii_extracted']
                for section, data in pii_extracted.items():
                    with st.expander(f"{section.upper().replace('_', ' ')}", expanded=True):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                clean_val = str(value).replace('\n', ' ')
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {clean_val}")
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    st.markdown("- " + ", ".join([f"{k}: {v}" for k, v in item.items()]))
                                else:
                                    st.markdown(f"- {item}")
                        else:
                            st.write(data)
            elif results.get('structured_data'):
                for section, data in results['structured_data'].items():
                    with st.expander(f"{section.upper()}", expanded=True):
                        for key, value in data.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.warning("No structured data could be extracted")

            # PII by Category
            if results.get('pii_entities'):
                st.subheader("PII by Category")

                categories = {}
                for entity in results['pii_entities']:
                    cat = entity.get('category', 'general')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(entity)

                for category, entities in categories.items():
                    with st.expander(f"{category.upper()} ({len(entities)} items)"):
                        for e in entities:
                            st.markdown(f"- **{e.get('type', 'Unknown')}**: {e.get('value', '')}")

        with col2:
            st.subheader("Extracted Text")

            if results.get('raw_text'):
                with st.expander("Raw OCR Text", expanded=False):
                    st.text_area("Raw Text", results.get('raw_text', ''), height=300, disabled=True)

            with st.expander("Cleaned Text", expanded=True):
                st.text_area("Cleaned Text", results.get('cleaned_text', ''), height=300, disabled=True)

            # Show JSON output for Azure
            if is_azure:
                st.subheader("JSON Output")
                with st.expander("Full JSON Result", expanded=False):
                    output = {
                        'document_type': results.get('document_type', ''),
                        'pii_extracted': results.get('pii_extracted', {})
                    }
                    st.json(output)

# ============================================
# TAB 4: METRICS
# ============================================

with tab4:
    if not st.session_state.processed or st.session_state.results is None:
        st.info("Upload and process a document to see metrics here")
    else:
        results = st.session_state.results
        is_azure = results.get('use_azure', False)
        metrics = results.get('metrics', {})

        st.subheader("Processing Metrics")

        if is_azure:
            # Azure metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("OCR Engine", "Azure Document Intelligence")
            with col2:
                st.metric("Processing Time", f"{metrics.get('timing', {}).get('total_ms', 0):.0f}ms")
            with col3:
                st.metric("OCR Confidence", f"{results.get('ocr_confidence', 0):.1f}%")

            st.markdown("---")

            # PII Statistics for Azure
            st.subheader("PII Statistics")
            pii_extracted = results.get('pii_extracted', {})
            pii_counts = {}
            for category, data in pii_extracted.items():
                if isinstance(data, dict):
                    pii_counts[category] = len(data)
                elif isinstance(data, list):
                    pii_counts[category] = len(data)
                else:
                    pii_counts[category] = 1

            if pii_counts:
                import pandas as pd
                pii_df = pd.DataFrame([
                    {'Category': k, 'Count': v}
                    for k, v in pii_counts.items()
                ])
                st.bar_chart(pii_df.set_index('Category'))

        else:
            # EasyOCR metrics
            st.subheader("Processing Time")

            timing = metrics.get('timing', {})
            time_col1, time_col2, time_col3, time_col4, time_col5 = st.columns(5)

            with time_col1:
                st.metric("Pre-processing", f"{timing.get('preprocessing_ms', 0):.0f}ms")
            with time_col2:
                st.metric("OCR", f"{timing.get('ocr_ms', 0):.0f}ms")
            with time_col3:
                st.metric("Text Cleaning", f"{timing.get('text_cleaning_ms', 0):.0f}ms")
            with time_col4:
                st.metric("PII Detection", f"{timing.get('pii_detection_ms', 0):.0f}ms")
            with time_col5:
                st.metric("Total", f"{timing.get('total_ms', 0):.0f}ms")

            st.markdown("---")

            st.subheader("OCR Statistics")

            ocr_stats = metrics.get('ocr_stats', {})
            ocr_col1, ocr_col2, ocr_col3, ocr_col4 = st.columns(4)

            with ocr_col1:
                st.metric("Characters Extracted", ocr_stats.get('total_characters', 0))
            with ocr_col2:
                st.metric("Words Extracted", ocr_stats.get('total_words', 0))
            with ocr_col3:
                st.metric("Text Regions", ocr_stats.get('text_regions', 0))
            with ocr_col4:
                st.metric("Avg Confidence", f"{ocr_stats.get('avg_confidence', 0):.1f}%")

            st.markdown("---")

            st.subheader("PII Statistics")

            pii_stats = metrics.get('pii_stats', {})
            st.metric("Total PII Found", pii_stats.get('total_found', 0))

            if pii_stats.get('by_type'):
                import pandas as pd
                pii_df = pd.DataFrame([
                    {'Type': k, 'Count': v}
                    for k, v in pii_stats['by_type'].items()
                ])
                st.bar_chart(pii_df.set_index('Type'))

        # Full metrics JSON
        with st.expander("Full Metrics JSON"):
            st.json(metrics)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>OCR PII Extraction Pipeline v2.0 | Built with Streamlit</p>
    <p>Supports: Names, Dates, Phone Numbers, Aadhaar, PAN, Medical IDs (UHID, IPD, MRN)</p>
</div>
""", unsafe_allow_html=True)
