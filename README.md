# OCR Pipeline - Handwritten Document PII Extraction

An end-to-end OCR pipeline for extracting Personal Identifiable Information (PII) from handwritten medical documents.

## Pipeline Flow

```
Input (JPEG) → Pre-processing → OCR → Text Cleaning → PII Detection → JSON Output
```

## Features

- **Image Pre-processing**: Deskew, denoise, contrast enhancement
- **Multiple OCR Engines**: Azure Document Intelligence (recommended), EasyOCR, Tesseract
- **PII Detection**: Patient names, IDs, dates, doctor names, diagnoses
- **Structured Output**: Clean JSON format with document classification
- **Web Interface**: Streamlit-based UI for easy document processing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Azure (Recommended)

Create a `.env` file:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your_key
```

### 3. Run the Application

**Web Interface:**
```bash
streamlit run streamlit_app.py
```

**Command Line:**
```bash
python azure_ocr.py
```

## Output Format

```json
{
  "document_type": "Hospital Progress Report",
  "pii_extracted": {
    "patient_info": {
      "name": "Santosh Pradhan",
      "age": "26Y",
      "uhid_no": "20250 4110195",
      "ipd_no": "2236927833",
      "bed_no": "10"
    },
    "organization": {
      "name": "INSTITUTE OF MEDICAL SCIENCES & SUM HOSPITAL",
      "university": "SIKSHA 'O' ANUSANDHAN",
      "address": "K-8, Kalinga Nagar, Bhubaneswar"
    },
    "healthcare_providers": [
      {"name": "Dr. Soumya Ranjan Dasn", "reg_no": "17112/2-09", "department": "Psychiatry"}
    ],
    "diagnosis": "BEHAVIORAL DISORDER DUE TO USE OF"
  }
}
```

## Supported Document Types

- Hospital Progress Reports
- Medication Administration Records
- Medical Prescriptions
- Discharge Summaries

## PII Fields Extracted

| Category | Fields |
|----------|--------|
| Patient Info | Name, Age, Sex, UHID, IPD No, Bed No |
| Organization | Hospital Name, University, Address |
| Healthcare Providers | Doctor Names, Registration Numbers, Departments |
| Clinical | Dates, Diagnosis, Medications, Routes |

## OCR Accuracy Comparison

| Engine | Confidence | Handwriting Recognition |
|--------|------------|------------------------|
| Azure Document Intelligence | ~75-80% | Excellent |
| EasyOCR | ~50-60% | Good |
| Tesseract | ~40-50% | Fair |

## Project Structure

```
├── azure_ocr.py           # Azure Document Intelligence integration
├── enhanced_pipeline.py   # EasyOCR/Tesseract pipeline
├── streamlit_app.py       # Web interface
├── ocr_pii_pipeline.ipynb # Jupyter notebook
├── samples/               # Sample document images
├── results_azure/         # Output JSON reports
├── requirements.txt       # Python dependencies
└── DEPENDENCIES.md        # Detailed dependency documentation
```

## Requirements

- Python 3.9+
- Azure Document Intelligence account (for best results)
- See `requirements.txt` for full list
