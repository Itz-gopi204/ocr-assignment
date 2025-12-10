# OCR Pipeline - Dependencies Document

## System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for EasyOCR)
- Tesseract OCR installed on system

## Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >= 4.8.0 | Image pre-processing (deskew, denoise, binarization) |
| numpy | >= 1.24.0 | Array operations and image manipulation |
| Pillow | >= 10.0.0 | Image loading and redaction drawing |
| pytesseract | >= 0.3.10 | Tesseract OCR Python wrapper |
| easyocr | >= 1.7.0 | Deep learning based OCR (better for handwriting) |
| spacy | >= 3.6.0 | Named Entity Recognition for PII detection |
| regex | >= 2023.6.3 | Advanced regex patterns for PII detection |
| matplotlib | >= 3.7.0 | Visualization of results |
| jupyter | >= 1.0.0 | Jupyter Notebook environment |

## External Dependencies

### Tesseract OCR
Tesseract must be installed separately on your system:

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add to system PATH: `C:\Program Files\Tesseract-OCR`
4. Verify installation: `tesseract --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

## Installation Steps

### Option 1: Using pip (Recommended)
```bash
# Create virtual environment
python -m venv ocr_env

# Activate environment
# Windows:
ocr_env\Scripts\activate
# Linux/Mac:
source ocr_env/bin/activate

# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Option 2: Using conda
```bash
# Create conda environment
conda create -n ocr_env python=3.9

# Activate environment
conda activate ocr_env

# Install packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## GPU Acceleration (Optional)
For faster EasyOCR processing on NVIDIA GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Running the Pipeline

```bash
# Start Jupyter Notebook
jupyter notebook ocr_pii_pipeline.ipynb
```

## Troubleshooting

### "Tesseract not found" Error
Ensure Tesseract is in your system PATH:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### Memory Issues with EasyOCR
EasyOCR loads deep learning models (~1GB). If memory is limited:
- Close other applications
- Use `gpu=False` in EasyOCR initialization
- Process images one at a time

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

## Version Compatibility
Tested on:
- Python 3.9, 3.10, 3.11
- Windows 10/11, Ubuntu 20.04/22.04, macOS 12+
