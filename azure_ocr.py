"""
Azure Document Intelligence Integration for OCR PII Extraction Pipeline
=========================================================================

This module uses Azure Document Intelligence (Form Recognizer) for superior
handwriting recognition and structured data extraction from medical documents.

Setup:
1. Create Azure Document Intelligence resource in Azure Portal
2. Get your endpoint and API key
3. Create .env file with:
   - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint
   - AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your_key
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use os.environ directly

# Azure SDK imports
try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure SDK not installed. Run: pip install azure-ai-formrecognizer")


@dataclass
class AzureConfig:
    """Configuration for Azure Document Intelligence"""
    endpoint: str = ""
    api_key: str = ""
    model_id: str = "prebuilt-document"  # Can use "prebuilt-read" for pure OCR

    def __post_init__(self):
        # Try to load from environment variables if not provided
        if not self.endpoint:
            self.endpoint = os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
                os.getenv("AZURE_DOCUMENT_ENDPOINT", "")
            )
        if not self.api_key:
            self.api_key = os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
                os.getenv("AZURE_DOCUMENT_KEY", "")
            )

    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)


# Get credentials from environment variables (loaded from .env)
DEFAULT_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
DEFAULT_API_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", "")


class AzureDocumentOCR:
    """
    Azure Document Intelligence OCR Engine

    Provides superior handwriting recognition and structured data extraction
    compared to traditional OCR engines like Tesseract or EasyOCR.
    """

    def __init__(self, config: AzureConfig = None):
        self.config = config or AzureConfig()
        self.client = None

        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK not installed. Run: pip install azure-ai-formrecognizer")

        if self.config.is_configured():
            self.client = DocumentAnalysisClient(
                endpoint=self.config.endpoint,
                credential=AzureKeyCredential(self.config.api_key)
            )
            print("Azure Document Intelligence client initialized successfully!")
        else:
            print("Warning: Azure credentials not configured. Set AZURE_DOCUMENT_ENDPOINT and AZURE_DOCUMENT_KEY")

    def analyze_document(self, image_path: str = None, image_bytes: bytes = None) -> Dict:
        """
        Analyze a document image using Azure Document Intelligence

        Args:
            image_path: Path to image file
            image_bytes: Raw image bytes

        Returns:
            Dictionary with extracted text and structured data
        """
        if not self.client:
            raise ValueError("Azure client not initialized. Check your credentials.")

        # Start analysis
        if image_path:
            with open(image_path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    self.config.model_id,
                    f
                )
        elif image_bytes:
            poller = self.client.begin_analyze_document(
                self.config.model_id,
                image_bytes
            )
        else:
            raise ValueError("Provide either image_path or image_bytes")

        # Wait for result
        result = poller.result()

        # Process and return structured result
        return self._process_result(result)

    def _process_result(self, result) -> Dict:
        """Process Azure analysis result into our standard format"""

        # Extract full text
        full_text = ""
        text_boxes = []
        confidences = []

        for page in result.pages:
            for line in page.lines:
                full_text += line.content + "\n"

                # Get bounding box
                if line.polygon:
                    x_coords = [p.x for p in line.polygon]
                    y_coords = [p.y for p in line.polygon]
                    bbox = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    )
                else:
                    bbox = (0, 0, 0, 0)

                text_boxes.append({
                    'text': line.content,
                    'confidence': 95.0,  # Azure doesn't provide line confidence
                    'bbox': bbox
                })

            # Get word-level confidence
            for word in page.words:
                if word.confidence:
                    confidences.append(word.confidence * 100)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Extract key-value pairs
        key_value_pairs = {}
        if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
            for kv in result.key_value_pairs:
                if kv.key and kv.value:
                    key = kv.key.content.strip()
                    value = kv.value.content.strip()
                    key_value_pairs[key] = value

        # Extract tables
        tables = []
        if hasattr(result, 'tables') and result.tables:
            for table in result.tables:
                table_data = []
                for cell in table.cells:
                    table_data.append({
                        'row': cell.row_index,
                        'col': cell.column_index,
                        'content': cell.content
                    })
                tables.append(table_data)

        return {
            'engine': 'azure_document_intelligence',
            'text': full_text.strip(),
            'boxes': text_boxes,
            'avg_confidence': avg_confidence,
            'key_value_pairs': key_value_pairs,
            'tables': tables,
            'raw_result': result
        }


class AzurePIIExtractor:
    """
    Extract PII from Azure Document Intelligence results
    Optimized for medical documents
    """

    def __init__(self):
        # Patterns for medical document PII
        self.patterns = {
            'patient_name': [
                r'(?:Patient\s*Name|Pat\.?\s*Name|Name)[:\s]+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})',
                r'(?:Patient)[:\s]+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})',
            ],
            'age': [
                r'(?:Age)[:\s]*(\d{1,3})\s*(?:Y|yr|years?|yrs?)?',
                r'(\d{1,3})\s*(?:Y|Yr|Years)\b',
            ],
            'sex': [
                r'(?:Sex|Gender)[:\s]*([MF]|Male|Female)',
            ],
            'uhid': [
                r'(?:UHID|U\.H\.I\.D)[:\s\.No]*[:\s]*([A-Z0-9\-/]+)',
            ],
            'ipd_no': [
                r'(?:IPD|I\.P\.D)[:\s\.No\-]*[:\s]*([A-Z0-9\-/]+)',
            ],
            'bed_no': [
                r'(?:Bed|Ward)[:\s\.No\-]*[:\s]*([A-Z0-9\-]+)',
            ],
            'date': [
                r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            ],
            'doctor_name': [
                r'(?:Dr\.?|Doctor)\s+([A-Za-z]+(?:\s+[A-Za-z\.]+){0,3})',
                r'(?:Consultant|Resident|Physician)[:\s]+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})',
            ],
            'phone': [
                r'(?:\+91[\-\s]?)?([6-9]\d{9})\b',
                r'(?:Mobile|Phone|Tel|Contact)[:\s]*([\d\-\s\+]{10,})',
            ],
            'registration_no': [
                r'(?:Reg|Registration|Regd)[:\s\.No\-]*[:\s]*([A-Z0-9\-/]+)',
            ],
        }

        # Medical document keywords for classification
        self.doc_type_keywords = {
            'Medication Administration Record': ['medication', 'drug', 'dose', 'route', 'frequency'],
            'Hospital Progress Report': ['progress', 'report', 'notes', 'treatment', 'advice'],
            'Medical Prescription': ['prescription', 'rx', 'prescribed'],
            'Discharge Summary': ['discharge', 'summary', 'admitted', 'discharged'],
        }

    def extract_from_text(self, text: str, key_value_pairs: Dict = None) -> Dict:
        """
        Extract PII from text and key-value pairs

        Args:
            text: Full extracted text
            key_value_pairs: Key-value pairs from Azure

        Returns:
            Structured PII data
        """
        text_upper = text.upper()

        # Determine document type
        doc_type = self._classify_document(text_upper)

        # Extract PII
        pii_extracted = {}

        # Patient Info
        patient_info = self._extract_patient_info(text, key_value_pairs)
        if patient_info:
            pii_extracted['patient_info'] = patient_info

        # Organization Info
        org_info = self._extract_organization(text)
        if org_info:
            pii_extracted['organization'] = org_info

        # Healthcare Providers
        providers = self._extract_healthcare_providers(text)
        if providers:
            pii_extracted['healthcare_providers'] = providers

        # Dates
        dates = self._extract_dates(text)
        if dates:
            pii_extracted['dates'] = dates

        # Medications (for medication records)
        if 'Medication' in doc_type:
            medications = self._extract_medications(text)
            if medications:
                pii_extracted['medications'] = medications
            routes = self._extract_routes(text)
            if routes:
                pii_extracted['medical_routes'] = routes

        # Diagnosis
        diagnosis = self._extract_diagnosis(text)
        if diagnosis:
            pii_extracted['diagnosis'] = diagnosis

        # Phone numbers
        phones = self._extract_pattern(text, 'phone')
        if phones:
            pii_extracted['mobile_no'] = phones[0] if len(phones) == 1 else phones

        return {
            'document_type': doc_type,
            'pii_extracted': pii_extracted
        }

    def _classify_document(self, text: str) -> str:
        """Classify document type based on content"""
        text_lower = text.lower()

        for doc_type, keywords in self.doc_type_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:
                return doc_type

        return "Medical Document"

    def _extract_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Extract values matching a pattern"""
        results = []
        patterns = self.patterns.get(pattern_name, [])

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                value = value.strip()
                if value and len(value) > 1:
                    results.append(value)

        return list(set(results))  # Remove duplicates

    def _extract_patient_info(self, text: str, key_value_pairs: Dict = None) -> Dict:
        """Extract patient information"""
        patient = {}

        # From key-value pairs first (more reliable)
        if key_value_pairs:
            kv_mapping = {
                'Patient Name': 'name',
                'Name': 'name',
                'Age': 'age',
                'Sex': 'sex',
                'Gender': 'sex',
                'UHID': 'uhid_no',
                'IPD': 'ipd_no',
                'IPD No': 'ipd_no',
                'Bed': 'bed_no',
                'Bed No': 'bed_no',
            }

            for kv_key, field in kv_mapping.items():
                for key, value in key_value_pairs.items():
                    if kv_key.lower() in key.lower():
                        patient[field] = value
                        break

        # From regex patterns (fallback)
        if 'name' not in patient:
            names = self._extract_pattern(text, 'patient_name')
            if names:
                patient['name'] = names[0]

        if 'age' not in patient:
            ages = self._extract_pattern(text, 'age')
            if ages:
                # Clean age - take only numeric part up to 3 digits
                age_val = ages[0]
                age_num = re.search(r'(\d{1,3})', age_val)
                if age_num:
                    patient['age'] = age_num.group(1) + 'Y'

        if 'sex' not in patient:
            sexes = self._extract_pattern(text, 'sex')
            if sexes:
                sex_val = sexes[0].upper()
                if sex_val in ['M', 'MALE']:
                    patient['sex'] = 'Male'
                elif sex_val in ['F', 'FEMALE']:
                    patient['sex'] = 'Female'
                else:
                    patient['sex'] = sex_val

        if 'uhid_no' not in patient:
            uhids = self._extract_pattern(text, 'uhid')
            if uhids:
                patient['uhid_no'] = uhids[0]

        if 'ipd_no' not in patient:
            ipds = self._extract_pattern(text, 'ipd_no')
            if ipds:
                patient['ipd_no'] = ipds[0]

        if 'bed_no' not in patient:
            beds = self._extract_pattern(text, 'bed_no')
            if beds:
                patient['bed_no'] = beds[0]

        return patient

    def _extract_organization(self, text: str) -> Dict:
        """Extract organization/hospital information"""
        org = {}

        # Hospital name patterns
        hospital_patterns = [
            r'((?:Institute\s+of\s+)?Medical\s+Sciences\s*(?:&|and)?\s*SUM\s+Hospital)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Hospital)',
            r'(Hospital\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

        for pattern in hospital_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                org['name'] = match.group(1).strip()
                break

        # University
        uni_match = re.search(r"(Siksha\s*['\"]?O['\"]?\s*Anusandhan[^)]*)", text, re.IGNORECASE)
        if uni_match:
            org['university'] = uni_match.group(1).strip()

        # Address
        addr_match = re.search(r'(K-?\d+,?\s*Kalinga\s*Nagar,?\s*Bhubaneswar)', text, re.IGNORECASE)
        if addr_match:
            org['address'] = addr_match.group(1).strip()

        return org

    def _extract_healthcare_providers(self, text: str) -> List[Dict]:
        """Extract doctor/healthcare provider information"""
        providers = []
        seen_names = set()

        # Better doctor name patterns
        doc_patterns = [
            r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s*\n|\s+Asst|\s+Professor|\s+Resident)',
            r'Dr\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]

        for pattern in doc_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                # Filter out noise
                if len(name) > 4 and name.lower() not in seen_names:
                    # Skip if it's actually a diagnosis or department
                    noise_words = ['MENTAL', 'BEHAVIORAL', 'DISORDER', 'SIGNATURE', 'MOBILE', 'FULL']
                    if not any(nw in name.upper() for nw in noise_words):
                        seen_names.add(name.lower())
                        providers.append({'name': f"Dr. {name}"})

        # Extract registration numbers and associate with doctors
        reg_patterns = [
            r'(?:Regd?\.?\s*No)[:\s\-]*([0-9\-/]+)',
            r'Reg(?:istration)?[:\s\.No\-]*[:\s]*(\d+[/\-]\d+)',
        ]

        for pattern in reg_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and providers:
                providers[-1]['reg_no'] = match.group(1)
                break

        # Extract department/designation
        dept_match = re.search(r'Dept\.?\s*(?:of\s+)?([A-Za-z]+)', text, re.IGNORECASE)
        if dept_match and providers:
            providers[-1]['department'] = dept_match.group(1)

        return providers

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        return self._extract_pattern(text, 'date')

    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication names"""
        medications = []

        # Common medication patterns
        med_patterns = [
            r'(?:Tab|TAB|Cap|CAP|Inj|INJ|Syp|SYP)[.\-\s]*([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'([A-Z]{2,}[\-\s]?[A-Z]+)',  # All caps medications
        ]

        for pattern in med_patterns:
            for match in re.finditer(pattern, text):
                med = match.group(1) if match.lastindex else match.group(0)
                med = med.strip()
                if len(med) > 2 and med.upper() not in ['THE', 'AND', 'FOR', 'WITH']:
                    medications.append(med)

        return list(set(medications))[:15]  # Limit to 15

    def _extract_routes(self, text: str) -> List[str]:
        """Extract medical administration routes"""
        routes = []
        route_patterns = [r'\b(IV|IM|PO|SC|ID|SL|PR|INH|TOP|PIO)\b']

        for pattern in route_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                routes.append(match.group(1).upper())

        return list(set(routes))

    def _extract_diagnosis(self, text: str) -> Optional[str]:
        """Extract diagnosis information"""
        diag_patterns = [
            r'(?:Diagnosis|Dx|D/x)[:\s]+([^\n]+)',
            r'(?:Mental|Behavioral)\s+(?:Disorder|disorder)[^\n]*',
            r'(?:Alcohol|Substance)\s+(?:Dependence|Abuse)[^\n]*',
        ]

        for pattern in diag_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)

        return None


def process_with_azure(image_path: str = None, image_bytes: bytes = None,
                       endpoint: str = None, api_key: str = None) -> Dict:
    """
    Process a document using Azure Document Intelligence

    Args:
        image_path: Path to image file
        image_bytes: Raw image bytes
        endpoint: Azure endpoint URL (uses default if not provided)
        api_key: Azure API key (uses default if not provided)

    Returns:
        Dictionary with document_type and pii_extracted
    """
    # Use defaults if not provided
    endpoint = endpoint or DEFAULT_ENDPOINT
    api_key = api_key or DEFAULT_API_KEY

    # Initialize Azure OCR
    config = AzureConfig(endpoint=endpoint, api_key=api_key)

    if not config.is_configured():
        raise ValueError(
            "Azure credentials not configured. "
            "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_API_KEY "
            "or pass endpoint and api_key parameters."
        )

    ocr = AzureDocumentOCR(config)

    # Analyze document
    result = ocr.analyze_document(image_path=image_path, image_bytes=image_bytes)

    # Extract PII
    extractor = AzurePIIExtractor()
    pii_result = extractor.extract_from_text(
        result['text'],
        result.get('key_value_pairs', {})
    )

    # Add metadata
    pii_result['timestamp'] = datetime.now().isoformat()
    pii_result['ocr_confidence'] = result['avg_confidence']
    pii_result['cleaned_text'] = result['text']

    return pii_result


# ============================================
# MAIN - Test with samples
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Azure Document Intelligence OCR Pipeline")
    print("=" * 60)

    # Use default credentials
    print(f"\n[OK] Using Azure Document Intelligence")
    print(f"   Endpoint: {DEFAULT_ENDPOINT}")

    # Process all samples
    samples = [
        "samples/sample1.jpg",
        "samples/sample2.jpg",
        "samples/sample3.jpg",
    ]

    results = {}

    for sample_path in samples:
        if os.path.exists(sample_path):
            sample_name = os.path.basename(sample_path).replace('.jpg', '')
            print(f"\n{'#' * 60}")
            print(f"Processing: {sample_path}")
            print(f"{'#' * 60}")

            try:
                result = process_with_azure(image_path=sample_path)
                results[sample_name] = {
                    'document_type': result['document_type'],
                    'pii_extracted': result['pii_extracted']
                }

                print(f"\nDocument Type: {result['document_type']}")
                print(f"OCR Confidence: {result.get('ocr_confidence', 'N/A')}%")
                print(f"\nPII Extracted:")
                for key, value in result['pii_extracted'].items():
                    print(f"  {key}: {value}")

                # Save individual result
                output_path = f"results_azure/{sample_name}_report.json"
                os.makedirs("results_azure", exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nSaved to: {output_path}")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Sample not found: {sample_path}")

    # Save combined results
    if results:
        print(f"\n{'=' * 60}")
        print("COMBINED RESULTS:")
        print("=" * 60)
        print(json.dumps(results, indent=2))

        with open("results_azure/all_samples_report.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved combined results to: results_azure/all_samples_report.json")
