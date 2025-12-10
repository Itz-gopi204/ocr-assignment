"""
OCR Pipeline Test Script
Run this to process sample images and generate results
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import easyocr
import re
import spacy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

print("Loading libraries...")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded!")

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class PipelineConfig:
    """Configuration for the OCR Pipeline"""
    resize_width: int = 2000
    denoise_strength: int = 10
    ocr_engine: str = "easyocr"
    tesseract_config: str = "--oem 3 --psm 6"
    easyocr_languages: List[str] = None

    def __post_init__(self):
        if self.easyocr_languages is None:
            self.easyocr_languages = ["en"]

# ============================================
# IMAGE PREPROCESSOR
# ============================================

class ImagePreprocessor:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if width > self.config.resize_width:
            ratio = self.config.resize_width / width
            new_height = int(height * ratio)
            image = cv2.resize(image, (self.config.resize_width, new_height),
                             interpolation=cv2.INTER_AREA)
        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None,
                                        self.config.denoise_strength, 7, 21)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = (theta * 180 / np.pi) - 90
                if -45 < angle < 45:
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        original = self.load_image(image_path)
        image = self.resize_image(original.copy())
        gray = self.convert_to_grayscale(image)
        gray = self.deskew(gray)
        gray = self.enhance_contrast(gray)
        gray = self.denoise(gray)
        return original, gray

# ============================================
# OCR ENGINE
# ============================================

class OCREngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.easyocr_reader = None

        if config.ocr_engine in ["easyocr", "both"]:
            print("Initializing EasyOCR... (this may take a moment)")
            self.easyocr_reader = easyocr.Reader(config.easyocr_languages, gpu=False, verbose=False)
            print("EasyOCR initialized!")

    def extract_with_easyocr(self, image: np.ndarray) -> Dict:
        results = self.easyocr_reader.readtext(image)

        text_parts = []
        boxes = []

        for (bbox, text, confidence) in results:
            text_parts.append(text)
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            boxes.append({
                'text': text,
                'confidence': confidence * 100,
                'bbox': (int(min(x_coords)), int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))),
                'polygon': bbox
            })

        return {
            'engine': 'easyocr',
            'text': '\n'.join(text_parts),
            'boxes': boxes
        }

    def extract(self, image: np.ndarray) -> Dict:
        return self.extract_with_easyocr(image)

# ============================================
# TEXT CLEANER
# ============================================

class TextCleaner:
    def __init__(self):
        pass

    def remove_extra_whitespace(self, text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def remove_noise_characters(self, text: str) -> str:
        text = re.sub(r'(?<![\w])[\_\-\~\^\*\#\@]+(?![\w])', '', text)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 1]
        return '\n'.join(cleaned_lines)

    def fix_common_ocr_errors(self, text: str) -> str:
        text = re.sub(r'\bPaticnt\b', 'Patient', text, flags=re.IGNORECASE)
        text = re.sub(r'\bNamc\b', 'Name', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAgc\b', 'Age', text, flags=re.IGNORECASE)
        text = re.sub(r'\bScx\b', 'Sex', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDatc\b', 'Date', text, flags=re.IGNORECASE)
        return text

    def normalize_dates(self, text: str) -> str:
        text = re.sub(r'(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{2,4})',
                     r'\1/\2/\3', text)
        return text

    def clean(self, text: str) -> str:
        text = self.remove_extra_whitespace(text)
        text = self.remove_noise_characters(text)
        text = self.fix_common_ocr_errors(text)
        text = self.normalize_dates(text)
        return text

# ============================================
# PII DETECTION
# ============================================

@dataclass
class PIIEntity:
    type: str
    value: str
    start: int
    end: int
    confidence: float

    def to_dict(self):
        return {
            'type': self.type,
            'value': self.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }

class PIIDetector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nlp = nlp

        self.patterns = {
            'PHONE': [
                r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b',
                r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{4}\b',
            ],
            'DATE': [
                r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',
                r'\b\d{1,2}[\s\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]\d{2,4}\b',
            ],
            'ID_NUMBER': [
                r'\b(?:UHID|IPD|OPD|MRN|Reg)[\s\.\-:No]*[\:\s]*([A-Z0-9\-]+)\b',
                r'\b(?:Bed)[\s\.\-:No]*[\:\s]*(\d+)\b',
            ],
            'AGE': [
                r'\b(?:Age)[\s:]*(\d+)[\s]?(?:Y|yr|years?|yrs?)?\b',
                r'\b(\d{1,3})[\s]?(?:Y|yr|years?|yrs?)\b',
            ],
            'GENDER': [
                r'\b(?:Sex|Gender)[\s:]*([MF]|Male|Female)\b',
            ],
            'AADHAAR': [
                r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
            ]
        }

        self.name_patterns = [
            r'(?:Patient[\s]*Name|Name)[\s:]*([A-Za-z]+(?:[\s\.]+[A-Za-z]+){1,3})',
            r'(?:Dr\.?|Doctor)[\s]*([A-Za-z]+(?:[\s\.]+[A-Za-z]+){0,2})',
            r'(?:Consultant|Resident|Physician)[\s:]*([A-Za-z]+(?:[\s\.]+[A-Za-z]+){0,2})',
        ]

    def detect_with_regex(self, text: str) -> List[PIIEntity]:
        entities = []

        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(1) if match.lastindex else match.group(0)
                    entities.append(PIIEntity(
                        type=pii_type,
                        value=value.strip(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))

        for pattern in self.name_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                entities.append(PIIEntity(
                    type='NAME',
                    value=value.strip(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))

        return entities

    def detect_with_ner(self, text: str) -> List[PIIEntity]:
        entities = []
        doc = self.nlp(text)

        type_mapping = {
            'PERSON': 'NAME',
            'DATE': 'DATE',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'ORG': 'ORGANIZATION',
        }

        for ent in doc.ents:
            if ent.label_ in type_mapping:
                entities.append(PIIEntity(
                    type=type_mapping[ent.label_],
                    value=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.75
                ))

        return entities

    def merge_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        if not entities:
            return []

        sorted_entities = sorted(entities, key=lambda x: (x.start, -x.confidence))
        merged = [sorted_entities[0]]

        for entity in sorted_entities[1:]:
            last = merged[-1]
            if entity.start < last.end:
                if entity.confidence > last.confidence or len(entity.value) > len(last.value):
                    merged[-1] = entity
            else:
                merged.append(entity)

        return merged

    def detect(self, text: str) -> List[PIIEntity]:
        regex_entities = self.detect_with_regex(text)
        ner_entities = self.detect_with_ner(text)

        all_entities = regex_entities + ner_entities
        merged_entities = self.merge_entities(all_entities)

        seen = set()
        unique_entities = []
        for entity in merged_entities:
            key = (entity.type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

# ============================================
# IMAGE REDACTOR
# ============================================

class ImageRedactor:
    def __init__(self, redaction_color=(0, 0, 0)):
        self.redaction_color = redaction_color

    def find_text_bbox(self, ocr_boxes: List[Dict], pii_value: str) -> Optional[Tuple]:
        pii_lower = pii_value.lower().strip()

        for box in ocr_boxes:
            box_text = box['text'].lower().strip()
            if pii_lower in box_text or box_text in pii_lower:
                return box['bbox']

        pii_words = pii_lower.split()
        for word in pii_words:
            if len(word) > 2:
                for box in ocr_boxes:
                    if word in box['text'].lower():
                        return box['bbox']

        return None

    def redact_image(self, image: np.ndarray,
                     ocr_result: Dict,
                     pii_entities: List[PIIEntity]) -> np.ndarray:
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_image)
        ocr_boxes = ocr_result.get('boxes', [])

        for entity in pii_entities:
            bbox = self.find_text_bbox(ocr_boxes, entity.value)
            if bbox:
                x, y, w, h = bbox
                padding = 5
                draw.rectangle(
                    [x - padding, y - padding, x + w + padding, y + h + padding],
                    fill=self.redaction_color
                )

        redacted = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return redacted

# ============================================
# MAIN PIPELINE
# ============================================

class OCRPIIPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.text_cleaner = TextCleaner()
        self.pii_detector = PIIDetector(self.config)
        self.redactor = ImageRedactor()

    def process(self, image_path: str, generate_redacted: bool = True) -> Dict:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        results = {
            'input_file': image_path,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }

        # Stage 1: Pre-processing
        print("\n[1/5] Pre-processing image...")
        original, processed = self.preprocessor.preprocess(image_path)
        results['stages']['preprocessing'] = {'status': 'completed'}
        print(f"  - Original size: {original.shape}")
        print(f"  - Processed size: {processed.shape}")

        # Stage 2: OCR
        print("\n[2/5] Extracting text with OCR...")
        ocr_result = self.ocr_engine.extract(processed)
        results['stages']['ocr'] = {'status': 'completed'}
        print(f"  - Extracted {len(ocr_result['text'])} characters")
        print(f"  - Found {len(ocr_result.get('boxes', []))} text regions")

        # Stage 3: Text Cleaning
        print("\n[3/5] Cleaning extracted text...")
        raw_text = ocr_result['text']
        cleaned_text = self.text_cleaner.clean(raw_text)
        results['raw_text'] = raw_text
        results['cleaned_text'] = cleaned_text
        print(f"  - Raw text length: {len(raw_text)}")
        print(f"  - Cleaned text length: {len(cleaned_text)}")

        # Stage 4: PII Detection
        print("\n[4/5] Detecting PII...")
        pii_entities = self.pii_detector.detect(cleaned_text)
        results['pii_entities'] = [e.to_dict() for e in pii_entities]
        print(f"  - Found {len(pii_entities)} PII entities")

        # Group PII by type
        pii_by_type = {}
        for entity in pii_entities:
            if entity.type not in pii_by_type:
                pii_by_type[entity.type] = []
            pii_by_type[entity.type].append(entity.value)
        results['pii_summary'] = pii_by_type

        for pii_type, values in pii_by_type.items():
            print(f"    - {pii_type}: {values}")

        # Stage 5: Image Redaction
        if generate_redacted:
            print("\n[5/5] Generating redacted image...")
            redacted_image = self.redactor.redact_image(processed, ocr_result, pii_entities)
            results['redacted_image'] = redacted_image

        results['original_image'] = original
        results['processed_image'] = processed

        print(f"\n{'='*60}")
        print("Processing completed!")
        print(f"{'='*60}")

        return results

# ============================================
# UTILITY FUNCTIONS
# ============================================

def save_results(result: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(result['input_file']))[0]

    # Save processed image
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_processed.jpg"),
        result['processed_image']
    )

    # Save redacted image
    if 'redacted_image' in result:
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_redacted.jpg"),
            result['redacted_image']
        )

    # Save JSON report
    report = {
        'input_file': result['input_file'],
        'timestamp': result['timestamp'],
        'cleaned_text': result['cleaned_text'],
        'pii_entities': result['pii_entities'],
        'pii_summary': result['pii_summary']
    }

    with open(os.path.join(output_dir, f"{base_name}_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Save extracted text
    with open(os.path.join(output_dir, f"{base_name}_text.txt"), 'w', encoding='utf-8') as f:
        f.write("RAW EXTRACTED TEXT:\n")
        f.write("="*50 + "\n")
        f.write(result['raw_text'])
        f.write("\n\n")
        f.write("CLEANED TEXT:\n")
        f.write("="*50 + "\n")
        f.write(result['cleaned_text'])

    print(f"Results saved to: {output_dir}/{base_name}_*")

def visualize_and_save(result: Dict, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Original image
    axes[0].imshow(cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Processed image
    axes[1].imshow(result['processed_image'], cmap='gray')
    axes[1].set_title('Processed Image', fontsize=14)
    axes[1].axis('off')

    # Redacted image
    if 'redacted_image' in result:
        axes[2].imshow(cv2.cvtColor(result['redacted_image'], cv2.COLOR_BGR2RGB))
        axes[2].set_title('Redacted Image', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def print_pii_report(result: Dict):
    print("\n" + "="*60)
    print("PII DETECTION REPORT")
    print("="*60)
    print(f"File: {result['input_file']}")
    print("-"*60)

    if 'pii_summary' in result:
        print("\nDetected PII Entities:")
        for pii_type, values in result['pii_summary'].items():
            print(f"\n  [{pii_type}]")
            for value in values:
                print(f"    - {value}")

    print("\n" + "="*60)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    SAMPLES_DIR = "samples"
    OUTPUT_DIR = "results"

    sample_images = [
        os.path.join(SAMPLES_DIR, "sample1.jpg"),
        os.path.join(SAMPLES_DIR, "sample2.jpg"),
        os.path.join(SAMPLES_DIR, "sample3.jpg"),
    ]

    # Verify samples exist
    print("\nChecking sample images...")
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"  Found: {img_path}")
        else:
            print(f"  MISSING: {img_path}")

    # Initialize pipeline
    print("\nInitializing pipeline...")
    config = PipelineConfig()
    pipeline = OCRPIIPipeline(config)

    # Process each sample
    all_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, img_path in enumerate(sample_images, 1):
        if os.path.exists(img_path):
            print(f"\n\n{'#'*60}")
            print(f"PROCESSING SAMPLE {i}")
            print(f"{'#'*60}")

            result = pipeline.process(img_path, generate_redacted=True)
            all_results.append(result)

            # Print PII report
            print_pii_report(result)

            # Save results
            save_results(result, OUTPUT_DIR)

            # Save visualization
            vis_path = os.path.join(OUTPUT_DIR, f"sample{i}_visualization.png")
            visualize_and_save(result, vis_path)

    # Final summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for i, result in enumerate(all_results, 1):
        print(f"\n--- Sample {i}: {os.path.basename(result['input_file'])} ---")
        if 'pii_summary' in result:
            total_pii = sum(len(v) for v in result['pii_summary'].values())
            print(f"Total PII Found: {total_pii}")
            for pii_type, values in result['pii_summary'].items():
                print(f"  {pii_type}: {len(values)} - {values[:3]}{'...' if len(values) > 3 else ''}")

    print(f"\n{'='*70}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}")
