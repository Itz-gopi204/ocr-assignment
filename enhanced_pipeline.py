"""
Enhanced OCR Pipeline for Handwritten Document PII Extraction
Version 2.0 - With improved detection, structured extraction, and metrics
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import easyocr
import re
import spacy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import os
import json
from datetime import datetime
import time

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class PipelineConfig:
    """Enhanced configuration for the OCR Pipeline"""
    # Pre-processing settings
    resize_width: int = 2500  # Larger for better OCR
    denoise_strength: int = 8  # Reduced to preserve details
    apply_morphology: bool = False  # Can blur handwriting
    use_adaptive_threshold: bool = True
    min_ocr_confidence: int = 25  # Lower threshold to capture more text

    # OCR settings
    ocr_engine: str = "both"  # Use both engines (falls back to EasyOCR if Tesseract unavailable)
    tesseract_config: str = "--oem 3 --psm 6"
    easyocr_languages: List[str] = field(default_factory=lambda: ["en"])

    # PII Detection settings
    detect_names: bool = True
    detect_dates: bool = True
    detect_phone: bool = True
    detect_ids: bool = True
    detect_age: bool = True
    detect_aadhaar: bool = True
    detect_pan: bool = True
    detect_medical_ids: bool = True

    # Redaction settings
    color_coded_redaction: bool = True
    redaction_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'NAME': (255, 0, 0),        # Red
        'DATE': (0, 128, 255),      # Orange
        'PHONE': (0, 255, 0),       # Green
        'ID_NUMBER': (255, 0, 255), # Magenta
        'AADHAAR': (128, 0, 128),   # Purple
        'PAN': (0, 128, 128),       # Teal
        'AGE': (255, 255, 0),       # Yellow (BGR)
        'GENDER': (255, 128, 0),    # Light Blue
        'LOCATION': (0, 165, 255),  # Orange
        'ORGANIZATION': (128, 128, 128),  # Gray
        'MEDICAL_ID': (0, 0, 128),  # Dark Red
        'DEFAULT': (0, 0, 0)        # Black
    })

# ============================================
# PII ENTITY CLASS
# ============================================

@dataclass
class PIIEntity:
    """Represents a detected PII entity with enhanced metadata"""
    type: str
    value: str
    start: int
    end: int
    confidence: float
    detection_method: str = "regex"  # "regex", "ner", "pattern", "combined"
    category: str = "general"  # "personal", "medical", "financial", "contact"

    def to_dict(self) -> Dict:
        return {
            'type': self.type,
            'value': self.value,
            'start': self.start,
            'end': self.end,
            'confidence': round(self.confidence, 3),
            'detection_method': self.detection_method,
            'category': self.category
        }

# ============================================
# PROCESSING METRICS
# ============================================

@dataclass
class ProcessingMetrics:
    """Track processing time and performance metrics"""
    preprocessing_time: float = 0.0
    ocr_time: float = 0.0
    text_cleaning_time: float = 0.0
    pii_detection_time: float = 0.0
    redaction_time: float = 0.0
    total_time: float = 0.0

    # OCR metrics
    total_characters: int = 0
    total_words: int = 0
    text_regions_found: int = 0
    text_regions_kept: int = 0
    avg_ocr_confidence: float = 0.0

    # PII metrics
    total_pii_found: int = 0
    pii_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timing': {
                'preprocessing_ms': round(self.preprocessing_time * 1000, 2),
                'ocr_ms': round(self.ocr_time * 1000, 2),
                'text_cleaning_ms': round(self.text_cleaning_time * 1000, 2),
                'pii_detection_ms': round(self.pii_detection_time * 1000, 2),
                'redaction_ms': round(self.redaction_time * 1000, 2),
                'total_ms': round(self.total_time * 1000, 2)
            },
            'ocr_stats': {
                'total_characters': self.total_characters,
                'total_words': self.total_words,
                'text_regions': self.text_regions_found,
                'text_regions_kept': self.text_regions_kept,
                'avg_confidence': round(self.avg_ocr_confidence, 2)
            },
            'pii_stats': {
                'total_found': self.total_pii_found,
                'by_type': self.pii_by_type
            }
        }

# ============================================
# IMAGE PREPROCESSOR (ENHANCED v2.1)
# ============================================

class ImagePreprocessor:
    """
    Enhanced image pre-processing optimized for:
    - Slightly tilted images (advanced deskew)
    - Different handwriting styles (adaptive preprocessing)
    - Doctor/clinic-style notes and forms
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes (for Streamlit uploads)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        if width > self.config.resize_width:
            ratio = self.config.resize_width / width
            new_height = int(height * ratio)
            image = cv2.resize(image, (self.config.resize_width, new_height),
                             interpolation=cv2.INTER_AREA)
        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        return cv2.fastNlMeansDenoising(image, None,
                                        self.config.denoise_strength, 7, 21)

    def deskew_hough(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew using Hough Line Transform - good for documents with lines"""
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
                return self._rotate_image(image, median_angle), median_angle
        return image, 0.0

    def deskew_minarea(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew using minimum area rectangle - good for handwritten text"""
        # Threshold to get binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, 0.0

        # Get all points from contours
        all_points = np.vstack(contours)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(all_points)
        angle = rect[-1]

        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        return self._rotate_image(image, angle), angle

    def deskew_projection(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew using projection profile - best for text documents"""
        # Threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_angle = 0
        best_score = 0

        # Try angles from -15 to 15 degrees
        for angle in np.arange(-15, 15, 0.5):
            rotated = self._rotate_image(binary, angle)
            # Calculate horizontal projection
            projection = np.sum(rotated, axis=1)
            # Score is the sum of squared differences (variance-like)
            score = np.sum((projection[1:] - projection[:-1]) ** 2)

            if score > best_score:
                best_score = score
                best_angle = angle

        return self._rotate_image(image, best_angle), best_angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def advanced_deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Conservative deskew - only rotates if image is clearly tilted.
        Preserves straight images without modification.
        Works for both tilted and normal images.
        """
        # Collect angles from multiple methods
        angles = []

        # Method 1: Hough Transform
        try:
            _, angle1 = self.deskew_hough(image.copy())
            if abs(angle1) < 45:
                angles.append(angle1)
        except Exception:
            pass

        # Method 2: MinArea Rectangle
        try:
            _, angle2 = self.deskew_minarea(image.copy())
            if abs(angle2) < 45:
                angles.append(angle2)
        except Exception:
            pass

        if not angles:
            return image

        # Use median angle if methods agree
        if len(angles) >= 2:
            # If methods disagree by more than 3 degrees, image is probably straight
            if abs(angles[0] - angles[1]) > 3:
                return image
            final_angle = np.median(angles)
        else:
            final_angle = angles[0]

        # Only rotate if angle is significant (> 2 degrees) but not extreme
        # This prevents unnecessary rotation of straight images
        if 2 < abs(final_angle) < 15:
            return self._rotate_image(image, final_angle)

        # Image is straight or angle is unreliable - don't rotate
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for different handwriting styles.
        Works better for varying ink intensities and paper backgrounds.
        """
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    def apply_morphological_ops(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations for cleaner text"""
        # Light dilation to connect broken characters
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        # Light erosion to remove noise
        image = cv2.erode(image, kernel, iterations=1)
        return image

    def _evaluate_binary(self, binary: np.ndarray) -> float:
        """
        Simple quality heuristic for a binarized image.
        Higher variance across rows means clearer text separation.
        """
        projection = np.sum(binary, axis=1)
        return float(np.var(projection))

    def binarize_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """
        Choose between Otsu and adaptive thresholding based on a variance heuristic.
        Returns a binary image better suited for OCR.
        """
        # Otsu
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Adaptive (optionally)
        adaptive_bin = self.adaptive_threshold(gray)

        # Pick the better one using a simple projection variance score
        otsu_score = self._evaluate_binary(otsu_bin)
        adaptive_score = self._evaluate_binary(adaptive_bin)
        if self.config.use_adaptive_threshold and adaptive_score > otsu_score:
            return adaptive_bin
        return otsu_bin

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows from image - important for phone camera photos"""
        if len(image.shape) == 3:
            rgb_planes = cv2.split(image)
        else:
            rgb_planes = [image]

        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            result_planes.append(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))

        if len(result_planes) > 1:
            return cv2.merge(result_planes)
        return result_planes[0]

    def enhance_handwriting(self, image: np.ndarray) -> np.ndarray:
        """
        Special enhancement for handwritten text.
        Handles different handwriting styles and ink intensities.
        """
        # Bilateral filter preserves edges while removing noise
        filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # Sharpen to make handwriting clearer
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)

        # Normalize
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    def detect_form_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect form regions/boxes in clinic-style documents.
        Returns list of (x, y, w, h) for each detected region.
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        h, w = image.shape[:2]
        min_area = (h * w) * 0.001  # Minimum 0.1% of image area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Filter out very thin regions
                if cw > 20 and ch > 10:
                    regions.append((x, y, cw, ch))

        return regions

    def preprocess(self, image_path: str = None, image_bytes: bytes = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full pre-processing pipeline optimized for:
        - Slightly tilted images
        - Different handwriting styles
        - Doctor/clinic-style notes and forms
        """
        if image_path:
            original = self.load_image(image_path)
        elif image_bytes:
            original = self.load_image_from_bytes(image_bytes)
        else:
            raise ValueError("Either image_path or image_bytes must be provided")

        # Step 1: Resize
        image = self.resize_image(original.copy())

        # Step 2: Convert to grayscale
        gray = self.convert_to_grayscale(image)

        # Step 3: Remove shadows (important for photos)
        gray = self.remove_shadows(gray)

        # Step 4: Advanced deskew for tilted images
        gray = self.advanced_deskew(gray)

        # Step 5: Enhance contrast
        gray = self.enhance_contrast(gray)

        # Step 6: Enhance handwriting
        gray = self.enhance_handwriting(gray)

        # Step 7: Denoise
        gray = self.denoise(gray)

        # Step 8: Optional morphological operations
        if self.config.apply_morphology:
            gray = self.apply_morphological_ops(gray)

        # Step 9: Binarize for OCR robustness
        processed = self.binarize_for_ocr(gray)

        return original, processed

# ============================================
# OCR ENGINE (ENHANCED)
# ============================================

class OCREngine:
    """Enhanced OCR with multiple engines and confidence tracking"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.easyocr_reader = None

        if config.ocr_engine in ["easyocr", "both"]:
            self.easyocr_reader = easyocr.Reader(
                config.easyocr_languages,
                gpu=False,
                verbose=False
            )

    def extract_with_tesseract(self, image: np.ndarray) -> Dict:
        """Extract text using Tesseract OCR"""
        data = pytesseract.image_to_data(image,
                                         config=self.config.tesseract_config,
                                         output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image, config=self.config.tesseract_config)

        boxes = []
        confidences = []
        min_conf = self.config.min_ocr_confidence
        filtered_tokens = []
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            token = data['text'][i].strip()
            if conf > min_conf and len(token) > 1:
                confidences.append(conf)
                filtered_tokens.append(token)
                boxes.append({
                    'text': token,
                    'confidence': conf,
                    'bbox': (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
                })

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            'engine': 'tesseract',
            'text': '\n'.join(filtered_tokens) if filtered_tokens else text,
            'boxes': boxes,
            'avg_confidence': avg_conf
        }

    def extract_with_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        results = self.easyocr_reader.readtext(image)

        text_parts = []
        boxes = []
        confidences = []

        min_conf = self.config.min_ocr_confidence / 100.0

        for (bbox, text, confidence) in results:
            if confidence < min_conf or len(text.strip()) < 2:
                continue
            text_parts.append(text)
            confidences.append(confidence * 100)
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            boxes.append({
                'text': text,
                'confidence': confidence * 100,
                'bbox': (int(min(x_coords)), int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))),
                'polygon': [[int(p[0]), int(p[1])] for p in bbox]
            })

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            'engine': 'easyocr',
            'text': '\n'.join(text_parts),
            'boxes': boxes,
            'avg_confidence': avg_conf
        }

    def extract_combined(self, image: np.ndarray) -> Dict:
        """Combine results from multiple OCR engines"""
        easyocr_result = self.extract_with_easyocr(image)

        # Try Tesseract if available
        tesseract_result = None
        try:
            tesseract_result = self.extract_with_tesseract(image)
        except Exception:
            pass  # Tesseract not installed

        if tesseract_result:
            # Merge text from both engines (remove duplicates)
            easy_text = easyocr_result['text']
            tess_text = tesseract_result['text']

            # Combine - use EasyOCR as base, add unique words from Tesseract
            combined_conf = (tesseract_result['avg_confidence'] + easyocr_result['avg_confidence']) / 2

            return {
                'engine': 'combined',
                'text': easy_text,  # EasyOCR better for handwriting
                'boxes': easyocr_result['boxes'],
                'avg_confidence': combined_conf,
                'tesseract_text': tess_text,
                'easyocr_text': easy_text
            }
        else:
            # Fallback to EasyOCR only
            return easyocr_result

    def extract(self, image: np.ndarray) -> Dict:
        """Extract text using configured engine(s)"""
        if self.config.ocr_engine == "tesseract":
            return self.extract_with_tesseract(image)
        elif self.config.ocr_engine == "easyocr":
            return self.extract_with_easyocr(image)
        elif self.config.ocr_engine == "both":
            return self.extract_combined(image)
        else:
            return self.extract_with_easyocr(image)

# ============================================
# TEXT CLEANER (ENHANCED)
# ============================================

class TextCleaner:
    """Enhanced text post-processing and cleaning"""

    def __init__(self):
        # Medical document specific corrections
        self.medical_corrections = {
            'Paticnt': 'Patient',
            'Namc': 'Name',
            'Agc': 'Age',
            'Scx': 'Sex',
            'Datc': 'Date',
            'Signaturc': 'Signature',
            'Doctcr': 'Doctor',
            'Bcd': 'Bed',
            'Dcpt': 'Dept',
            'Timc': 'Time',
        }

    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace while preserving structure"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def remove_noise_characters(self, text: str) -> str:
        """Remove common noise characters from OCR"""
        text = re.sub(r'(?<![\w])[\_\-\~\^\*\#\@]+(?![\w])', '', text)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 1]
        return '\n'.join(cleaned_lines)

    def fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR misrecognitions"""
        for wrong, correct in self.medical_corrections.items():
            text = re.sub(rf'\b{wrong}\b', correct, text, flags=re.IGNORECASE)
        return text

    def normalize_dates(self, text: str) -> str:
        """Normalize date formats"""
        text = re.sub(r'(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{2,4})',
                     r'\1/\2/\3', text)
        return text

    def clean(self, text: str) -> str:
        """Apply full text cleaning pipeline"""
        text = self.remove_extra_whitespace(text)
        text = self.remove_noise_characters(text)
        text = self.fix_common_ocr_errors(text)
        text = self.normalize_dates(text)
        return text

# ============================================
# PII DETECTOR (ENHANCED)
# ============================================

class PIIDetector:
    """Enhanced PII detection with Indian patterns and medical IDs"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None

        # Enhanced regex patterns (all available)
        all_patterns = {
            'PHONE': {
                'patterns': [
                    r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b',  # Indian mobile
                    r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{4}\b',  # General format
                    r'\b(?:Mobile|Phone|Tel|Contact)[:\s]*([+\d\-\s]{10,})\b'
                ],
                'category': 'contact',
                'confidence': 0.95
            },
            'DATE': {
                'patterns': [
                    r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',
                    r'\b\d{1,2}[\s\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]?\d{2,4}\b',
                ],
                'category': 'general',
                'confidence': 0.90
            },
            'AADHAAR': {
                'patterns': [
                    r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # 12-digit Aadhaar
                    r'\b(?:Aadhaar|Aadhar|UID)[:\s]*(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b'
                ],
                'category': 'financial',
                'confidence': 0.95
            },
            'PAN': {
                'patterns': [
                    r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN format: ABCDE1234F
                    r'\b(?:PAN)[:\s]*([A-Z]{5}\d{4}[A-Z])\b'
                ],
                'category': 'financial',
                'confidence': 0.95
            },
            'MEDICAL_ID': {
                'patterns': [
                    r'\b(?:UHID|IPD|OPD|MRN|Reg)[\s\.:\-No]*[:\s]*([A-Z0-9\-]{4,})\b',
                    r'\b(?:Bed)[\s\.:\-No]*[:\s]*(\d+)\b',
                    r'\b(?:MCI|NMC)[\s\-]?\d+\b',  # Medical Council ID
                ],
                'category': 'medical',
                'confidence': 0.90
            },
            'AGE': {
                'patterns': [
                    r'\b(?:Age)[:\s]*(\d{1,3})[\s]?(?:Y|yr|years?|yrs?)?\b',
                    r'\b(\d{1,3})[\s]?(?:Y|yr|years|yrs)\b',
                ],
                'category': 'personal',
                'confidence': 0.85
            },
            'GENDER': {
                'patterns': [
                    r'\b(?:Sex|Gender)[:\s]*([MF]|Male|Female)\b',
                ],
                'category': 'personal',
                'confidence': 0.90
            },
            'EMAIL': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                'category': 'contact',
                'confidence': 0.95
            },
            'BLOOD_GROUP': {
                'patterns': [
                    r'\b(?:Blood[\s]?(?:Group|Type)?)[:\s]*((?:A|B|AB|O)[+-]?)\b',
                    r'\b((?:A|B|AB|O)[+-])\s*(?:ve|positive|negative)?\b'
                ],
                'category': 'medical',
                'confidence': 0.85
            }
        }

        # Respect config toggles
        self.patterns = {}
        if config.detect_phone:
            self.patterns['PHONE'] = all_patterns['PHONE']
        if config.detect_dates:
            self.patterns['DATE'] = all_patterns['DATE']
        if config.detect_aadhaar:
            self.patterns['AADHAAR'] = all_patterns['AADHAAR']
        if config.detect_pan:
            self.patterns['PAN'] = all_patterns['PAN']
        if config.detect_medical_ids:
            self.patterns['MEDICAL_ID'] = all_patterns['MEDICAL_ID']
        if config.detect_age:
            self.patterns['AGE'] = all_patterns['AGE']
        if config.detect_names:
            self.patterns['GENDER'] = all_patterns['GENDER']
            self.patterns['EMAIL'] = all_patterns['EMAIL']
            self.patterns['BLOOD_GROUP'] = all_patterns['BLOOD_GROUP']

        # Name patterns for medical documents - more specific
        self.name_patterns = [
            (r'(?:Patient\s*Name|Pat\.?\s*Name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'personal'),
            (r'(?:Dr\.?|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[a-z]*)?)', 'medical'),
            (r'(?:Consultant|Resident|Physician)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', 'medical'),
            (r'(?:S/O|D/O|W/O|C/O)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', 'personal'),
        ]

    def _is_noise(self, value: str) -> bool:
        """Heuristic to drop OCR noise tokens - aggressive filtering"""
        v = value.strip()

        # Too short
        if len(v) < 3:
            return True

        # All same character
        if len(set(v.replace('\n', '').replace(' ', ''))) <= 2:
            return True

        # Mostly special characters
        alpha_count = sum(1 for c in v if c.isalnum())
        if alpha_count < len(v) * 0.6:
            return True

        # Contains newlines (likely multiple fragments incorrectly joined)
        if '\n' in v:
            return True

        # Common OCR noise patterns
        noise_words = [
            '&', 'Mle', 'Jle', 'Bl', 'ug', 'cr', 'Ln', 'LU', 'Til', 'Det',
            'Doao', 'Rote', 'Dora', 'Nama', 'Tlene', 'Time', 'Gu', 'Vt',
            'Olwv', 'Balt', 'Heari', 'wt', 'TAL', 'Rearon', 'LFC', 'Mitiia',
            'MIAMIN', 'NICCG', 'tsl', 'Nare', 'Nagat', 'Nagar'
        ]
        if v in noise_words:
            return True

        # Looks like form field labels (not actual PII)
        form_words = [
            'Date', 'Time', 'Name', 'Age', 'Sex', 'Bed', 'Route', 'Dose',
            'Notes', 'Signature', 'Doctor', 'Patient', 'TREATMENT', 'ADVICE',
            'PROGRESS', 'REPORT', 'INSTITUTE', 'MEDICAL', 'HOSPITAL', 'SCIENCES'
        ]
        if v in form_words:
            return True

        return False

    def _passes_format_checks(self, pii_type: str, value: str) -> bool:
        """Additional validation for specific PII formats"""
        if pii_type == 'AADHAAR':
            digits = re.sub(r'\D', '', value)
            return len(digits) == 12
        if pii_type == 'PAN':
            return bool(re.match(r'^[A-Z]{5}\d{4}[A-Z]$', value.upper()))
        if pii_type == 'DATE':
            # Reject obviously invalid years
            m = re.search(r'(19|20)\d{2}', value)
            return m is None or (1900 <= int(m.group(0)) <= 2099)
        if pii_type == 'PHONE':
            digits = re.sub(r'\D', '', value)
            return 8 < len(digits) <= 12
        return True

    def detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using enhanced regex patterns"""
        entities = []

        for pii_type, config in self.patterns.items():
            for pattern in config['patterns']:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(1) if match.lastindex else match.group(0)
                    if self._is_noise(value):
                        continue
                    value = value.strip()
                    if not self._passes_format_checks(pii_type, value):
                        continue
                    entities.append(PIIEntity(
                        type=pii_type,
                        value=value.strip(),
                        start=match.start(),
                        end=match.end(),
                        confidence=config['confidence'],
                        detection_method='regex',
                        category=config['category']
                    ))

        # Detect names
        if self.config.detect_names:
            for pattern, category in self.name_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(1) if match.lastindex else match.group(0)
                    # Filter out very short or noise matches
                    if len(value.strip()) > 2 and not value.strip().isdigit() and not self._is_noise(value):
                        entities.append(PIIEntity(
                            type='NAME',
                            value=value.strip(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.85,
                            detection_method='regex',
                            category=category
                        ))

        return entities

    def detect_with_ner(self, text: str) -> List[PIIEntity]:
        """Detect PII using spaCy NER"""
        if self.nlp is None:
            return []

        entities = []
        doc = self.nlp(text)

        type_mapping = {
            'PERSON': ('NAME', 'personal'),
            'DATE': ('DATE', 'general'),
            'GPE': ('LOCATION', 'general'),
            'LOC': ('LOCATION', 'general'),
            'ORG': ('ORGANIZATION', 'general'),
        }

        for ent in doc.ents:
            if ent.label_ in type_mapping:
                pii_type, category = type_mapping[ent.label_]
                if pii_type == 'NAME' and not self.config.detect_names:
                    continue
                if pii_type == 'DATE' and not self.config.detect_dates:
                    continue
                if pii_type == 'LOCATION' and not self.config.detect_names:
                    continue
                if self._is_noise(ent.text):
                    continue
                entities.append(PIIEntity(
                    type=pii_type,
                    value=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.75,
                    detection_method='ner',
                    category=category
                ))

        return entities

    def merge_and_deduplicate(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge overlapping entities and remove duplicates"""
        if not entities:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(entities, key=lambda x: (x.start, -x.confidence))
        merged = [sorted_entities[0]]

        for entity in sorted_entities[1:]:
            last = merged[-1]
            if entity.start < last.end:
                # Overlapping - keep higher confidence
                if entity.confidence > last.confidence:
                    merged[-1] = entity
            else:
                merged.append(entity)

        # Remove value duplicates
        seen = set()
        unique = []
        for entity in merged:
            key = (entity.type, entity.value.lower().strip())
            if key not in seen and len(entity.value.strip()) > 1 and not self._is_noise(entity.value):
                seen.add(key)
                unique.append(entity)

        return unique

    def detect(self, text: str) -> List[PIIEntity]:
        """Detect all PII in text"""
        regex_entities = self.detect_with_regex(text)
        ner_entities = self.detect_with_ner(text)

        all_entities = regex_entities + ner_entities
        return self.merge_and_deduplicate(all_entities)

# ============================================
# STRUCTURED DATA EXTRACTOR
# ============================================

class StructuredDataExtractor:
    """
    Extract structured data from medical documents.
    Optimized for doctor/clinic-style notes and hospital forms.
    """

    def __init__(self):
        # Comprehensive patterns for medical document fields
        self.field_patterns = {
            'patient_name': [
                r'(?:Patient\s*Name|Pat\.?\s*Name)[:\s]*([A-Za-z\s\.]+?)(?=\s+(?:Age|Sex|Gen|C/O|D/O|S/O|Address|Hosp)|$|\n)',
                # Removed generic Name pattern to avoid matching addresses like "Kalinga Nagar"
            ],
            'age': [
                r'(?:Age|Yr|Yrs)[:\s]*([1-9]\d{0,2})\s*(?:Y|yrs|years)?',
                r'\b([1-9]\d{0,2})\s*(?:Years?|Yrs?)\b',
            ],
            'sex': [
                r'(?:Sex|Gender)[:\s]*([MF]|Male|Female)',
                r'\b(Male|Female)\b',
            ],
            'uhid': [
                r'(?:UHID|Reg\.?\s*No|Record\s*No)[:\s\.]*([A-Z0-9\-/]+)',
            ],
            'ipd_no': [
                r'(?:IPD|I\.P\.D)[\s\.:\-]*(?:No\.?)?[\s\.:\-]*([A-Z0-9\-/]+)',
            ],
            'opd_no': [
                r'(?:OPD|O\.P\.D)[\s\.:\-]*(?:No\.?)?[\s\.:\-]*([A-Z0-9\-/]+)',
            ],
            'mrn': [
                r'(?:MRN|Medical\s*Record)[\s\.:\-]*(?:No\.?)?[\s\.:\-]*([A-Z0-9\-/]+)',
            ],
            'bed_no': [
                r'(?:Bed|Ward)[\s\.:\-]*(?:No\.?)?[\s\.:\-]*([A-Z0-9]+)',
            ],
            'department': [
                r'(?:Dept|Department)[:\s]*([A-Za-z\s]+)',
            ],
            'doctor_name': [
                r'(?:Dr\.?|Doctor|Cons\.)[\s]*([A-Za-z\s]+)(?=\n|$)',
            ],
            'date': [
                r'(?:Date|Dt)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
                r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            ],
            'time': [
                r'(?:Time)[:\s]*(\d{1,2}[:\.]\d{2}\s*(?:AM|PM)?)',
            ],
            'diagnosis': [
                r'(?:Diagnosis|Dx)[:\s]*([A-Za-z\s,\-]+)(?=\n|$)',
            ],
            'prescription': [
                r'(?:Rx|Adv|Treatment)[:\s]([\s\S]+?)(?=\n\n|$)',
            ],
            'blood_group': [
                r'(?:Blood\s*Group|BG)[:\s]*([A|B|AB|O][+\-])',
            ],
            'weight': [
                r'(?:Weight|Wt)[:\s]*(\d+(?:\.\d+)?\s*(?:kg)?)',
            ],
            'bp': [
                r'(?:BP|B\.P)[:\s]*(\d{2,3}/\d{2,3})',
            ],
            'pulse': [
                r'(?:Pulse|Rate)[:\s]*(\d{2,3})',
            ],
            'hospital_name': [
                r'((?:Hospital|Hosp|Clinic|Med)[A-Za-z\s]+)',
            ]
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract structured fields from medical documents.
        Handles doctor/clinic-style notes and hospital forms.
        """
        structured_data = {
            'patient': {},
            'visit': {},
            'medical_staff': {},
            'clinical': {},
            'vitals': {},
            'hospital': {}
        }

        # Patient info
        patient_fields = ['patient_name', 'age', 'sex', 'uhid', 'ipd_no', 'opd_no', 'mrn', 'bed_no', 'blood_group']
        for field in patient_fields:
            value = self._extract_field(text, field)
            if value:
                clean_field = field.replace('patient_', '').replace('_no', '').replace('_', ' ')
                structured_data['patient'][clean_field] = value

        # Visit info
        visit_fields = ['date', 'time', 'department']
        for field in visit_fields:
            value = self._extract_field(text, field)
            if value:
                structured_data['visit'][field] = value

        # Medical staff
        value = self._extract_field(text, 'doctor_name')
        if value:
            structured_data['medical_staff']['doctor'] = value

        # Clinical info
        clinical_fields = ['diagnosis', 'prescription']
        for field in clinical_fields:
            value = self._extract_field(text, field)
            if value:
                structured_data['clinical'][field] = value

        # Vitals
        vital_fields = ['bp', 'pulse', 'weight']
        for field in vital_fields:
            value = self._extract_field(text, field)
            if value:
                structured_data['vitals'][field] = value

        # Hospital info
        value = self._extract_field(text, 'hospital_name')
        if value:
            structured_data['hospital']['name'] = value

        # Remove empty sections
        structured_data = {k: v for k, v in structured_data.items() if v}

        return structured_data

    def _extract_field(self, text: str, field: str) -> Optional[str]:
        """Extract a single field using patterns"""
        if field not in self.field_patterns:
            return None

        for pattern in self.field_patterns[field]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = self._normalize_value(value)
                if len(value) > 0:
                    return value
        return None

    def _normalize_value(self, value: str) -> str:
        """Collapse whitespace/newlines and strip punctuation noise"""
        value = re.sub(r'\s+', ' ', value)
        value = value.strip(" :;-")
        return value.strip()

# ============================================
# IMAGE REDACTOR (ENHANCED)
# ============================================

class ImageRedactor:
    """Enhanced redaction with color coding by PII type"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_redaction_color(self, pii_type: str) -> Tuple[int, int, int]:
        """Get color for PII type"""
        if self.config.color_coded_redaction:
            return self.config.redaction_colors.get(
                pii_type,
                self.config.redaction_colors['DEFAULT']
            )
        return self.config.redaction_colors['DEFAULT']

    def find_text_bbox(self, ocr_boxes: List[Dict], pii_value: str) -> List[Tuple]:
        """Find all bounding boxes for PII text"""
        pii_lower = pii_value.lower().strip()
        found_boxes = []

        # Direct match
        for box in ocr_boxes:
            box_text = box['text'].lower().strip()
            if pii_lower in box_text or box_text in pii_lower:
                found_boxes.append(box['bbox'])

        # Word-level matching with fuzzy distance
        if not found_boxes:
            pii_words = [w for w in pii_lower.split() if len(w) > 2]
            for word in pii_words:
                for box in ocr_boxes:
                    candidate = box['text'].lower()
                    if word in candidate:
                        found_boxes.append(box['bbox'])
                        continue
                    # Fuzzy: allow small edit distance for noisy OCR
                    if self._is_fuzzy_match(word, candidate):
                        found_boxes.append(box['bbox'])

        return found_boxes

    def _is_fuzzy_match(self, a: str, b: str, max_ratio: float = 0.3) -> bool:
        """Lightweight normalized edit distance for noisy OCR matching"""
        if not a or not b:
            return False
        dist = self._levenshtein(a, b)
        ratio = dist / max(len(a), len(b), 1)
        return ratio <= max_ratio

    def _levenshtein(self, a: str, b: str) -> int:
        """Compute Levenshtein distance (small strings only)"""
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                ins = previous[j] + 1
                dele = current[j - 1] + 1
                subst = previous[j - 1] + (ca != cb)
                current.append(min(ins, dele, subst))
            previous = current
        return previous[-1]

    def redact_image(self, image: np.ndarray,
                     ocr_result: Dict,
                     pii_entities: List[PIIEntity]) -> np.ndarray:
        """Redact PII from image with color coding"""
        # Convert to PIL for drawing
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_image)
        ocr_boxes = ocr_result.get('boxes', [])

        for entity in pii_entities:
            color = self.get_redaction_color(entity.type)
            # Convert BGR to RGB for PIL
            color_rgb = (color[2], color[1], color[0])

            bboxes = self.find_text_bbox(ocr_boxes, entity.value)
            for bbox in bboxes:
                x, y, w, h = bbox
                padding = 5
                draw.rectangle(
                    [x - padding, y - padding, x + w + padding, y + h + padding],
                    fill=color_rgb
                )

        # Convert back to OpenCV format
        redacted = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return redacted

    def create_legend(self, pii_types: List[str]) -> np.ndarray:
        """Create a color legend image"""
        height = 30 * len(pii_types) + 20
        width = 200
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255

        y_offset = 10
        for pii_type in pii_types:
            color = self.get_redaction_color(pii_type)
            # Draw color box
            cv2.rectangle(legend, (10, y_offset), (30, y_offset + 20), color, -1)
            cv2.rectangle(legend, (10, y_offset), (30, y_offset + 20), (0, 0, 0), 1)
            # Draw text
            cv2.putText(legend, pii_type, (40, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 30

        return legend

# ============================================
# MAIN PIPELINE (ENHANCED)
# ============================================

class EnhancedOCRPipeline:
    """Enhanced pipeline with metrics and structured extraction"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.text_cleaner = TextCleaner()
        self.pii_detector = PIIDetector(self.config)
        self.structured_extractor = StructuredDataExtractor()
        self.redactor = ImageRedactor(self.config)

    def process(self, image_path: str = None,
                image_bytes: bytes = None,
                generate_redacted: bool = True) -> Dict:
        """Process an image through the enhanced pipeline"""
        metrics = ProcessingMetrics()
        total_start = time.time()

        results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }

        if image_path:
            results['input_file'] = image_path

        # Stage 1: Pre-processing
        start = time.time()
        original, processed = self.preprocessor.preprocess(
            image_path=image_path,
            image_bytes=image_bytes
        )
        metrics.preprocessing_time = time.time() - start
        results['stages']['preprocessing'] = {'status': 'completed'}

        # Stage 2: OCR
        start = time.time()
        ocr_result = self.ocr_engine.extract(processed)
        metrics.ocr_time = time.time() - start
        metrics.text_regions_found = len(ocr_result.get('boxes', []))
        metrics.text_regions_kept = len(ocr_result.get('boxes', []))
        metrics.avg_ocr_confidence = ocr_result.get('avg_confidence', 0)
        results['stages']['ocr'] = {'status': 'completed'}
        results['quality_flags'] = []
        if metrics.avg_ocr_confidence < 40:
            results['quality_flags'].append("Low OCR confidence; consider adjusting preprocessing.")
        if metrics.text_regions_found < 5:
            results['quality_flags'].append("Very few text regions detected; image may be too noisy or blank.")

        # Stage 3: Text Cleaning
        start = time.time()
        raw_text = ocr_result['text']
        cleaned_text = self.text_cleaner.clean(raw_text)
        metrics.text_cleaning_time = time.time() - start
        metrics.total_characters = len(cleaned_text)
        metrics.total_words = len(cleaned_text.split())
        results['raw_text'] = raw_text
        results['cleaned_text'] = cleaned_text
        results['stages']['text_cleaning'] = {'status': 'completed'}

        # Stage 4: PII Detection
        start = time.time()
        pii_entities = self.pii_detector.detect(cleaned_text)
        metrics.pii_detection_time = time.time() - start
        metrics.total_pii_found = len(pii_entities)

        # Count by type
        for entity in pii_entities:
            metrics.pii_by_type[entity.type] = metrics.pii_by_type.get(entity.type, 0) + 1

        results['pii_entities'] = [e.to_dict() for e in pii_entities]
        results['stages']['pii_detection'] = {'status': 'completed'}

        # Group PII by type
        pii_by_type = {}
        for entity in pii_entities:
            if entity.type not in pii_by_type:
                pii_by_type[entity.type] = []
            pii_by_type[entity.type].append({
                'value': entity.value,
                'confidence': entity.confidence,
                'category': entity.category
            })
        results['pii_summary'] = pii_by_type

        # Stage 5: Structured Data Extraction
        structured_data = self.structured_extractor.extract(cleaned_text)
        results['structured_data'] = structured_data

        # Stage 6: Image Redaction
        if generate_redacted:
            start = time.time()
            redacted_image = self.redactor.redact_image(
                processed, ocr_result, pii_entities
            )
            metrics.redaction_time = time.time() - start
            results['redacted_image'] = redacted_image
            results['stages']['redaction'] = {'status': 'completed'}

            # Create legend if color-coded
            if self.config.color_coded_redaction and pii_by_type:
                results['legend'] = self.redactor.create_legend(list(pii_by_type.keys()))

        # Store images
        results['original_image'] = original
        results['processed_image'] = processed
        results['ocr_boxes'] = ocr_result.get('boxes', [])

        # Finalize metrics
        metrics.total_time = time.time() - total_start
        results['metrics'] = metrics.to_dict()

        return results


# ============================================
# UTILITY FUNCTIONS
# ============================================

def format_pii_output(result: Dict) -> Dict:
    """
    Format PII output in the expected structure:
    {
        "document_type": "...",
        "pii_extracted": {
            "patient_info": {...},
            "organization": {...},
            "healthcare_providers": [...],
            ...
        }
    }
    """
    # Determine document type from structured data
    structured = result.get('structured_data', {})
    cleaned_text = result.get('cleaned_text', '').upper()

    if 'MEDICATION' in cleaned_text or 'DRUG' in cleaned_text or 'DOSE' in cleaned_text:
        doc_type = "Medication Administration Record"
    elif 'PROGRESS' in cleaned_text or 'REPORT' in cleaned_text:
        doc_type = "Hospital Progress Report"
    elif 'PRESCRIPTION' in cleaned_text:
        doc_type = "Medical Prescription"
    else:
        doc_type = "Medical Document"

    # Build pii_extracted structure
    pii_extracted = {}

    # Patient info
    patient_info = {}
    if 'patient' in structured:
        patient_data = structured['patient']
        if 'name' in patient_data:
            patient_info['name'] = patient_data['name']
        if 'uhid' in patient_data:
            patient_info['uhid_no'] = patient_data['uhid']
        if 'ipd' in patient_data:
            patient_info['ipd_no'] = patient_data['ipd']
        if 'bed' in patient_data:
            patient_info['bed_no'] = patient_data['bed']
        if 'age' in patient_data:
            patient_info['age'] = patient_data['age']
        if 'sex' in patient_data:
            patient_info['sex'] = patient_data['sex']

    # Add from PII entities
    for entity in result.get('pii_entities', []):
        etype = entity['type']
        value = entity['value']
        if etype == 'NAME' and entity.get('category') == 'personal':
            if 'name' not in patient_info:
                patient_info['name'] = value
        elif etype == 'AGE':
            patient_info['age'] = value
        elif etype == 'GENDER':
            patient_info['sex'] = value
        elif etype == 'MEDICAL_ID':
            if 'uhid' in value.lower() or len(value) > 5:
                patient_info['uhid_no'] = value

    if patient_info:
        pii_extracted['patient_info'] = patient_info

    # Organization info
    organization = {}
    if 'hospital' in structured:
        hosp = structured['hospital']
        if 'name' in hosp:
            organization['name'] = hosp['name']
        if 'address' in hosp:
            organization['address'] = hosp['address']

    # Add from entities
    for entity in result.get('pii_entities', []):
        if entity['type'] == 'ORGANIZATION':
            if 'name' not in organization:
                organization['name'] = entity['value']
        elif entity['type'] == 'LOCATION':
            if 'address' not in organization:
                organization['address'] = entity['value']

    if organization:
        pii_extracted['organization'] = organization

    # Healthcare providers
    providers = []
    if 'medical_staff' in structured:
        staff = structured['medical_staff']
        if 'doctor' in staff:
            providers.append({'name': f"Dr. {staff['doctor']}"})

    for entity in result.get('pii_entities', []):
        if entity['type'] == 'NAME' and entity.get('category') == 'medical':
            providers.append({'name': entity['value']})

    if providers:
        pii_extracted['healthcare_providers'] = providers

    # Dates
    dates = []
    for entity in result.get('pii_entities', []):
        if entity['type'] == 'DATE':
            dates.append(entity['value'])
    if dates:
        pii_extracted['dates'] = list(set(dates))

    # Diagnosis
    if 'clinical' in structured and 'diagnosis' in structured['clinical']:
        pii_extracted['diagnosis'] = structured['clinical']['diagnosis']

    # Medications (if applicable)
    if 'clinical' in structured and 'prescription' in structured['clinical']:
        pii_extracted['medications'] = structured['clinical']['prescription']

    return {
        'document_type': doc_type,
        'pii_extracted': pii_extracted
    }


def save_enhanced_results(result: Dict, output_dir: str, base_name: str = None):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)

    if not base_name:
        base_name = os.path.splitext(os.path.basename(result.get('input_file', 'output')))[0]

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

    # Save legend
    if 'legend' in result:
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_legend.jpg"),
            result['legend']
        )

    # Format output in expected structure
    formatted_output = format_pii_output(result)

    # Save comprehensive JSON report
    report = {
        'input_file': result.get('input_file', 'uploaded'),
        'timestamp': result['timestamp'],
        'document_type': formatted_output['document_type'],
        'pii_extracted': formatted_output['pii_extracted'],
        'cleaned_text': result['cleaned_text']
    }

    with open(os.path.join(output_dir, f"{base_name}_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    return os.path.join(output_dir, base_name)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Enhanced OCR Pipeline v2.0")
    print("=" * 60)

    # Configuration
    SAMPLES_DIR = "samples"
    OUTPUT_DIR = "results_enhanced"

    sample_images = [
        os.path.join(SAMPLES_DIR, "sample1.jpg"),
        os.path.join(SAMPLES_DIR, "sample2.jpg"),
        os.path.join(SAMPLES_DIR, "sample3.jpg"),
    ]

    # Initialize pipeline
    print("\nInitializing enhanced pipeline...")
    config = PipelineConfig(color_coded_redaction=True)
    pipeline = EnhancedOCRPipeline(config)
    print("Pipeline ready!")

    # Process each sample
    for i, img_path in enumerate(sample_images, 1):
        if os.path.exists(img_path):
            print(f"\n{'#' * 60}")
            print(f"PROCESSING SAMPLE {i}: {img_path}")
            print(f"{'#' * 60}")

            result = pipeline.process(image_path=img_path, generate_redacted=True)

            # Print metrics
            print(f"\n--- Processing Metrics ---")
            metrics = result['metrics']
            print(f"  Total Time: {metrics['timing']['total_ms']:.0f}ms")
            print(f"  OCR Confidence: {metrics['ocr_stats']['avg_confidence']:.1f}%")
            print(f"  PII Found: {metrics['pii_stats']['total_found']}")

            # Print structured data
            print(f"\n--- Structured Data ---")
            for section, data in result['structured_data'].items():
                print(f"  {section}: {data}")

            # Print PII summary
            print(f"\n--- PII Summary ---")
            for pii_type, items in result['pii_summary'].items():
                values = [item['value'] for item in items[:3]]
                print(f"  {pii_type}: {values}{'...' if len(items) > 3 else ''}")

            # Save results
            save_enhanced_results(result, OUTPUT_DIR, f"sample{i}")
            print(f"\nResults saved to: {OUTPUT_DIR}/sample{i}_*")

    print(f"\n{'=' * 60}")
    print("All processing complete!")
    print(f"{'=' * 60}")
