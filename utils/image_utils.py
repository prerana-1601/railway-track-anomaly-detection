"""
Image processing utilities for Railway Track Anomaly Detection System
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate position to center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image for better detection"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def draw_bounding_boxes(image: np.ndarray, detections: List[dict], 
                       color_map: Optional[dict] = None) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    
    if color_map is None:
        color_map = {
            'missing_clamp': (0, 0, 255),    # Red
            'ridge': (0, 165, 255),          # Orange
            'crack': (0, 0, 255),            # Red
            'debris': (0, 255, 255),         # Yellow
            'missing_bolt': (0, 0, 255)      # Red
        }
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            
            # Get color for class
            color = color_map.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(annotated_image,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label,
                       (x1 + 5, y1 - 5),
                       font, font_scale, (255, 255, 255), thickness)
    
    return annotated_image


def extract_roi(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Extract region of interest from image"""
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    return image


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_noise_reduction(image: np.ndarray) -> np.ndarray:
    """Apply noise reduction to image"""
    # Apply bilateral filter for noise reduction while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def create_image_mosaic(images: List[np.ndarray], cols: int = 3) -> np.ndarray:
    """Create a mosaic of images"""
    if not images:
        return np.array([])
    
    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Get image dimensions
    h, w = images[0].shape[:2]
    
    # Create mosaic canvas
    mosaic = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        # Ensure image is the right size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        # Place image in mosaic
        y_start = row * h
        y_end = (row + 1) * h
        x_start = col * w
        x_end = (col + 1) * w
        
        mosaic[y_start:y_end, x_start:x_end] = img
    
    return mosaic


def save_image_with_metadata(image: np.ndarray, filepath: str, 
                           metadata: Optional[dict] = None):
    """Save image with metadata"""
    # Save the image
    cv2.imwrite(filepath, image)
    
    # Save metadata as JSON file with same name but .json extension
    if metadata:
        import json
        metadata_file = filepath.rsplit('.', 1)[0] + '.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_image_with_metadata(filepath: str) -> Tuple[np.ndarray, Optional[dict]]:
    """Load image with metadata"""
    # Load the image
    image = cv2.imread(filepath)
    
    # Load metadata if available
    metadata = None
    metadata_file = filepath.rsplit('.', 1)[0] + '.json'
    try:
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        pass
    
    return image, metadata 