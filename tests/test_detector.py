"""
Unit tests for Railway Track Anomaly Detection System
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from server.anomaly_detector import AnomalyDetector
from utils.image_utils import (
    draw_bounding_boxes, calculate_iou, enhance_image,
    resize_image, create_image_mosaic
)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = AnomalyDetector()
        
        # Create a synthetic test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some railway tracks
        cv2.line(self.test_image, (100, 240), (540, 240), (128, 128, 128), 8)
        cv2.line(self.test_image, (100, 248), (540, 248), (128, 128, 128), 8)
        
        # Add some test anomalies
        cv2.rectangle(self.test_image, (300, 230), (320, 250), (0, 0, 255), -1)  # Red rectangle
        cv2.circle(self.test_image, (400, 240), 15, (0, 255, 255), -1)  # Yellow circle
        
    def test_fallback_detection(self):
        """Test fallback detection method"""
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = self.detector._fallback_detection(
            self.test_image, gps, timestamp, drone_id
        )
        
        # Should detect at least one anomaly
        self.assertGreater(len(detections), 0)
        
        # Check detection structure
        for detection in detections:
            self.assertIn('class', detection)
            self.assertIn('confidence', detection)
            self.assertIn('bbox', detection)
            self.assertIn('description', detection)
            self.assertIn('severity', detection)
            self.assertIn('action_required', detection)
            
    def test_detection_bbox_format(self):
        """Test that bounding boxes are in correct format"""
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = self.detector._fallback_detection(
            self.test_image, gps, timestamp, drone_id
        )
        
        for detection in detections:
            bbox = detection['bbox']
            self.assertEqual(len(bbox), 4)  # Should have 4 coordinates
            self.assertIsInstance(bbox[0], (int, float))
            self.assertIsInstance(bbox[1], (int, float))
            self.assertIsInstance(bbox[2], (int, float))
            self.assertIsInstance(bbox[3], (int, float))
            
    def test_detection_confidence_range(self):
        """Test that confidence values are in valid range"""
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = self.detector._fallback_detection(
            self.test_image, gps, timestamp, drone_id
        )
        
        for detection in detections:
            confidence = detection['confidence']
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
    def test_detection_class_types(self):
        """Test that detection classes are valid"""
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = self.detector._fallback_detection(
            self.test_image, gps, timestamp, drone_id
        )
        
        valid_classes = ['missing_clamp', 'debris', 'ridge', 'crack', 'missing_bolt']
        
        for detection in detections:
            class_name = detection['class']
            self.assertIn(class_name, valid_classes)
            
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        initial_stats = self.detector.get_statistics()
        
        # Process a frame
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = self.detector._detect_anomalies(
            self.test_image, gps, timestamp, drone_id
        )
        
        updated_stats = self.detector.get_statistics()
        
        # Check that statistics were updated
        self.assertGreater(updated_stats['frames_processed'], initial_stats['frames_processed'])
        self.assertGreaterEqual(updated_stats['detections_found'], len(detections))


class TestImageUtils(unittest.TestCase):
    """Test cases for image utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
    def test_resize_image(self):
        """Test image resizing"""
        target_size = (200, 150)
        resized = resize_image(self.test_image, target_size)
        
        self.assertEqual(resized.shape[:2], target_size[::-1])  # OpenCV uses (height, width)
        self.assertEqual(resized.shape[2], 3)  # Should maintain 3 channels
        
    def test_enhance_image(self):
        """Test image enhancement"""
        enhanced = enhance_image(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, self.test_image.dtype)
        
    def test_draw_bounding_boxes(self):
        """Test bounding box drawing"""
        detections = [
            {
                'class': 'missing_clamp',
                'confidence': 0.95,
                'bbox': [10, 10, 50, 50]
            },
            {
                'class': 'debris',
                'confidence': 0.75,
                'bbox': [60, 60, 90, 90]
            }
        ]
        
        annotated = draw_bounding_boxes(self.test_image, detections)
        
        self.assertEqual(annotated.shape, self.test_image.shape)
        # The annotated image should be different from original
        self.assertFalse(np.array_equal(annotated, self.test_image))
        
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        
        iou = calculate_iou(box1, box2)
        
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
        
        # Test overlapping boxes
        box3 = [0, 0, 10, 10]
        box4 = [0, 0, 10, 10]
        iou_same = calculate_iou(box3, box4)
        self.assertEqual(iou_same, 1.0)
        
        # Test non-overlapping boxes
        box5 = [0, 0, 5, 5]
        box6 = [10, 10, 15, 15]
        iou_no_overlap = calculate_iou(box5, box6)
        self.assertEqual(iou_no_overlap, 0.0)
        
    def test_create_image_mosaic(self):
        """Test image mosaic creation"""
        images = [self.test_image] * 4  # Create 4 copies
        
        mosaic = create_image_mosaic(images, cols=2)
        
        # Should create a 2x2 grid
        expected_height = self.test_image.shape[0] * 2
        expected_width = self.test_image.shape[1] * 2
        
        self.assertEqual(mosaic.shape[0], expected_height)
        self.assertEqual(mosaic.shape[1], expected_width)
        self.assertEqual(mosaic.shape[2], 3)
        
    def test_image_with_metadata(self):
        """Test saving and loading images with metadata"""
        metadata = {
            'timestamp': '2024-01-15T10:30:00Z',
            'gps': {'lat': 40.7128, 'lng': -74.0060},
            'detections': [{'class': 'missing_clamp', 'confidence': 0.95}]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Save image with metadata
            from utils.image_utils import save_image_with_metadata, load_image_with_metadata
            
            save_image_with_metadata(self.test_image, temp_path, metadata)
            
            # Load image with metadata
            loaded_image, loaded_metadata = load_image_with_metadata(temp_path)
            
            # Check that image was saved and loaded correctly
            self.assertTrue(np.array_equal(loaded_image, self.test_image))
            self.assertEqual(loaded_metadata, metadata)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            metadata_path = temp_path.rsplit('.', 1)[0] + '.json'
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_detection_pipeline(self):
        """Test the complete detection pipeline"""
        # Create a test image with known anomalies
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a red rectangle (missing clamp)
        cv2.rectangle(test_image, (300, 230), (320, 250), (0, 0, 255), -1)
        
        # Create detector
        detector = AnomalyDetector()
        
        # Test detection
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = "2024-01-15T10:30:00Z"
        drone_id = "test_drone"
        
        detections = detector._detect_anomalies(test_image, gps, timestamp, drone_id)
        
        # Should detect the red rectangle as missing_clamp
        self.assertGreater(len(detections), 0)
        
        # Check that at least one detection is a missing_clamp
        detection_classes = [d['class'] for d in detections]
        self.assertIn('missing_clamp', detection_classes)


if __name__ == '__main__':
    unittest.main() 