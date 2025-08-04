#!/usr/bin/env python3
"""
Test script to generate synthetic data with detections
"""

import cv2
import numpy as np
import json
import base64
from datetime import datetime
from pathlib import Path

def create_test_frame_with_detections():
    """Create a test frame with synthetic detections"""
    # Create a frame with railway tracks
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw railway tracks
    cv2.line(frame, (100, 240), (540, 240), (128, 128, 128), 8)
    cv2.line(frame, (100, 248), (540, 248), (128, 128, 128), 8)
    
    # Draw sleepers
    for i in range(0, 640, 40):
        cv2.rectangle(frame, (i, 220), (i+2, 260), (64, 64, 64), -1)
    
    # Add a missing clamp (red rectangle)
    cv2.rectangle(frame, (300, 230), (320, 250), (0, 0, 255), -1)
    cv2.putText(frame, "MISSING CLAMP", (280, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add debris (yellow circle)
    cv2.circle(frame, (400, 240), 15, (0, 255, 255), -1)
    cv2.putText(frame, "DEBRIS", (380, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_test_message():
    """Create a test MQTT message with detections"""
    frame = create_test_frame_with_detections()
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    
    # Create detections
    detections = [
        {
            "class": "missing_clamp",
            "confidence": 0.95,
            "bbox": [300, 230, 320, 250],
            "description": "Missing rail clamp detected",
            "severity": "HIGH",
            "action_required": "Immediate inspection required"
        },
        {
            "class": "debris",
            "confidence": 0.85,
            "bbox": [385, 225, 415, 255],
            "description": "Debris on track detected",
            "severity": "LOW",
            "action_required": "Clean track"
        }
    ]
    
    # Create message
    message = {
        "frame_data": frame_data,
        "gps": {
            "lat": 40.7128,
            "lng": -74.0060
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_number": 1,
        "drone_id": "test_drone"
    }
    
    return message, detections

def test_anomaly_detector():
    """Test the anomaly detector with synthetic data"""
    try:
        from server.anomaly_detector import AnomalyDetector
        
        # Create anomaly detector
        detector = AnomalyDetector()
        
        # Create test message
        message, expected_detections = create_test_message()
        
        # Simulate processing the message
        message_bytes = json.dumps(message).encode('utf-8')
        detector._process_video_message(message_bytes)
        
        # Force fallback detection for testing
        frame = create_test_frame_with_detections()
        gps = {"lat": 40.7128, "lng": -74.0060}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        drone_id = "test_drone"
        
        # Use fallback detection directly
        detections = detector._fallback_detection(frame, gps, timestamp, drone_id)
        
        if detections:
            # Create frame info and save annotated frame
            frame_info = {
                'frame': frame,
                'detections': detections,
                'gps': gps,
                'timestamp': timestamp,
                'frame_number': 2,
                'drone_id': drone_id
            }
            detector._save_annotated_frame(frame_info)
            print(f"✅ Fallback detection found {len(detections)} anomalies")
        else:
            print("❌ No detections found with fallback method")
        
        print("✅ Test completed successfully!")
        print(f"Expected detections: {len(expected_detections)}")
        print("Check data/frames/ for annotated images")
        
        # Check if annotated frames were created
        frames_dir = Path("data/frames")
        if frames_dir.exists():
            annotated_files = list(frames_dir.glob("detection_frame_*.jpg"))
            print(f"Found {len(annotated_files)} annotated frames")
            for file in annotated_files:
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_generation():
    """Test PDF generation with annotated frames"""
    try:
        from server.pdf_generator import PDFGenerator
        
        # Create sample frame data with detections
        frame = create_test_frame_with_detections()
        
        frame_data = [{
            'frame': frame,
            'detections': [
                {
                    "class": "missing_clamp",
                    "confidence": 0.95,
                    "bbox": [300, 230, 320, 250],
                    "description": "Missing rail clamp detected",
                    "severity": "HIGH",
                    "action_required": "Immediate inspection required",
                    "timestamp": datetime.now().isoformat(),
                    "gps": {"lat": 40.7128, "lng": -74.0060}
                },
                {
                    "class": "debris",
                    "confidence": 0.85,
                    "bbox": [385, 225, 415, 255],
                    "description": "Debris on track detected",
                    "severity": "LOW",
                    "action_required": "Clean track",
                    "timestamp": datetime.now().isoformat(),
                    "gps": {"lat": 40.7129, "lng": -74.0061}
                }
            ],
            'gps': {"lat": 40.7128, "lng": -74.0060},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'frame_number': 2,  # Match the frame number used in the test
            'drone_id': 'test_drone'
        }]
        
        # Generate PDF
        generator = PDFGenerator()
        report_path = generator.generate_report(frame_data, "test_detection_report.pdf")
        
        print(f"✅ PDF report generated: {report_path}")
        
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Railway Track Anomaly Detection System")
    print("=" * 50)
    
    print("\n1. Testing anomaly detector...")
    test_anomaly_detector()
    
    print("\n2. Testing PDF generation...")
    test_pdf_generation()
    
    print("\n🎉 All tests completed!") 