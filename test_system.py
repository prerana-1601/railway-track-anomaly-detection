#!/usr/bin/env python3
"""
Simple test script to generate data for Railway Track Anomaly Detection System
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

def create_sample_video():
    """Create a sample video file"""
    video_path = Path("data/videos/sample_railway.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    print(f"Creating sample video: {video_path}")
    
    for frame_num in range(300):  # 10 seconds at 30fps
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw railway tracks
        cv2.line(frame, (100, 240), (540, 240), (128, 128, 128), 8)
        cv2.line(frame, (100, 248), (540, 248), (128, 128, 128), 8)
        
        # Draw sleepers
        for i in range(0, 640, 40):
            cv2.line(frame, (i, 220), (i, 260), (64, 64, 64), 2)
        
        # Add anomalies every 50 frames
        if frame_num % 50 == 0:
            # Add missing clamp
            cv2.rectangle(frame, (300, 230), (320, 250), (0, 0, 255), -1)
            cv2.putText(frame, "MISSING CLAMP", (280, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif frame_num % 75 == 0:
            # Add debris
            cv2.circle(frame, (400, 240), 15, (0, 255, 255), -1)
            cv2.putText(frame, "DEBRIS", (380, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {video_path}")
    return str(video_path)

def create_sample_frames():
    """Create sample extracted frames"""
    frames_dir = Path("data/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample frames in: {frames_dir}")
    
    for i in range(10):
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw railway tracks
        cv2.line(frame, (100, 240), (540, 240), (128, 128, 128), 8)
        cv2.line(frame, (100, 248), (540, 248), (128, 128, 128), 8)
        
        # Add some anomalies
        if i % 3 == 0:
            cv2.rectangle(frame, (300, 230), (320, 250), (0, 0, 255), -1)
            cv2.putText(frame, "MISSING CLAMP", (280, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif i % 3 == 1:
            cv2.circle(frame, (400, 240), 15, (0, 255, 255), -1)
            cv2.putText(frame, "DEBRIS", (380, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add timestamp and frame info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save frame
        frame_path = frames_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Create metadata file
        metadata = {
            "frame_number": i,
            "timestamp": timestamp,
            "gps": {
                "lat": 40.7128 + (i * 0.0001),
                "lng": -74.0060 + (i * 0.0001)
            },
            "detections": [
                {
                    "class": "missing_clamp" if i % 3 == 0 else "debris",
                    "confidence": 0.85,
                    "bbox": [300, 230, 320, 250] if i % 3 == 0 else [385, 225, 415, 255],
                    "description": "Missing rail clamp detected" if i % 3 == 0 else "Debris on track detected"
                }
            ]
        }
        
        metadata_path = frames_dir / f"frame_{i:03d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Created {10} sample frames with metadata")

def create_sample_report():
    """Create a sample PDF report"""
    try:
        from server.pdf_generator import PDFGenerator
        
        # Create sample detections
        sample_detections = [
            {
                "class": "missing_clamp",
                "confidence": 0.95,
                "bbox": [100, 150, 200, 250],
                "description": "Missing rail clamp detected",
                "severity": "HIGH",
                "action_required": "Immediate inspection required",
                "timestamp": datetime.now().isoformat(),
                "gps": {"lat": 40.7128, "lng": -74.0060}
            },
            {
                "class": "debris",
                "confidence": 0.75,
                "bbox": [300, 200, 350, 250],
                "description": "Debris on track detected",
                "severity": "LOW",
                "action_required": "Clean track",
                "timestamp": datetime.now().isoformat(),
                "gps": {"lat": 40.7129, "lng": -74.0061}
            }
        ]
        
        # Generate report
        generator = PDFGenerator()
        report_path = generator.generate_simple_report(sample_detections, "sample_report.pdf")
        
        print(f"Sample report created: {report_path}")
        
    except Exception as e:
        print(f"Could not create PDF report: {e}")
        print("Creating JSON report instead...")
        
        # Create JSON report as fallback
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "class": "missing_clamp",
                    "confidence": 0.95,
                    "description": "Missing rail clamp detected",
                    "severity": "HIGH",
                    "gps": {"lat": 40.7128, "lng": -74.0060}
                },
                {
                    "class": "debris", 
                    "confidence": 0.75,
                    "description": "Debris on track detected",
                    "severity": "LOW",
                    "gps": {"lat": 40.7129, "lng": -74.0061}
                }
            ]
        }
        
        report_path = reports_dir / "sample_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"JSON report created: {report_path}")

def main():
    """Generate all sample data"""
    print("Generating sample data for Railway Track Anomaly Detection System...")
    print("=" * 60)
    
    # Create directories if they don't exist
    for dir_name in ["videos", "frames", "reports"]:
        Path(f"data/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    create_sample_video()
    print()
    
    create_sample_frames()
    print()
    
    create_sample_report()
    print()
    
    print("Sample data generation complete!")
    print("\nGenerated files:")
    print("- Sample video: data/videos/sample_railway.mp4")
    print("- Sample frames: data/frames/frame_*.jpg")
    print("- Sample report: data/reports/sample_report.pdf")
    
    print("\nYou can now run the system with real data:")
    print("python client/drone_client.py --video-source data/videos/sample_railway.mp4")

if __name__ == "__main__":
    main() 