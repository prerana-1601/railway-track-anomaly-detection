#!/usr/bin/env python3
"""
Run the complete Railway Track Anomaly Detection System
"""

import time
import threading
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from client.drone_client import DroneClient
from server.anomaly_detector import AnomalyDetector
from server.pdf_generator import PDFGenerator

def run_complete_system():
    """Run the complete system with synthetic data"""
    print("🚀 Starting Railway Track Anomaly Detection System")
    print("=" * 60)
    
    # Create components
    drone = DroneClient(drone_id="railway_drone")
    detector = AnomalyDetector()
    
    # Start anomaly detector in background
    detector_thread = threading.Thread(target=detector.run, daemon=True)
    detector_thread.start()
    
    print("✅ Anomaly detector started")
    time.sleep(2)  # Give detector time to initialize
    
    # Start drone client
    print("✅ Starting drone client with synthetic data...")
    drone.start_video_capture()  # Use synthetic video
    
    try:
        # Run for 30 seconds
        drone.run(duration=30)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping system...")
    finally:
        # Stop components
        drone.stop()
        detector.stop()
        
        # Generate report from collected data
        print("📊 Generating PDF report...")
        if hasattr(detector, 'frame_buffer') and detector.frame_buffer:
            pdf_gen = PDFGenerator()
            frames_with_detections = []
            
            for frame_data in detector.frame_buffer:
                if frame_data.get('detections'):
                    frames_with_detections.append(frame_data)
            
            if frames_with_detections:
                report_path = pdf_gen.generate_report(frames_with_detections)
                print(f"✅ PDF report generated: {report_path}")
            else:
                print("ℹ️  No detections found to report")
        else:
            print("ℹ️  No frame data available for report")
    
    print("🎉 System completed!")

if __name__ == "__main__":
    run_complete_system() 