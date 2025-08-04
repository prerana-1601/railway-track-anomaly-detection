#!/usr/bin/env python3
"""
Demo script for Railway Track Anomaly Detection System

This script demonstrates the complete system by running:
1. MQTT Server
2. Anomaly Detector
3. Drone Client (simulation)
4. PDF Report Generation
"""

import time
import threading
import signal
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from server.mqtt_server import MQTTServer
from server.anomaly_detector import AnomalyDetector
from client.drone_client import DroneClient
from server.pdf_generator import PDFGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RailwayDetectionDemo:
    """Demo class to run the complete railway detection system"""
    
    def __init__(self):
        self.mqtt_server = None
        self.anomaly_detector = None
        self.drone_client = None
        self.pdf_generator = None
        self.is_running = False
        
        # Store frame data for PDF generation
        self.frame_data = []
        
    def start_mqtt_server(self):
        """Start MQTT server in a separate thread"""
        logger.info("Starting MQTT server...")
        self.mqtt_server = MQTTServer()
        
        def run_server():
            self.mqtt_server.run()
            
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        logger.info("MQTT server started")
        
    def start_anomaly_detector(self):
        """Start anomaly detector in a separate thread"""
        logger.info("Starting anomaly detector...")
        self.anomaly_detector = AnomalyDetector()
        
        # Override the frame buffer to collect data for PDF
        original_append = self.anomaly_detector.frame_buffer.append
        
        def custom_append(frame_info):
            original_append(frame_info)
            self.frame_data.append(frame_info)
            
        self.anomaly_detector.frame_buffer.append = custom_append
        
        def run_detector():
            self.anomaly_detector.run()
            
        detector_thread = threading.Thread(target=run_detector, daemon=True)
        detector_thread.start()
        
        # Wait a moment for detector to start
        time.sleep(2)
        logger.info("Anomaly detector started")
        
    def start_drone_client(self, duration: int = 30):
        """Start drone client in a separate thread"""
        logger.info("Starting drone client...")
        self.drone_client = DroneClient(drone_id="demo_drone")
        self.drone_client.start_video_capture()  # Use synthetic video
        
        def run_drone():
            self.drone_client.run(duration=duration)
            
        drone_thread = threading.Thread(target=run_drone, daemon=True)
        drone_thread.start()
        
        logger.info("Drone client started")
        
    def generate_report(self):
        """Generate PDF report from collected data"""
        if not self.frame_data:
            logger.warning("No frame data available for report generation")
            return
            
        logger.info("Generating PDF report...")
        self.pdf_generator = PDFGenerator()
        
        # Filter frames with detections
        frames_with_detections = [frame for frame in self.frame_data if frame.get('detections')]
        
        if frames_with_detections:
            report_path = self.pdf_generator.generate_report(frames_with_detections)
            logger.info(f"PDF report generated: {report_path}")
        else:
            logger.info("No detections found, generating simple report...")
            # Generate a simple report with all detections
            all_detections = []
            for frame in self.frame_data:
                all_detections.extend(frame.get('detections', []))
                
            if all_detections:
                report_path = self.pdf_generator.generate_simple_report(all_detections)
                logger.info(f"Simple PDF report generated: {report_path}")
            else:
                logger.warning("No detections found in any frames")
                
    def run_demo(self, duration: int = 30):
        """Run the complete demo"""
        logger.info("Starting Railway Track Anomaly Detection Demo")
        logger.info(f"Demo will run for {duration} seconds")
        
        try:
            # Start MQTT server
            self.start_mqtt_server()
            
            # Start anomaly detector
            self.start_anomaly_detector()
            
            # Start drone client
            self.start_drone_client(duration)
            
            # Wait for demo to complete
            logger.info("Demo is running... Press Ctrl+C to stop early")
            time.sleep(duration)
            
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
            
        finally:
            # Stop all components
            self.stop_demo()
            
            # Generate final report
            self.generate_report()
            
            logger.info("Demo completed!")
            
    def stop_demo(self):
        """Stop all demo components"""
        logger.info("Stopping demo components...")
        
        if self.drone_client:
            self.drone_client.stop()
            
        if self.anomaly_detector:
            self.anomaly_detector.stop()
            
        if self.mqtt_server:
            self.mqtt_server.stop()
            
    def print_statistics(self):
        """Print demo statistics"""
        if self.anomaly_detector:
            stats = self.anomaly_detector.get_statistics()
            logger.info("Detection Statistics:")
            logger.info(f"  Frames processed: {stats['frames_processed']}")
            logger.info(f"  Detections found: {stats['detections_found']}")
            logger.info(f"  Average processing time: {stats['processing_time_avg']:.2f}ms")
            
        if self.mqtt_server:
            stats = self.mqtt_server.get_statistics()
            logger.info("MQTT Statistics:")
            logger.info(f"  Messages received: {stats['messages_received']}")
            logger.info(f"  Messages forwarded: {stats['messages_forwarded']}")
            logger.info(f"  Connected clients: {stats['clients_connected']}")


def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    logger.info("Received interrupt signal, stopping demo...")
    sys.exit(0)


def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Track Anomaly Detection Demo")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Demo duration in seconds (default: 30)")
    parser.add_argument("--drone-id", default="demo_drone",
                       help="Drone ID for the demo (default: demo_drone)")
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run demo
    demo = RailwayDetectionDemo()
    
    try:
        demo.run_demo(duration=args.duration)
        demo.print_statistics()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 