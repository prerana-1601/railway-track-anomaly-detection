"""
Anomaly Detector Server for Railway Track Anomaly Detection System

Uses YOLO for real-time detection of railway track anomalies and processes MQTT messages.
"""

import cv2
import json
import time
import base64
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from collections import deque

import paho.mqtt.client as mqtt
import numpy as np
from ultralytics import YOLO

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE,
    MQTT_TOPIC_VIDEO, MQTT_TOPIC_GPS, MQTT_TOPIC_DETECTION,
    YOLO_CONFIG, DETECTION_CLASSES, PERFORMANCE_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """YOLO-based anomaly detector for railway tracks"""
    
    def __init__(self):
        self.model = None
        self.is_running = False
        self.mqtt_client = None
        self.detection_cache = deque(maxlen=PERFORMANCE_CONFIG["cache_size"])
        self.frame_buffer = deque(maxlen=10)
        
        # Initialize YOLO model
        self._load_model()
        
        # Initialize MQTT client
        self._setup_mqtt()
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "detections_found": 0,
            "processing_time_avg": 0.0
        }
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            # For demonstration, we'll use a pre-trained model
            # In production, you would use your custom-trained model
            self.model = YOLO('yolov8n.pt')  # Use nano model for speed
            
            # If you have custom weights, use:
            # self.model = YOLO(YOLO_CONFIG["weights"])
            
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to a dummy model for demonstration
            self.model = None
            
    def _setup_mqtt(self):
        """Setup MQTT client connection"""
        self.mqtt_client = mqtt.Client(client_id="anomaly_detector")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.mqtt_client.subscribe(MQTT_TOPIC_VIDEO, qos=1)
            self.mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("MQTT connection successful")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning("MQTT connection lost")
        
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            if msg.topic == MQTT_TOPIC_VIDEO:
                self._process_video_message(msg.payload)
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def _process_video_message(self, payload: bytes):
        """Process incoming video frame message"""
        try:
            # Parse JSON payload
            data = json.loads(payload.decode('utf-8'))
            
            # Decode base64 image
            frame_data = base64.b64decode(data['frame_data'])
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # Extract GPS and metadata
            gps = data['gps']
            timestamp = data['timestamp']
            frame_number = data['frame_number']
            drone_id = data['drone_id']
            
            # Process frame for anomalies
            detections = self._detect_anomalies(frame, gps, timestamp, drone_id)
            
            # Store frame and detections for PDF generation
            frame_info = {
                'frame': frame,
                'detections': detections,
                'gps': gps,
                'timestamp': timestamp,
                'frame_number': frame_number,
                'drone_id': drone_id
            }
            
            # Save annotated frame if detections found
            if detections:
                self._save_annotated_frame(frame_info)
            
            self.frame_buffer.append(frame_info)
            
            # Publish detection results
            if detections:
                self._publish_detections(detections, gps, timestamp, drone_id)
                
        except Exception as e:
            logger.error(f"Error processing video message: {e}")
            
    def _detect_anomalies(self, frame: np.ndarray, gps: Dict, 
                          timestamp: str, drone_id: str) -> List[Dict]:
        """Detect anomalies in the frame using YOLO"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # Fallback detection for demonstration
                return self._fallback_detection(frame, gps, timestamp, drone_id)
            
            # Run YOLO detection
            results = self.model(frame, conf=YOLO_CONFIG["confidence"], 
                               iou=YOLO_CONFIG["iou_threshold"])
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection details
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Map class ID to class name
                        class_name = YOLO_CONFIG["classes"][class_id] if class_id < len(YOLO_CONFIG["classes"]) else f"class_{class_id}"
                        
                        # Create detection object
                        detection = {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "description": DETECTION_CLASSES.get(class_name, {}).get("description", f"{class_name} detected"),
                            "severity": DETECTION_CLASSES.get(class_name, {}).get("severity", "MEDIUM"),
                            "action_required": DETECTION_CLASSES.get(class_name, {}).get("action_required", "Inspect area"),
                            "timestamp": timestamp,
                            "gps": gps,
                            "drone_id": drone_id
                        }
                        
                        detections.append(detection)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats["frames_processed"] += 1
            self.stats["detections_found"] += len(detections)
            self.stats["processing_time_avg"] = (
                (self.stats["processing_time_avg"] * (self.stats["frames_processed"] - 1) + processing_time) 
                / self.stats["frames_processed"]
            )
            
            logger.info(f"Processed frame in {processing_time:.2f}ms, found {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
            
    def _fallback_detection(self, frame: np.ndarray, gps: Dict, 
                           timestamp: str, drone_id: str) -> List[Dict]:
        """Fallback detection method for demonstration"""
        detections = []
        
        # Simple color-based detection for demonstration
        # Look for red rectangles (missing clamps) and yellow circles (debris)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red regions (missing clamps)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        # Detect yellow regions (debris)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours for red regions
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detection = {
                    "class": "missing_clamp",
                    "confidence": 0.85,
                    "bbox": [x, y, x + w, y + h],
                    "description": "Missing rail clamp detected",
                    "severity": "HIGH",
                    "action_required": "Immediate inspection required",
                    "timestamp": timestamp,
                    "gps": gps,
                    "drone_id": drone_id
                }
                detections.append(detection)
        
        # Find contours for yellow regions
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 50:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detection = {
                    "class": "debris",
                    "confidence": 0.75,
                    "bbox": [x, y, x + w, y + h],
                    "description": "Debris on track detected",
                    "severity": "LOW",
                    "action_required": "Clean track",
                    "timestamp": timestamp,
                    "gps": gps,
                    "drone_id": drone_id
                }
                detections.append(detection)
        
        return detections
        
    def _publish_detections(self, detections: List[Dict], gps: Dict, 
                           timestamp: str, drone_id: str):
        """Publish detection results to MQTT"""
        detection_data = {
            "drone_id": drone_id,
            "timestamp": timestamp,
            "gps": gps,
            "detections": detections,
            "total_detections": len(detections)
        }
        
        self.mqtt_client.publish(
            MQTT_TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        
        logger.info(f"Published {len(detections)} detections for drone {drone_id}")
        
    def _save_annotated_frame(self, frame_info: Dict):
        """Save frame with detection annotations"""
        try:
            from pathlib import Path
            import cv2
            
            # Create frames directory
            frames_dir = Path("data/frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            frame = frame_info['frame']
            detections = frame_info['detections']
            timestamp = frame_info['timestamp']
            frame_number = frame_info.get('frame_number', 0)
            
            # Create annotated image
            annotated_frame = frame.copy()
            
            for detection in detections:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                # Get detection info
                class_name = detection["class"]
                confidence = detection["confidence"]
                severity = detection.get("severity", "MEDIUM")
                
                # Choose color based on severity
                if severity == "HIGH":
                    color = (0, 0, 255)  # Red
                elif severity == "MEDIUM":
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                    
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label = f"{class_name.upper()}: {confidence:.2f}"
                
                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width + 10, y1), 
                             color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, 
                           (x1 + 5, y1 - 5), 
                           font, font_scale, (255, 255, 255), thickness)
            
            # Add timestamp and frame info
            cv2.putText(annotated_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save annotated frame
            frame_filename = f"detection_frame_{frame_number:04d}_{timestamp.replace(':', '-')}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), annotated_frame)
            
            # Save metadata
            metadata = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "gps": frame_info['gps'],
                "detections": detections,
                "drone_id": frame_info['drone_id']
            }
            
            metadata_filename = f"detection_frame_{frame_number:04d}_{timestamp.replace(':', '-')}.json"
            metadata_path = frames_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved annotated frame: {frame_path}")
            
        except Exception as e:
            logger.error(f"Error saving annotated frame: {e}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            "cache_size": len(self.detection_cache),
            "frame_buffer_size": len(self.frame_buffer)
        }
        
    def run(self):
        """Run the anomaly detector"""
        self.is_running = True
        logger.info("Anomaly detector started")
        
        try:
            while self.is_running:
                time.sleep(1)
                
                # Log statistics every 30 seconds
                if self.stats["frames_processed"] % 30 == 0 and self.stats["frames_processed"] > 0:
                    stats = self.get_statistics()
                    logger.info(f"Statistics: {stats}")
                    
        except KeyboardInterrupt:
            logger.info("Anomaly detector stopped by user")
        except Exception as e:
            logger.error(f"Error in anomaly detector: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the anomaly detector"""
        self.is_running = False
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        logger.info("Anomaly detector stopped")


def main():
    """Main function to run the anomaly detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Track Anomaly Detector")
    parser.add_argument("--model-path", help="Path to custom YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Update configuration if provided
    if args.model_path:
        YOLO_CONFIG["weights"] = args.model_path
    if args.confidence:
        YOLO_CONFIG["confidence"] = args.confidence
    if args.device:
        YOLO_CONFIG["device"] = args.device
    
    # Create and run anomaly detector
    detector = AnomalyDetector()
    
    try:
        detector.run()
    except KeyboardInterrupt:
        logger.info("Stopping anomaly detector...")
        detector.stop()


if __name__ == "__main__":
    main() 