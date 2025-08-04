"""
Drone Client for Railway Track Anomaly Detection System

Simulates a drone capturing video and GPS data, then transmitting it via MQTT.
"""

import cv2
import json
import time
import base64
import threading
from datetime import datetime
from typing import Optional, Tuple
import logging

import paho.mqtt.client as mqtt
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE,
    MQTT_TOPIC_VIDEO, MQTT_TOPIC_GPS, MQTT_TOPIC_STATUS,
    VIDEO_CONFIG, GPS_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DroneClient:
    """Simulates a drone capturing video and GPS data"""
    
    def __init__(self, drone_id: str = "drone_001"):
        self.drone_id = drone_id
        self.is_running = False
        self.mqtt_client = None
        self.video_capture = None
        self.current_gps = (40.7128, -74.0060)  # Default coordinates (NYC)
        
        # Initialize MQTT client
        self._setup_mqtt()
        
        # Video processing settings
        self.frame_interval = VIDEO_CONFIG["frame_interval"]
        self.frame_count = 0
        
    def _setup_mqtt(self):
        """Setup MQTT client connection"""
        self.mqtt_client = mqtt.Client(client_id=f"{self.drone_id}_client")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_publish = self._on_mqtt_publish
        
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("MQTT connection successful")
            # Publish drone status
            self._publish_status("connected")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning("MQTT connection lost")
        self._publish_status("disconnected")
        
    def _on_mqtt_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logger.debug(f"Message published with ID: {mid}")
        
    def _publish_status(self, status: str):
        """Publish drone status"""
        status_data = {
            "drone_id": self.drone_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "gps": {
                "lat": self.current_gps[0],
                "lng": self.current_gps[1]
            }
        }
        
        self.mqtt_client.publish(
            MQTT_TOPIC_STATUS,
            json.dumps(status_data),
            qos=1
        )
        
    def _publish_gps(self, lat: float, lng: float):
        """Publish GPS coordinates"""
        gps_data = {
            "drone_id": self.drone_id,
            "timestamp": datetime.now().isoformat(),
            "lat": lat,
            "lng": lng
        }
        
        self.mqtt_client.publish(
            MQTT_TOPIC_GPS,
            json.dumps(gps_data),
            qos=1
        )
        
    def _publish_video_frame(self, frame: np.ndarray, gps: Tuple[float, float]):
        """Publish video frame as base64 encoded image"""
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, VIDEO_CONFIG["compression_quality"]])
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare video data
        video_data = {
            "drone_id": self.drone_id,
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frame_count,
            "gps": {
                "lat": gps[0],
                "lng": gps[1]
            },
            "frame_data": frame_base64,
            "frame_size": {
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
        }
        
        # Publish to MQTT
        self.mqtt_client.publish(
            MQTT_TOPIC_VIDEO,
            json.dumps(video_data),
            qos=1
        )
        
        logger.debug(f"Published frame {self.frame_count} with GPS: {gps}")
        
    def _simulate_gps_movement(self) -> Tuple[float, float]:
        """Simulate GPS movement along a railway track"""
        # Simulate movement along a railway track
        base_lat, base_lng = 40.7128, -74.0060
        
        # Add some movement (simulating drone flying along tracks)
        movement_factor = time.time() * 0.0001
        new_lat = base_lat + movement_factor
        new_lng = base_lng + movement_factor * 0.5
        
        self.current_gps = (new_lat, new_lng)
        return self.current_gps
        
    def _generate_synthetic_video(self) -> np.ndarray:
        """Generate synthetic video frame for demonstration"""
        # Create a synthetic railway track image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw railway tracks
        cv2.line(frame, (100, 240), (540, 240), (128, 128, 128), 8)  # Main track
        cv2.line(frame, (100, 248), (540, 248), (128, 128, 128), 8)  # Parallel track
        
        # Draw sleepers
        for i in range(0, 640, 40):
            cv2.line(frame, (i, 220), (i, 260), (64, 64, 64), 2)
            
        # Occasionally add anomalies for testing
        if self.frame_count % 100 == 0:  # Every 100 frames
            # Add a "missing clamp" (red rectangle)
            cv2.rectangle(frame, (300, 230), (320, 250), (0, 0, 255), -1)
            cv2.putText(frame, "MISSING CLAMP", (280, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        elif self.frame_count % 150 == 0:  # Every 150 frames
            # Add "debris" (yellow circle)
            cv2.circle(frame, (400, 240), 15, (0, 255, 255), -1)
            cv2.putText(frame, "DEBRIS", (380, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add GPS coordinates
        gps_text = f"GPS: {self.current_gps[0]:.6f}, {self.current_gps[1]:.6f}"
        cv2.putText(frame, gps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def start_video_capture(self, video_source: Optional[str] = None):
        """Start video capture from camera or video file"""
        if video_source is None:
            # Use synthetic video
            self.video_capture = None
            logger.info("Using synthetic video generation")
        else:
            # Use real video source
            self.video_capture = cv2.VideoCapture(video_source)
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open video source: {video_source}")
                return
            logger.info(f"Started video capture from: {video_source}")
            
    def run(self, duration: int = 60):
        """Run the drone client for specified duration"""
        self.is_running = True
        start_time = time.time()
        
        logger.info(f"Starting drone client for {duration} seconds")
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Update GPS
                gps = self._simulate_gps_movement()
                self._publish_gps(*gps)
                
                # Capture and process video frame
                if self.frame_count % self.frame_interval == 0:
                    if self.video_capture is None:
                        # Generate synthetic frame
                        frame = self._generate_synthetic_video()
                    else:
                        # Capture from real source
                        ret, frame = self.video_capture.read()
                        if not ret:
                            logger.warning("Failed to read frame from video source")
                            continue
                            
                    # Publish frame
                    self._publish_video_frame(frame, gps)
                    
                self.frame_count += 1
                time.sleep(1.0 / VIDEO_CONFIG["frame_rate"])
                
        except KeyboardInterrupt:
            logger.info("Drone client stopped by user")
        except Exception as e:
            logger.error(f"Error in drone client: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the drone client"""
        self.is_running = False
        
        if self.video_capture:
            self.video_capture.release()
            
        if self.mqtt_client:
            self._publish_status("stopped")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        logger.info("Drone client stopped")


def main():
    """Main function to run the drone client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Track Drone Client")
    parser.add_argument("--drone-id", default="drone_001", help="Drone ID")
    parser.add_argument("--duration", type=int, default=60, help="Run duration in seconds")
    parser.add_argument("--video-source", help="Video source (camera index or file path)")
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame")
    
    args = parser.parse_args()
    
    # Create and run drone client
    drone = DroneClient(drone_id=args.drone_id)
    drone.frame_interval = args.frame_interval
    drone.start_video_capture(args.video_source)
    
    try:
        drone.run(duration=args.duration)
    except KeyboardInterrupt:
        logger.info("Stopping drone client...")
        drone.stop()


if __name__ == "__main__":
    main() 