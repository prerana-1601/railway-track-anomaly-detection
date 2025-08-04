"""
MQTT Server for Railway Track Anomaly Detection System

Handles MQTT message routing and broker functionality for the railway detection system.
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import threading

import paho.mqtt.client as mqtt

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE,
    MQTT_TOPIC_VIDEO, MQTT_TOPIC_GPS, MQTT_TOPIC_DETECTION, MQTT_TOPIC_STATUS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MQTTServer:
    """MQTT Server for handling railway detection messages"""
    
    def __init__(self):
        self.is_running = False
        self.clients = {}  # Store connected clients
        self.message_history = {}  # Store recent messages
        self.stats = {
            "messages_received": 0,
            "messages_forwarded": 0,
            "clients_connected": 0,
            "start_time": None
        }
        
        # Initialize MQTT broker
        self._setup_broker()
        
    def _setup_broker(self):
        """Setup MQTT broker"""
        try:
            # Create broker client
            self.broker = mqtt.Client(client_id="railway_broker")
            self.broker.on_connect = self._on_broker_connect
            self.broker.on_disconnect = self._on_broker_disconnect
            self.broker.on_message = self._on_broker_message
            
            # Connect to broker
            self.broker.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.broker.loop_start()
            
            logger.info(f"MQTT broker started on {MQTT_BROKER}:{MQTT_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to start MQTT broker: {e}")
            
    def _on_broker_connect(self, client, userdata, flags, rc):
        """Broker connection callback"""
        if rc == 0:
            logger.info("MQTT broker connected successfully")
            # Subscribe to all relevant topics
            topics = [
                (MQTT_TOPIC_VIDEO, 1),
                (MQTT_TOPIC_GPS, 1),
                (MQTT_TOPIC_DETECTION, 1),
                (MQTT_TOPIC_STATUS, 1)
            ]
            
            for topic, qos in topics:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"MQTT broker connection failed with code {rc}")
            
    def _on_broker_disconnect(self, client, userdata, rc):
        """Broker disconnection callback"""
        logger.warning("MQTT broker disconnected")
        
    def _on_broker_message(self, client, userdata, msg):
        """Broker message callback"""
        try:
            # Parse message
            payload = json.loads(msg.payload.decode('utf-8'))
            topic = msg.topic
            
            # Update statistics
            self.stats["messages_received"] += 1
            
            # Store message in history
            if topic not in self.message_history:
                self.message_history[topic] = []
            
            self.message_history[topic].append({
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "topic": topic
            })
            
            # Keep only last 100 messages per topic
            if len(self.message_history[topic]) > 100:
                self.message_history[topic] = self.message_history[topic][-100:]
            
            # Handle different message types
            if topic == MQTT_TOPIC_VIDEO:
                self._handle_video_message(payload)
            elif topic == MQTT_TOPIC_GPS:
                self._handle_gps_message(payload)
            elif topic == MQTT_TOPIC_DETECTION:
                self._handle_detection_message(payload)
            elif topic == MQTT_TOPIC_STATUS:
                self._handle_status_message(payload)
                
            # Forward message to other subscribers
            self._forward_message(topic, payload)
            
        except Exception as e:
            logger.error(f"Error processing broker message: {e}")
            
    def _handle_video_message(self, payload: Dict[str, Any]):
        """Handle video frame messages"""
        drone_id = payload.get('drone_id', 'unknown')
        frame_number = payload.get('frame_number', 0)
        
        logger.info(f"Received video frame {frame_number} from drone {drone_id}")
        
        # Here you could add additional processing like:
        # - Frame validation
        # - Quality assessment
        # - Storage to database
        # - Forwarding to analysis services
        
    def _handle_gps_message(self, payload: Dict[str, Any]):
        """Handle GPS coordinate messages"""
        drone_id = payload.get('drone_id', 'unknown')
        lat = payload.get('lat', 0)
        lng = payload.get('lng', 0)
        
        logger.info(f"Received GPS coordinates ({lat:.6f}, {lng:.6f}) from drone {drone_id}")
        
        # Here you could add additional processing like:
        # - Coordinate validation
        # - Route tracking
        # - Geofencing
        # - Alert generation for out-of-bounds areas
        
    def _handle_detection_message(self, payload: Dict[str, Any]):
        """Handle detection result messages"""
        drone_id = payload.get('drone_id', 'unknown')
        detections = payload.get('detections', [])
        total_detections = payload.get('total_detections', 0)
        
        logger.info(f"Received {total_detections} detections from drone {drone_id}")
        
        # Log detection details
        for detection in detections:
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            severity = detection.get('severity', 'MEDIUM')
            
            logger.info(f"Detection: {class_name} (confidence: {confidence:.2f}, severity: {severity})")
            
        # Here you could add additional processing like:
        # - Alert generation for high-severity detections
        # - Database storage
        # - Report generation
        # - Notification systems
        
    def _handle_status_message(self, payload: Dict[str, Any]):
        """Handle status messages"""
        drone_id = payload.get('drone_id', 'unknown')
        status = payload.get('status', 'unknown')
        gps = payload.get('gps', {})
        
        logger.info(f"Drone {drone_id} status: {status}")
        
        # Update client tracking
        if status == "connected":
            self.clients[drone_id] = {
                "status": status,
                "last_seen": datetime.now().isoformat(),
                "gps": gps
            }
            self.stats["clients_connected"] = len(self.clients)
        elif status == "disconnected":
            if drone_id in self.clients:
                del self.clients[drone_id]
                self.stats["clients_connected"] = len(self.clients)
                
    def _forward_message(self, topic: str, payload: Dict[str, Any]):
        """Forward message to other subscribers"""
        try:
            # Publish to the same topic for other subscribers
            self.broker.publish(topic, json.dumps(payload), qos=1)
            self.stats["messages_forwarded"] += 1
            
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get current server statistics"""
        uptime = 0
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
            
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "connected_clients": list(self.clients.keys()),
            "message_history_size": {topic: len(messages) for topic, messages in self.message_history.items()}
        }
        
    def get_recent_messages(self, topic: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Get recent messages from history"""
        if topic:
            messages = self.message_history.get(topic, [])
            return {topic: messages[-limit:]}
        else:
            return {topic: messages[-limit:] for topic, messages in self.message_history.items()}
            
    def run(self):
        """Run the MQTT server"""
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        logger.info("MQTT server started")
        
        try:
            while self.is_running:
                time.sleep(1)
                
                # Log statistics every 60 seconds
                if self.stats["messages_received"] % 60 == 0 and self.stats["messages_received"] > 0:
                    stats = self.get_statistics()
                    logger.info(f"Server Statistics: {stats}")
                    
        except KeyboardInterrupt:
            logger.info("MQTT server stopped by user")
        except Exception as e:
            logger.error(f"Error in MQTT server: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the MQTT server"""
        self.is_running = False
        
        if hasattr(self, 'broker'):
            self.broker.loop_stop()
            self.broker.disconnect()
            
        logger.info("MQTT server stopped")


def main():
    """Main function to run the MQTT server"""
    import argparse
    global MQTT_BROKER, MQTT_PORT
    parser = argparse.ArgumentParser(description="Railway Track MQTT Server")
    parser.add_argument("--host", default=MQTT_BROKER, help="MQTT broker host")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help="MQTT broker port")
    
    args = parser.parse_args()
    MQTT_BROKER = args.host
    MQTT_PORT = args.port
    
    # Create and run MQTT server
    server = MQTTServer()
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Stopping MQTT server...")
        server.stop()


if __name__ == "__main__":
    main() 