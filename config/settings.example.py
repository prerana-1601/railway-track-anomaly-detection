"""
Example configuration settings for Railway Track Anomaly Detection System

Copy this file to config/settings.py and modify the values as needed.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_KEEPALIVE = 60
MQTT_TOPIC_VIDEO = "railway/video"
MQTT_TOPIC_GPS = "railway/gps"
MQTT_TOPIC_DETECTION = "railway/detection"
MQTT_TOPIC_STATUS = "railway/status"

# YOLO Model Configuration
YOLO_CONFIG = {
    "weights": str(PROJECT_ROOT / "models" / "yolo_weights" / "yolov8n.pt"),
    "confidence": 0.5,
    "iou_threshold": 0.45,
    "classes": ["missing_clamp", "ridge", "crack", "debris", "missing_bolt"],
    "img_size": 640,
    "device": "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "frame_rate": 30,
    "frame_interval": 1,  # Process every nth frame
    "max_frame_size": (1920, 1080),
    "compression_quality": 85
}

# GPS Configuration
GPS_CONFIG = {
    "update_interval": 1.0,  # seconds
    "precision": 6  # decimal places
}

# PDF Report Configuration
PDF_CONFIG = {
    "page_size": "A4",
    "margin": 50,
    "images_per_page": 2,
    "include_gps": True,
    "include_timestamp": True,
    "output_dir": str(PROJECT_ROOT / "data" / "reports")
}

# Data Storage Configuration
DATA_CONFIG = {
    "videos_dir": str(PROJECT_ROOT / "data" / "videos"),
    "frames_dir": str(PROJECT_ROOT / "data" / "frames"),
    "reports_dir": str(PROJECT_ROOT / "data" / "reports"),
    "max_storage_gb": 10,
    "cleanup_interval_hours": 24
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(PROJECT_ROOT / "logs" / "railway_detection.log")
}

# Detection Classes and Descriptions
DETECTION_CLASSES = {
    "missing_clamp": {
        "description": "Missing rail clamp detected",
        "severity": "HIGH",
        "action_required": "Immediate inspection required"
    },
    "ridge": {
        "description": "Railway track ridge detected",
        "severity": "MEDIUM",
        "action_required": "Schedule maintenance"
    },
    "crack": {
        "description": "Track crack detected",
        "severity": "HIGH",
        "action_required": "Immediate repair required"
    },
    "debris": {
        "description": "Debris on track detected",
        "severity": "LOW",
        "action_required": "Clean track"
    },
    "missing_bolt": {
        "description": "Missing track bolt detected",
        "severity": "HIGH",
        "action_required": "Replace bolt immediately"
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "max_processing_time_ms": 100,
    "batch_size": 4,
    "enable_caching": True,
    "cache_size": 1000
}

# Database Configuration (if using)
DATABASE_CONFIG = {
    "type": "sqlite",  # or "postgresql", "mysql"
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "railway_detection"),
    "username": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "sqlite_path": str(PROJECT_ROOT / "data" / "railway_detection.db")
}

# API Configuration (if exposing REST API)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "cors_origins": ["*"]
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        PROJECT_ROOT / "data" / "videos",
        PROJECT_ROOT / "data" / "frames", 
        PROJECT_ROOT / "data" / "reports",
        PROJECT_ROOT / "models" / "yolo_weights",
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories() 