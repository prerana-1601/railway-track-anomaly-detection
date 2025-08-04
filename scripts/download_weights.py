#!/usr/bin/env python3
"""
Script to download YOLO model weights for Railway Track Anomaly Detection System
"""

import os
import sys
import requests
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file from URL with progress tracking"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
        
        print(f"\nDownloaded: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def main():
    """Download YOLO model weights"""
    # Create models directory
    models_dir = PROJECT_ROOT / "models" / "yolo_weights"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO model URLs (using pre-trained models for demonstration)
    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    }
    
    print("Downloading YOLO model weights...")
    print("Note: For production use, you should train custom models on railway track data.")
    print()
    
    for model_name, url in model_urls.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"Model {model_name} already exists, skipping...")
            continue
            
        print(f"Downloading {model_name}...")
        success = download_file(url, model_path)
        
        if success:
            print(f"Successfully downloaded {model_name}")
        else:
            print(f"Failed to download {model_name}")
    
    print("\nDownload complete!")
    print("Available models:")
    for model_file in models_dir.glob("*.pt"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.1f} MB)")
    
    print("\nTo use a specific model, update the weights path in config/settings.py:")
    print("YOLO_CONFIG['weights'] = 'models/yolo_weights/yolov8n.pt'")


if __name__ == "__main__":
    main() 