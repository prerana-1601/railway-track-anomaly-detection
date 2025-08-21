# Quick Start Guide

This guide will help you get the Railway Track Anomaly Detection System running in minutes.

## Prerequisites

- Python 3.8 or higher
- Git
- Internet connection (for downloading dependencies and models)
- MQTT broker (Mosquitto recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prerana-1601/railway-track-anomaly-detection.git
   cd railway-track-anomaly-detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MQTT broker (if not already installed)**
   ```bash
   # Option A: Use the installation script (recommended)
   ./scripts/install_mqtt.sh
   
   # Option B: Manual installation
   # Ubuntu/Debian
   sudo apt install mosquitto mosquitto-clients
   sudo systemctl enable mosquitto
   sudo systemctl start mosquitto
   
   # macOS
   brew install mosquitto
   brew services start mosquitto
   
   # Windows: Download from https://mosquitto.org/download/
   ```

5. **Download YOLO model weights**
   ```bash
   python scripts/download_weights.py
   ```

6. **Configure settings (optional)**
   ```bash
   cp config/settings.example.py config/settings.py
   # Edit config/settings.py if needed
   ```

7. **Test MQTT connection**
   ```bash
   mosquitto_pub -h localhost -t test/topic -m "Hello MQTT"
   mosquitto_sub -h localhost -t test/topic
   ```

## Running the Demo

The easiest way to see the system in action is to run the demo script:

```bash
python scripts/run_demo.py --duration 30
```

This will:
- Start the MQTT server
- Start the anomaly detector
- Simulate a drone capturing video
- Generate a PDF report with detections

## Running Individual Components

### 1. MQTT Server
```bash
python server/mqtt_server.py
```

### 2. Anomaly Detector
```bash
python server/anomaly_detector.py
```

### 3. Drone Client (Simulation)
```bash
python client/drone_client.py --duration 60
```

### 4. PDF Report Generator
```bash
python server/pdf_generator.py --output test_report.pdf
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_detector.py
```

## Configuration

The system is highly configurable. Key settings can be modified in `config/settings.py`:

- **MQTT Settings**: Broker host, port, topics
- **YOLO Settings**: Model weights, confidence threshold, device (CPU/GPU)
- **Video Settings**: Frame rate, processing interval
- **PDF Settings**: Report format, image layout

## Troubleshooting

### Common Issues

#### **MQTT Connection Issues**

**Error: `ModuleNotFoundError: No module named 'paho'`**
```bash
pip install paho-mqtt
```

**Error: `Failed to start MQTT broker: [Errno 111] Connection refused`**
```bash
# Check if Mosquitto is running
sudo systemctl status mosquitto

# Start Mosquitto if not running
sudo systemctl start mosquitto

# Enable auto-start
sudo systemctl enable mosquitto
```

**Error: `Connection refused` when running components**
```bash
# Test MQTT broker
mosquitto_pub -h localhost -t test/topic -m "Hello"
mosquitto_sub -h localhost -t test/topic

# If this fails, restart Mosquitto
sudo systemctl restart mosquitto
```

#### **Alternative: Use Cloud MQTT Broker**
If local MQTT setup fails, use a cloud broker:
1. Go to https://www.hivemq.com/public-mqtt-broker/
2. Update `config/settings.py`:
   ```python
   MQTT_BROKER = "broker.hivemq.com"
   MQTT_PORT = 1883
   ```

1. **Import errors**: Make sure you're in the project root directory
2. **MQTT connection failed**: Check if MQTT broker is running (default: localhost:1883)
3. **YOLO model not found**: Run `python scripts/download_weights.py`
4. **Permission errors**: Check file permissions for data directories

### GPU Support

To use GPU acceleration:

1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Set environment variable:
   ```bash
   export USE_GPU=true
   ```

### Logs

Check logs for debugging:
- Application logs: `logs/railway_detection.log`
- Console output shows real-time status

## Next Steps

1. **Custom Training**: Train YOLO on your own railway track data
2. **Real Drone Integration**: Replace simulation with actual drone hardware
3. **Database Integration**: Add persistent storage for detections
4. **Web Interface**: Build a web dashboard for monitoring
5. **Alert System**: Add email/SMS notifications for critical detections

## Support

- Check the main [README.md](README.md) for detailed documentation
- Review the test files for usage examples
- Open an issue on GitHub for bugs or feature requests

---

