#!/bin/bash
# MQTT Installation Script for Railway Track Anomaly Detection System

echo "🚀 Installing MQTT Broker for Railway Track Anomaly Detection System"
echo "=================================================================="

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "📦 Detected Linux system"
    
    # Check if apt is available (Debian/Ubuntu)
    if command -v apt &> /dev/null; then
        echo "🔧 Installing Mosquitto on Ubuntu/Debian..."
        sudo apt update
        sudo apt install -y mosquitto mosquitto-clients
        
        echo "🚀 Starting Mosquitto service..."
        sudo systemctl enable mosquitto
        sudo systemctl start mosquitto
        
        echo "✅ Mosquitto installed and started successfully!"
        
    # Check if yum is available (RHEL/CentOS)
    elif command -v yum &> /dev/null; then
        echo "🔧 Installing Mosquitto on RHEL/CentOS..."
        sudo yum install -y mosquitto mosquitto-clients
        
        echo "🚀 Starting Mosquitto service..."
        sudo systemctl enable mosquitto
        sudo systemctl start mosquitto
        
        echo "✅ Mosquitto installed and started successfully!"
        
    else
        echo "❌ Unsupported Linux distribution. Please install Mosquitto manually:"
        echo "   Visit: https://mosquitto.org/download/"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📦 Detected macOS system"
    
    if command -v brew &> /dev/null; then
        echo "🔧 Installing Mosquitto using Homebrew..."
        brew install mosquitto
        
        echo "🚀 Starting Mosquitto service..."
        brew services start mosquitto
        
        echo "✅ Mosquitto installed and started successfully!"
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   Then run: brew install mosquitto"
    fi
    
else
    # Windows or other
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "📋 Please install Mosquitto manually:"
    echo "   1. Visit: https://mosquitto.org/download/"
    echo "   2. Download and install the Windows version"
    echo "   3. Start the Mosquitto service"
fi

echo ""
echo "🧪 Testing MQTT connection..."
echo "Publishing test message..."

# Test MQTT connection
if command -v mosquitto_pub &> /dev/null; then
    mosquitto_pub -h localhost -t test/railway -m "Hello from Railway Detection System!" -q 1
    
    if [ $? -eq 0 ]; then
        echo "✅ MQTT connection test successful!"
        echo "🎉 MQTT broker is ready for the Railway Track Anomaly Detection System"
    else
        echo "❌ MQTT connection test failed"
        echo "🔧 Troubleshooting tips:"
        echo "   1. Check if Mosquitto is running: sudo systemctl status mosquitto"
        echo "   2. Restart Mosquitto: sudo systemctl restart mosquitto"
        echo "   3. Check firewall settings"
    fi
else
    echo "❌ mosquitto_pub not found. Please ensure Mosquitto is properly installed."
fi

echo ""
echo "📚 Next steps:"
echo "   1. Run: python scripts/run_demo.py --duration 30"
echo "   2. Check the generated PDF reports in data/reports/"
echo "   3. View annotated frames in data/frames/" 