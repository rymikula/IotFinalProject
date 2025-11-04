#!/bin/bash

# Mosquitto MQTT Broker Setup Script for Raspberry Pi
# This script installs and configures Mosquitto for network access

set -e  # Exit on error

echo "Installing Mosquitto MQTT Broker..."

# Update package list
sudo apt update

# Install Mosquitto broker and client tools
sudo apt install -y mosquitto mosquitto-clients

echo "Configuring Mosquitto for network access..."

# Backup original config if it exists
if [ -f /etc/mosquitto/mosquitto.conf ]; then
    sudo cp /etc/mosquitto/mosquitto.conf /etc/mosquitto/mosquitto.conf.backup
fi

# Create or update configuration to allow network connections
sudo tee /etc/mosquitto/mosquitto.conf > /dev/null <<EOF
# Mosquitto MQTT Broker Configuration
# Listens on all interfaces for network access
listener 1883 0.0.0.0

# Allow anonymous connections (for simple setup)
# For production, consider using authentication
allow_anonymous true

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
log_type information

# Persistence
persistence true
persistence_location /var/lib/mosquitto/
EOF

echo "Enabling and starting Mosquitto service..."

# Enable Mosquitto to start on boot
sudo systemctl enable mosquitto

# Start Mosquitto service
sudo systemctl restart mosquitto

# Check if firewall is active and open port if needed
if command -v ufw &> /dev/null; then
    if sudo ufw status | grep -q "Status: active"; then
        echo "Opening firewall port 1883..."
        sudo ufw allow 1883/tcp
        echo "Firewall configured."
    fi
fi

# Wait a moment for service to start
sleep 2

# Check service status
if sudo systemctl is-active --quiet mosquitto; then
    echo ""
    echo "✓ Mosquitto broker is running successfully!"
    echo ""
    echo "Broker is listening on port 1883"
    echo "To test locally, run: mosquitto_sub -h localhost -t test"
    echo ""
    echo "To find your IP address, run: hostname -I"
    echo "Other devices can connect using: mosquitto_sub -h <your-ip> -t test"
else
    echo "Error: Mosquitto service failed to start. Check logs with: sudo journalctl -u mosquitto"
    exit 1
fi

