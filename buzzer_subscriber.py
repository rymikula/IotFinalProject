#!/usr/bin/env python3
"""
MQTT Buzzer Subscriber
Listens for messages on the "buzz" topic and activates a GPIO-controlled buzzer.
"""

import sys
import time
import logging
import signal
from paho.mqtt import client as mqtt_client

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Error: RPi.GPIO library not found. This script must run on a Raspberry Pi.")
    sys.exit(1)

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
buzzer_pin = config.BUZZER_GPIO_PIN
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info("Shutting down...")
    running = False
    GPIO.cleanup()
    sys.exit(0)


def setup_gpio():
    """Initialize GPIO pin for buzzer control"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(buzzer_pin, GPIO.OUT)
    GPIO.output(buzzer_pin, GPIO.LOW)
    logger.info(f"GPIO {buzzer_pin} configured for buzzer control")


def activate_buzzer(duration=config.BUZZER_DURATION):
    """Activate buzzer for specified duration"""
    try:
        logger.info(f"Activating buzzer for {duration} seconds...")
        GPIO.output(buzzer_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(buzzer_pin, GPIO.LOW)
        logger.info("Buzzer deactivated")
    except Exception as e:
        logger.error(f"Error controlling buzzer: {e}")


def on_connect(client, userdata, flags, rc):
    """Callback for when client connects to broker"""
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}")
        # Subscribe to the buzz topic
        client.subscribe(config.MQTT_TOPIC)
        logger.info(f"Subscribed to topic: {config.MQTT_TOPIC}")
    else:
        logger.error(f"Failed to connect to broker. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    """Callback for when client disconnects from broker"""
    if rc != 0:
        logger.warning("Unexpected disconnection from broker. Will attempt to reconnect.")


def on_message(client, userdata, msg):
    """Callback for when a message is received"""
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        logger.info(f"Message received on topic '{topic}': {payload}")
        
        # Activate buzzer when message is received
        activate_buzzer()
    except Exception as e:
        logger.error(f"Error processing message: {e}")


def main():
    """Main function"""
    global running
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup GPIO
    setup_gpio()
    
    # Create MQTT client
    client = mqtt_client.Client(client_id=config.MQTT_CLIENT_ID_SUBSCRIBER)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # Connect to broker
    logger.info(f"Connecting to MQTT broker at {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}...")
    
    try:
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, config.MQTT_KEEPALIVE)
    except Exception as e:
        logger.error(f"Failed to connect to broker: {e}")
        logger.error("Please check:")
        logger.error("  1. Broker IP address in config.py")
        logger.error("  2. Broker is running and accessible")
        logger.error("  3. Network connectivity")
        GPIO.cleanup()
        sys.exit(1)
    
    # Start the loop
    logger.info("Starting MQTT client loop...")
    logger.info("Waiting for messages. Press Ctrl+C to stop.")
    
    try:
        client.loop_start()
        
        # Keep the script running
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Cleaning up...")
        client.loop_stop()
        client.disconnect()
        GPIO.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

