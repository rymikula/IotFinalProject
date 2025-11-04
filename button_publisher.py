#!/usr/bin/env python3
"""
MQTT Button Publisher
Monitors a GPIO button and publishes to the "buzz" topic when pressed.
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
button_pin = config.BUTTON_GPIO_PIN
debounce_time = config.BUTTON_DEBOUNCE_TIME
last_press_time = 0
running = True
mqtt_client_instance = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info("Shutting down...")
    running = False
    GPIO.cleanup()
    sys.exit(0)


def setup_gpio():
    """Initialize GPIO pin for button input"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    # Set up button pin with pull-up resistor (button connects to GND when pressed)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    logger.info(f"GPIO {button_pin} configured for button input (pull-up enabled)")


def button_callback(channel):
    """Callback function for button press"""
    global last_press_time
    
    current_time = time.time()
    
    # Debounce: ignore presses that occur too soon after the last one
    if current_time - last_press_time < debounce_time:
        return
    
    last_press_time = current_time
    
    # Small delay to ensure button state is stable
    time.sleep(0.01)
    
    # Check if button is still pressed (LOW = pressed with pull-up)
    if GPIO.input(button_pin) == GPIO.LOW:
        logger.info("Button pressed!")
        publish_buzz_message()


def publish_buzz_message():
    """Publish message to buzz topic"""
    global mqtt_client_instance
    
    if mqtt_client_instance is None:
        logger.error("MQTT client not initialized")
        return
    
    try:
        message = "buzz"
        result = mqtt_client_instance.publish(config.MQTT_TOPIC, message, qos=1)
        
        if result.rc == mqtt_client.MQTT_ERR_SUCCESS:
            logger.info(f"Published '{message}' to topic '{config.MQTT_TOPIC}'")
        else:
            logger.error(f"Failed to publish message. Return code: {result.rc}")
    except Exception as e:
        logger.error(f"Error publishing message: {e}")


def on_connect(client, userdata, flags, rc):
    """Callback for when client connects to broker"""
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}")
    else:
        logger.error(f"Failed to connect to broker. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    """Callback for when client disconnects from broker"""
    if rc != 0:
        logger.warning("Unexpected disconnection from broker. Will attempt to reconnect.")


def on_publish(client, userdata, mid):
    """Callback for when message is published"""
    logger.debug(f"Message published with mid: {mid}")


def main():
    """Main function"""
    global running, mqtt_client_instance
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup GPIO
    setup_gpio()
    
    # Create MQTT client
    mqtt_client_instance = mqtt_client.Client(client_id=config.MQTT_CLIENT_ID_PUBLISHER)
    mqtt_client_instance.on_connect = on_connect
    mqtt_client_instance.on_disconnect = on_disconnect
    mqtt_client_instance.on_publish = on_publish
    
    # Connect to broker
    logger.info(f"Connecting to MQTT broker at {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}...")
    
    try:
        mqtt_client_instance.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, config.MQTT_KEEPALIVE)
    except Exception as e:
        logger.error(f"Failed to connect to broker: {e}")
        logger.error("Please check:")
        logger.error("  1. Broker IP address in config.py")
        logger.error("  2. Broker is running and accessible")
        logger.error("  3. Network connectivity")
        GPIO.cleanup()
        sys.exit(1)
    
    # Setup button interrupt
    GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=button_callback, bouncetime=int(debounce_time * 1000))
    logger.info(f"Button monitoring started on GPIO {button_pin}")
    logger.info(f"Debounce time: {debounce_time} seconds")
    
    # Start the MQTT loop
    logger.info("Starting MQTT client loop...")
    logger.info("Waiting for button presses. Press Ctrl+C to stop.")
    
    try:
        mqtt_client_instance.loop_start()
        
        # Keep the script running
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Cleaning up...")
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()
        GPIO.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

