"""
Configuration file for MQTT buzzer subscriber and button publisher.
Edit these values to match your setup.
"""

# MQTT Broker Configuration
MQTT_BROKER_HOST = "192.168.1.100"  # Change to your broker Pi's IP address
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "buzz"

# GPIO Pin Configuration
BUZZER_GPIO_PIN = 18  # GPIO pin connected to buzzer relay/transistor
BUTTON_GPIO_PIN = 23  # GPIO pin connected to button

# Timing Configuration
BUZZER_DURATION = 2  # Duration in seconds to keep buzzer active
BUTTON_DEBOUNCE_TIME = 0.2  # Debounce time in seconds for button presses

# MQTT Client Configuration
MQTT_CLIENT_ID_SUBSCRIBER = "buzzer_subscriber"
MQTT_CLIENT_ID_PUBLISHER = "button_publisher"
MQTT_KEEPALIVE = 60  # Keepalive interval in seconds

