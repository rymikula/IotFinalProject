# MQTT Buzzer Subscriber Setup

## Overview

Create a Python-based MQTT subscriber that runs on a Raspberry Pi, connects to a Mosquitto broker, subscribes to the "buzz" topic, and triggers a GPIO-controlled buzzer for 2 seconds when a message is received.

## Implementation Details

### Files to Create

1. **`buzzer_subscriber.py`** - Main subscriber script
   - Uses `paho-mqtt` library for MQTT communication
   - Uses `RPi.GPIO` library for GPIO control
   - Subscribes to "buzz" topic
   - On message received: activates GPIO pin for 2 seconds, then deactivates
   - Includes error handling and logging
   - Configurable GPIO pin (default: GPIO 18)
   - Configurable MQTT broker connection settings

2. **`button_publisher.py`** - Button publisher script
   - Uses `paho-mqtt` library for MQTT communication
   - Uses `RPi.GPIO` library for GPIO button input
   - Monitors GPIO button pin for press events
   - Publishes message to "buzz" topic when button is pressed
   - Includes button debouncing to prevent multiple triggers
   - Configurable button GPIO pin (default: GPIO 23)
   - Configurable MQTT broker connection settings
   - Can run on same or different Pi as subscriber

3. **`config.py`** - Configuration file
   - MQTT broker host/IP address
   - MQTT broker port (default: 1883)
   - Topic name ("buzz")
   - Buzzer GPIO pin number (for subscriber)
   - Button GPIO pin number (for publisher, default: GPIO 23)
   - Buzzer duration (2 seconds)
   - Button debounce time (default: 0.2 seconds)

4. **`requirements.txt`** - Python dependencies
   - paho-mqtt
   - RPi.GPIO

5. **`README.md`** - Setup and usage instructions
   - Installation steps
   - Configuration guide
   - Running the subscriber
   - Running the button publisher
   - Troubleshooting tips

### Technical Approach

- **MQTT Client**: Use paho-mqtt with persistent connection and auto-reconnect
- **GPIO Control (Subscriber)**: Set GPIO pin HIGH to activate buzzer, LOW to deactivate
- **GPIO Input (Publisher)**: Monitor button pin with pull-up resistor, detect falling edge (button press)
- **Button Debouncing**: Use time-based debouncing to prevent multiple triggers from single press
- **Timing**: Use `time.sleep(2)` to maintain buzzer state for 2 seconds
- **Thread Safety**: Ensure GPIO operations don't block MQTT message handling
- **Configuration**: Use config file with defaults, easily adjustable

### Configuration Assumptions

- Default buzzer GPIO pin: 18 (configurable)
- Default button GPIO pin: 23 (configurable)
- MQTT broker port: 1883 (standard, configurable)
- Topic: "buzz" (fixed as specified)
- Buzzer duration: 2 seconds (as specified)
- Button debounce time: 0.2 seconds (configurable)
- Broker IP/hostname: will be configurable in config.py

## Notes

- The buzzer is already wired (assumes relay/transistor circuit is in place)
- GPIO pin can be changed in config.py
- MQTT broker connection details need to be configured before running
- Script runs continuously until interrupted (Ctrl+C)

## Project Setup

### System Architecture

- **MQTT Broker**: Running on a separate Raspberry Pi with Mosquitto
- **Publisher Options**:
  - WiFi-enabled doorbell device that publishes to "buzz" topic
  - Physical button on Raspberry Pi running `button_publisher.py` (for testing or alternative input)
- **Subscriber**: Raspberry Pi running `buzzer_subscriber.py`, subscribed to "buzz" topic, controlling the buzzer

### Hardware Requirements

- Raspberry Pi (subscriber) - runs buzzer_subscriber.py
- Raspberry Pi (optional publisher) - runs button_publisher.py if using physical button
- 12V DC Active Electronic Buzzer (Electromagnetic Type 12095)
- Relay or transistor circuit for controlling 12V buzzer from 3.3V GPIO
- Buzzer GPIO pin connection (default: GPIO 18)
- Push button (if using button_publisher.py) with pull-up resistor
- Button GPIO pin connection (default: GPIO 23)

### To-dos

- [ ] Create requirements.txt with paho-mqtt and RPi.GPIO dependencies
- [ ] Create config.py with MQTT broker settings, GPIO pins, and buzzer duration configuration
- [ ] Create buzzer_subscriber.py with MQTT subscriber logic and GPIO buzzer control
- [ ] Create button_publisher.py with MQTT publisher logic and GPIO button input with debouncing
- [ ] Create README.md with installation, configuration, and usage instructions
