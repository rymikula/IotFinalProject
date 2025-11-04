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

2. **`config.py`** - Configuration file
   - MQTT broker host/IP address
   - MQTT broker port (default: 1883)
   - Topic name ("buzz")
   - GPIO pin number
   - Buzzer duration (2 seconds)

3. **`requirements.txt`** - Python dependencies
   - paho-mqtt
   - RPi.GPIO

4. **`README.md`** - Setup and usage instructions
   - Installation steps
   - Configuration guide
   - Running the subscriber
   - Troubleshooting tips

### Technical Approach

- **MQTT Client**: Use paho-mqtt with persistent connection and auto-reconnect
- **GPIO Control**: Set GPIO pin HIGH to activate buzzer, LOW to deactivate
- **Timing**: Use `time.sleep(2)` to maintain buzzer state for 2 seconds
- **Thread Safety**: Ensure GPIO operations don't block MQTT message handling
- **Configuration**: Use config file with defaults, easily adjustable

### Configuration Assumptions

- Default GPIO pin: 18 (configurable)
- MQTT broker port: 1883 (standard, configurable)
- Topic: "buzz" (fixed as specified)
- Buzzer duration: 2 seconds (as specified)
- Broker IP/hostname: will be configurable in config.py

## Notes

- The buzzer is already wired (assumes relay/transistor circuit is in place)
- GPIO pin can be changed in config.py
- MQTT broker connection details need to be configured before running
- Script runs continuously until interrupted (Ctrl+C)

## Project Setup

### System Architecture

- **MQTT Broker**: Running on a separate Raspberry Pi with Mosquitto
- **Publisher**: WiFi-enabled doorbell device that publishes to "buzz" topic
- **Subscriber**: Raspberry Pi running this code, subscribed to "buzz" topic, controlling the buzzer

### Hardware Requirements

- Raspberry Pi (subscriber)
- 12V DC Active Electronic Buzzer (Electromagnetic Type 12095)
- Relay or transistor circuit for controlling 12V buzzer from 3.3V GPIO
- GPIO pin connection (default: GPIO 18)

### To-dos

- [ ] Create requirements.txt with paho-mqtt and RPi.GPIO dependencies
- [ ] Create config.py with MQTT broker settings, GPIO pin, and buzzer duration configuration
- [ ] Create buzzer_subscriber.py with MQTT subscriber logic and GPIO buzzer control
- [ ] Create README.md with installation, configuration, and usage instructions
