# MQTT Buzzer System

A Raspberry Pi-based MQTT buzzer system that triggers a 12V buzzer when messages are received on the "buzz" topic. Includes both a subscriber (buzzer controller) and an optional button publisher for testing.

## System Architecture

- **MQTT Broker**: Running on a separate Raspberry Pi with Mosquitto
- **Publisher Options**:
  - WiFi-enabled doorbell device that publishes to "buzz" topic
  - Physical button on Raspberry Pi running `button_publisher.py` (for testing or alternative input)
- **Subscriber**: Raspberry Pi running `buzzer_subscriber.py`, subscribed to "buzz" topic, controlling the buzzer

## Hardware Requirements

- Raspberry Pi (broker) - runs Mosquitto MQTT broker
- Raspberry Pi (subscriber) - runs buzzer_subscriber.py
- Raspberry Pi (optional publisher) - runs button_publisher.py if using physical button
- 12V DC Active Electronic Buzzer (Electromagnetic Type 12095)
- Relay or transistor circuit for controlling 12V buzzer from 3.3V GPIO
- Buzzer GPIO pin connection (default: GPIO 18)
- Push button (if using button_publisher.py) with pull-up resistor
- Button GPIO pin connection (default: GPIO 23)

## Installation

### 1. Broker Setup (Mosquitto)

**Option A: Using the setup script (Recommended)**

1. Copy `broker_setup.sh` to your broker Raspberry Pi
2. Make it executable:
   ```bash
   chmod +x broker_setup.sh
   ```
3. Run the script:
   ```bash
   ./broker_setup.sh
   ```

**Option B: Manual installation**

1. Install Mosquitto:
   ```bash
   sudo apt update
   sudo apt install mosquitto mosquitto-clients
   ```

2. Configure Mosquitto for network access:
   ```bash
   sudo nano /etc/mosquitto/mosquitto.conf
   ```
   
   Add or modify these lines:
   ```
   listener 1883 0.0.0.0
   allow_anonymous true
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl enable mosquitto
   sudo systemctl start mosquitto
   ```

4. If using a firewall, open port 1883:
   ```bash
   sudo ufw allow 1883/tcp
   ```

**Testing the broker:**

- Test locally:
  ```bash
  mosquitto_sub -h localhost -t test
  ```

- Test from another device:
  ```bash
  mosquitto_sub -h <broker-ip> -t test
  ```

- Find your broker's IP address:
  ```bash
  hostname -I
  ```

### 2. Subscriber Setup (Buzzer Controller)

1. Install Python dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

2. Configure the settings in `config.py`:
   - Set `MQTT_BROKER_HOST` to your broker Pi's IP address
   - Adjust `BUZZER_GPIO_PIN` if using a different GPIO pin (default: 18)
   - Adjust `BUZZER_DURATION` if needed (default: 2 seconds)

3. Run the subscriber:
   ```bash
   python3 buzzer_subscriber.py
   ```

### 3. Button Publisher Setup (Optional)

1. Install Python dependencies (if not already installed):
   ```bash
   pip3 install -r requirements.txt
   ```

2. Configure the settings in `config.py`:
   - Set `MQTT_BROKER_HOST` to your broker Pi's IP address
   - Adjust `BUTTON_GPIO_PIN` if using a different GPIO pin (default: 23)
   - Adjust `BUTTON_DEBOUNCE_TIME` if needed (default: 0.2 seconds)

3. Wire the button:
   - Connect one side of button to GPIO pin (default: GPIO 23)
   - Connect other side to GND
   - Use internal pull-up resistor (already configured in code)

4. Run the publisher:
   ```bash
   python3 button_publisher.py
   ```

## Configuration

Edit `config.py` to customize settings:

```python
# MQTT Broker Configuration
MQTT_BROKER_HOST = "192.168.1.100"  # Change to your broker Pi's IP
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "buzz"

# GPIO Pin Configuration
BUZZER_GPIO_PIN = 18  # GPIO pin for buzzer control
BUTTON_GPIO_PIN = 23  # GPIO pin for button input

# Timing Configuration
BUZZER_DURATION = 2  # Seconds to keep buzzer active
BUTTON_DEBOUNCE_TIME = 0.2  # Debounce time for button
```

## Usage

### Running the Subscriber

The subscriber runs continuously, listening for messages on the "buzz" topic:

```bash
python3 buzzer_subscriber.py
```

When a message is received, the buzzer will activate for 2 seconds (or configured duration).

Press `Ctrl+C` to stop.

### Running the Button Publisher

The button publisher monitors the GPIO button and publishes to "buzz" when pressed:

```bash
python3 button_publisher.py
```

Press the button to trigger a buzz message. The script includes debouncing to prevent multiple triggers.

Press `Ctrl+C` to stop.

### Testing the System

1. **Test with button publisher:**
   - Run `buzzer_subscriber.py` on one Pi
   - Run `button_publisher.py` on another Pi (or same Pi)
   - Press the button - buzzer should activate

2. **Test with command line:**
   ```bash
   mosquitto_pub -h <broker-ip> -t buzz -m "test"
   ```

3. **Test with WiFi doorbell:**
   - Configure doorbell to publish to topic "buzz"
   - Ensure broker IP is configured correctly
   - When doorbell triggers, buzzer should activate

## Troubleshooting

### Subscriber Issues

**"Failed to connect to broker"**
- Check broker IP address in `config.py`
- Verify broker is running: `sudo systemctl status mosquitto`
- Test connectivity: `ping <broker-ip>`
- Check firewall: `sudo ufw status`

**"RPi.GPIO library not found"**
- Install: `pip3 install RPi.GPIO`
- Ensure running on Raspberry Pi (not regular Linux)

**Buzzer doesn't activate**
- Check GPIO pin wiring
- Verify relay/transistor circuit is correct
- Test GPIO pin manually: `gpio -g write 18 1` (then `0` to turn off)
- Check logs for error messages

### Publisher Issues

**Button triggers multiple times**
- Increase `BUTTON_DEBOUNCE_TIME` in `config.py`
- Check button wiring (should use pull-up resistor)

**"Failed to connect to broker"**
- Same troubleshooting as subscriber (see above)

### Broker Issues

**Cannot connect from other devices**
- Check `mosquitto.conf` has `listener 1883 0.0.0.0`
- Verify firewall allows port 1883
- Check broker logs: `sudo journalctl -u mosquitto`

**Service won't start**
- Check configuration syntax: `sudo mosquitto -c /etc/mosquitto/mosquitto.conf -v`
- Review logs: `sudo journalctl -u mosquitto`

## GPIO Pin Reference

Common GPIO pins (BCM numbering):
- GPIO 18: PWM capable, good for buzzer
- GPIO 23: Standard GPIO, good for button

To see pin layout:
```bash
pinout
```

## Files

- `buzzer_subscriber.py` - Main subscriber script
- `button_publisher.py` - Button publisher script
- `config.py` - Configuration file
- `requirements.txt` - Python dependencies
- `broker_setup.sh` - Automated broker setup script
- `mosquitto.conf` - Example Mosquitto configuration
- `README.md` - This file

## Notes

- The buzzer requires a relay or transistor circuit since it's 12V and GPIO is 3.3V
- GPIO pins use BCM numbering (not physical pin numbers)
- Scripts run continuously until interrupted (Ctrl+C)
- Both scripts handle graceful shutdown and cleanup GPIO on exit
- For production use, consider adding MQTT authentication to the broker

## License

This project is provided as-is for educational and personal use.

