Code for each week has its own branch.

Please do not modify main unless it is final project implementation.

## Person and Package Detection System

A real-time object detection system using OpenCV and YOLOv8 nano to detect people and packages in video frames, with event signals when objects enter or leave the frame.

### Features

- Real-time person and package detection using YOLOv8 nano (optimized for Raspberry Pi)
- Event system for enter/leave notifications
- Webcam support (configurable for ESP32 camera)
- Visual feedback with bounding boxes and labels
- FPS display and detection counts

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. The YOLOv8 nano model will be automatically downloaded on first run (via ultralytics).

### Usage

Run the detection system:
```bash
python main.py
```

Press 'q' to quit.

### Configuration

#### Camera Source

**Webcam (default):**
- Uses camera index 0 in `main.py`
- Change `camera_index = 0` if needed

**ESP32 Camera:**
To use an ESP32 camera, modify `main.py`:

```python
# For ESP32 camera stream (example)
esp32_url = "http://ESP32_IP_ADDRESS:81/stream"
cap = cv2.VideoCapture(esp32_url)
```

Or use RTSP stream:
```python
rtsp_url = "rtsp://ESP32_IP_ADDRESS:554/stream"
cap = cv2.VideoCapture(rtsp_url)
```

#### Detection Parameters

In `main.py`, you can adjust:
- `model_size`: 'nano' (default, best for Pi), 'small', 'medium', 'large', 'xlarge'
- `confidence_threshold`: Minimum confidence for detections (default: 0.5)

### Event System

The system emits Python events when objects enter or leave the frame:

- `PERSON_ENTER` - Person detected entering frame
- `PERSON_LEAVE` - Person detected leaving frame
- `PACKAGE_ENTER` - Package detected entering frame
- `PACKAGE_LEAVE` - Package detected leaving frame

#### Custom Event Handlers

You can register custom callbacks in `main.py`:

```python
def my_custom_handler(event: DetectionEvent):
    print(f"Custom handler: {event}")
    # Your custom logic here

event_emitter.register_callback(EventType.PERSON_ENTER, my_custom_handler)
```

### Project Structure

- `detector.py` - Object detection using YOLOv8
- `events.py` - Event system for enter/leave notifications
- `main.py` - Main entry point with webcam integration
- `requirements.txt` - Python dependencies

### Performance Notes

- YOLOv8 nano is optimized for Raspberry Pi and should run at reasonable FPS
- For better performance, reduce frame resolution or increase confidence threshold
- The system uses IoU-based tracking to maintain object identity across frames