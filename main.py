"""
Main entry point for person and package detection system.
"""
import cv2
import time
from detector import ObjectDetector
from events import DetectionEventEmitter, EventType, DetectionEvent


def on_person_enter(event: DetectionEvent):
    """Callback for person enter events."""
    print(f"[EVENT] Person entered frame: {event}")


def on_person_leave(event: DetectionEvent):
    """Callback for person leave events."""
    print(f"[EVENT] Person left frame: {event}")


def on_package_enter(event: DetectionEvent):
    """Callback for package enter events."""
    print(f"[EVENT] Package entered frame: {event}")


def on_package_leave(event: DetectionEvent):
    """Callback for package leave events."""
    print(f"[EVENT] Package left frame: {event}")


def on_any_event(event: DetectionEvent):
    """Callback for all events."""
    print(f"[ALL EVENTS] {event}")


def test_camera(camera_index=0):
    """Test if a camera can be opened and read from."""
    print(f"\nTesting camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"  Camera {camera_index}: Cannot open")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        print(f"  Camera {camera_index}: Working! Frame size: {frame.shape}")
        return True
    else:
        print(f"  Camera {camera_index}: Opens but cannot read frames")
        return False


def main():
    """Main function to run the detection system."""
    # Initialize event emitter
    event_emitter = DetectionEventEmitter()
    
    # Register event callbacks
    event_emitter.register_callback(EventType.PERSON_ENTER, on_person_enter)
    event_emitter.register_callback(EventType.PERSON_LEAVE, on_person_leave)
    event_emitter.register_callback(EventType.PACKAGE_ENTER, on_package_enter)
    event_emitter.register_callback(EventType.PACKAGE_LEAVE, on_package_leave)
    
    # Optional: register a callback for all events
    # event_emitter.register_all_events_callback(on_any_event)
    
    # Initialize detector
    print("Initializing detector...")
    detector = ObjectDetector(
        model_size='nano',  # Use 'nano' for Raspberry Pi
        confidence_threshold=0.3,  # Lower threshold to catch more detections
        event_emitter=event_emitter,
        debug_mode=True  # Set to False to disable debug output
    )
    
    # Initialize webcam
    # Try multiple camera indices and backends to find a working camera
    camera_indices = [0, 1, 2]  # Try first 3 cameras
    
    # Try different backends (order matters - try most reliable first)
    backends = [
        (cv2.CAP_ANY, "Default"),       # Default backend (tries best available)
    ]
    
    cap = None
    working_camera = None
    
    for camera_index in camera_indices:
        for backend_id, backend_name in backends:
            print(f"Trying camera {camera_index} with {backend_name} backend...")
            try:
                if backend_id == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(camera_index)
                else:
                    cap = cv2.VideoCapture(camera_index, backend_id)
                
                if cap.isOpened():
                    # Try to read a few frames to ensure it's working
                    success_count = 0
                    for _ in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            success_count += 1
                            time.sleep(0.1)  # Small delay between test reads
                    
                    if success_count > 0:
                        print(f"Successfully opened camera {camera_index} with {backend_name} backend!")
                        working_camera = camera_index
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                print(f"  Error trying camera {camera_index}: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap is not None and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print("\nError: Could not open any camera")
        print("\nRunning camera diagnostics...")
        for idx in camera_indices:
            test_camera(idx)
        
        print("\nTroubleshooting tips:")
        print("  1. Make sure no other application is using the camera (Zoom, Teams, etc.)")
        print("  2. Test your camera with Windows Camera app first")
        print("  3. Check Windows Camera privacy settings:")
        print("     Settings → Privacy & Security → Camera → Allow apps to access your camera")
        print("  4. Try running as administrator")
        print("  5. Restart your computer if the camera was recently used by another app")
        print("  6. If using a USB camera, try unplugging and replugging it")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Give the camera a moment to initialize
    time.sleep(0.5)
    
    print("Starting detection... Press 'q' to quit")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    try:
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Error: Failed to read frame {max_failures} times in a row")
                    break
                time.sleep(0.1)  # Brief pause before retry
                continue
            
            consecutive_failures = 0  # Reset counter on success
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display FPS and detection count
            person_count = sum(1 for d in detections if d['type'] == 'person')
            package_count = sum(1 for d in detections if d['type'] == 'package')
            
            info_text = f"FPS: {fps:.1f} | Persons: {person_count} | Packages: {package_count}"
            cv2.putText(annotated_frame, info_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Person & Package Detection', annotated_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


if __name__ == "__main__":
    main()

