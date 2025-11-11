"""
Object detection using YOLOv8 nano for person and package detection.
"""
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from events import DetectionEventEmitter, DetectionEvent, EventType


class ObjectDetector:
    """Detects people and packages in video frames using YOLOv8."""
    
    # COCO class IDs
    PERSON_CLASS_ID = 0
    # Package-related classes in COCO dataset
    PACKAGE_CLASS_IDS = [24, 26, 28]  # backpack, handbag, suitcase
    
    # Additional COCO classes that could be packages/boxes
    # These are objects that are commonly box-shaped or could contain packages
    POTENTIAL_PACKAGE_CLASS_IDS = [
        24,  # backpack
        26,  # handbag
        28,  # suitcase
        39,  # bottle (could be in a box)
        40,  # wine glass (could be packaged)
        67,  # cell phone (often in boxes)
        73,  # laptop (often in boxes)
        77,  # mouse (often in boxes)
    ]
    
    # Classes to exclude from package detection (definitely not packages)
    EXCLUDED_CLASS_IDS = [0]  # person
    
    def __init__(self, 
                 model_size: str = 'nano',
                 confidence_threshold: float = 0.5,
                 event_emitter: Optional[DetectionEventEmitter] = None,
                 debug_mode: bool = False):
        """Initialize the object detector.
        
        Args:
            model_size: YOLOv8 model size ('nano', 'small', 'medium', 'large', 'xlarge')
            confidence_threshold: Minimum confidence for detections
            event_emitter: Optional event emitter for enter/leave events
            debug_mode: If True, show all detections and debug info
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.event_emitter = event_emitter or DetectionEventEmitter()
        self.debug_mode = debug_mode
        
        # Map model size to YOLOv8 model file names
        model_size_map = {
            'nano': 'n',
            'small': 's',
            'medium': 'm',
            'large': 'l',
            'xlarge': 'x'
        }
        
        # Load YOLOv8 model
        model_suffix = model_size_map.get(model_size.lower(), 'n')  # Default to nano
        model_name = f'yolov8{model_suffix}.pt'
        print(f"Loading YOLOv8 {model_size} model ({model_name})...")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
        # Track objects across frames
        # Format: {object_id: {'type': 'person'|'package', 'bbox': (x1,y1,x2,y2), 'frame_count': int}}
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_object_id = 0
        
        # Previous frame detections for comparison
        self.previous_detections: Dict[int, Tuple[str, Tuple]] = {}  # {id: (type, bbox)}
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _is_package(self, class_id: int, class_name: str, bbox: Tuple[float, float, float, float] = None) -> bool:
        """Determine if a detection is a package.
        
        Args:
            class_id: COCO class ID
            class_name: COCO class name
            bbox: Optional bounding box (x1, y1, x2, y2) for shape-based detection
            
        Returns:
            True if the object is considered a package
        """
        # Exclude person
        if class_id in self.EXCLUDED_CLASS_IDS:
            return False
        
        # Check if it's in our package class IDs
        if class_id in self.PACKAGE_CLASS_IDS:
            return True
        
        # Check class name for package-related keywords
        package_keywords = ['box', 'package', 'suitcase', 'bag', 'backpack', 'handbag', 'container']
        class_name_lower = class_name.lower()
        if any(keyword in class_name_lower for keyword in package_keywords):
            return True
        
        # Shape-based detection: check if object has box-like aspect ratio
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                area = width * height
                
                # Box-like objects typically have:
                # - Aspect ratio between 1.0 and 3.0 (roughly rectangular)
                # - Reasonable size (not too small)
                # - Not too elongated (like a person)
                if 1.0 <= aspect_ratio <= 3.5 and area > 1000:  # Minimum area threshold
                    # Additional check: if it's a medium to large rectangular object
                    # that's not a person, it could be a package
                    if area > 5000:  # Large enough to be a package
                        return True
        
        return False
    
    def _track_objects(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Track objects across frames using IoU matching.
        
        Args:
            detections: List of detection dicts with 'type', 'bbox', 'confidence'
            
        Returns:
            Dictionary mapping object IDs to detection info
        """
        current_detections: Dict[int, Dict] = {}
        
        # Match current detections with previous ones
        matched_previous = set()
        
        for det in detections:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold for matching
            
            # Try to match with previous detections
            for prev_id, (prev_type, prev_bbox) in self.previous_detections.items():
                if prev_id in matched_previous:
                    continue
                
                # Only match same type
                if prev_type != det['type']:
                    continue
                
                iou = self._calculate_iou(det['bbox'], prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = prev_id
            
            # Assign ID
            if best_match_id is not None:
                det_id = best_match_id
                matched_previous.add(best_match_id)
            else:
                # New object
                det_id = self.next_object_id
                self.next_object_id += 1
            
            current_detections[det_id] = det
        
        return current_detections
    
    def _detect_enter_leave(self, current_detections: Dict[int, Dict]):
        """Detect enter/leave events by comparing with previous frame.
        
        Args:
            current_detections: Current frame detections with IDs
        """
        current_ids = set(current_detections.keys())
        previous_ids = set(self.previous_detections.keys())
        
        # Objects that entered (in current but not in previous)
        entered_ids = current_ids - previous_ids
        for obj_id in entered_ids:
            det = current_detections[obj_id]
            event_type = EventType.PERSON_ENTER if det['type'] == 'person' else EventType.PACKAGE_ENTER
            event = DetectionEvent(
                event_type=event_type,
                object_type=det['type'],
                timestamp=det.get('timestamp'),
                bounding_box=det['bbox'],
                confidence=det['confidence']
            )
            self.event_emitter.emit(event)
        
        # Objects that left (in previous but not in current)
        left_ids = previous_ids - current_ids
        for obj_id in left_ids:
            prev_type, prev_bbox = self.previous_detections[obj_id]
            event_type = EventType.PERSON_LEAVE if prev_type == 'person' else EventType.PACKAGE_LEAVE
            event = DetectionEvent(
                event_type=event_type,
                object_type=prev_type,
                timestamp=datetime.now(),
                bounding_box=prev_bbox,
                confidence=0.0  # We don't have confidence for objects that left
            )
            self.event_emitter.emit(event)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame and detect objects.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, detections_list)
            detections_list contains dicts with 'type', 'bbox', 'confidence', 'class_id'
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Get bounding box for shape analysis
                bbox_coords = (float(x1), float(y1), float(x2), float(y2))
                
                # Determine object type
                if class_id == self.PERSON_CLASS_ID:
                    obj_type = 'person'
                elif self._is_package(class_id, class_name, bbox_coords):
                    obj_type = 'package'
                else:
                    # In debug mode, show what YOLO detected but we're skipping
                    if self.debug_mode:
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                        print(f"[DEBUG] Skipped: {class_name} (id:{class_id}) conf:{confidence:.2f} "
                              f"area:{area:.0f} aspect:{aspect_ratio:.2f}")
                    continue  # Skip other objects
                
                detections.append({
                    'type': obj_type,
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'timestamp': datetime.now()
                })
                
                # Draw bounding box
                color = (0, 255, 0) if obj_type == 'person' else (255, 0, 0)
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 2)
                
                # Draw label
                label = f"{obj_type} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)),
                            color, -1)
                cv2.putText(annotated_frame, label,
                          (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Track objects and detect enter/leave events
        current_detections = self._track_objects(detections)
        self._detect_enter_leave(current_detections)
        
        # Update previous detections
        self.previous_detections = {
            obj_id: (det['type'], det['bbox'])
            for obj_id, det in current_detections.items()
        }
        
        return annotated_frame, detections

