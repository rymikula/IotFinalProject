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
                 debug_mode: bool = False,
                 min_frames_for_confirmation: int = 8,
                 package_zone_y_percent: float = 0.6,
                 shape_confidence_threshold: float = 0.2,  # Lowered from 0.55 for more lenient detection
                 background_learning_frames: int = 100,
                 background_threshold: float = 0.15,
                 foreground_overlap_threshold: float = 0.8,  # Very lenient - only filter obvious background
                 static_background_threshold: int = 30):
        """Initialize the object detector.
        
        Args:
            model_size: YOLOv8 model size ('nano', 'small', 'medium', 'large', 'xlarge')
            confidence_threshold: Minimum confidence for COCO package class detections
            event_emitter: Optional event emitter for enter/leave events
            debug_mode: If True, show all detections and debug info
            min_frames_for_confirmation: Minimum frames a package must be detected before confirming
            package_zone_y_percent: Percentage of frame height where packages typically appear (0.0-1.0)
            shape_confidence_threshold: Minimum confidence for shape-based package detection
            background_learning_frames: Number of frames to use for background learning
            background_threshold: Threshold for background subtractor (0.0-1.0)
            foreground_overlap_threshold: Minimum percentage of bbox that must be foreground (0.0-1.0)
            static_background_threshold: Threshold for static background frame differencing (0-255)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.shape_confidence_threshold = shape_confidence_threshold
        self.event_emitter = event_emitter or DetectionEventEmitter()
        self.debug_mode = debug_mode
        self.min_frames_for_confirmation = min_frames_for_confirmation
        self.package_zone_y_percent = package_zone_y_percent
        self.background_learning_frames = background_learning_frames
        self.background_threshold = background_threshold
        self.foreground_overlap_threshold = foreground_overlap_threshold
        self.static_background_threshold = static_background_threshold
        
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
        # Format: {object_id: {'type': 'person'|'package', 'bbox': (x1,y1,x2,y2), 'frame_count': int, 
        #          'detection_count': int, 'positions': [(x,y), ...], 'velocities': [float, ...]}}
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_object_id = 0
        
        # Previous frame detections for comparison
        # Format: {id: {'type': str, 'bbox': tuple, 'center': tuple, 'timestamp': datetime}}
        self.previous_detections: Dict[int, Dict] = {}
        
        # Confirmed packages (detected consistently across multiple frames)
        # Format: {object_id: {'bbox': tuple, 'detection_count': int, 'first_seen': datetime, 
        #          'last_seen': datetime, 'avg_velocity': float}}
        self.confirmed_packages: Dict[int, Dict] = {}
        
        # Track frame number for temporal analysis
        self.frame_number = 0
        
        # Background subtraction setup
        # Use MOG2 (Mixture of Gaussians) for adaptive background subtraction during learning
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=background_learning_frames,
            varThreshold=50,
            detectShadows=True
        )
        self.background_learned = False
        self.background_learning_frame_count = 0
        self.reference_background_frame = None
        self.static_background_frame = None  # Static background after learning completes
        self.use_static_background = True  # Use static background after learning to prevent packages from being absorbed
        self.background_accumulator = None  # For computing running average during learning
    
    def _calculate_bbox_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate center point of a bounding box.
        
        Args:
            bbox: (x1, y1, x2, y2)
            
        Returns:
            (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            point1: (x1, y1)
            point2: (x2, y2)
            
        Returns:
            Distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _detect_box_edges(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        """Detect box-like edges in a region of the frame with enhanced square/rectangular detection.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Edge score (0.0-1.0) indicating how box-like the region is
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract region of interest
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to better detect edges
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Also use Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Combine both edge detection methods
        combined_edges = cv2.bitwise_or(edges, thresh)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour (likely the box)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        roi_area = roi.shape[0] * roi.shape[1]
        
        if roi_area == 0:
            return 0.0
        
        # Check if contour covers a reasonable portion of the ROI
        coverage = contour_area / roi_area
        if coverage < 0.3:  # Contour should cover at least 30% of ROI
            return 0.0
        
        # Approximate contour to polygon with tighter epsilon for better corner detection
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Calculate bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box_width, box_height = rect[1]
        if box_width == 0 or box_height == 0:
            return 0.0
        
        # Calculate aspect ratio of the detected shape
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        
        # Score based on corner count (4 corners = rectangle/square)
        corner_count = len(approx)
        corner_score = 0.0
        if corner_count == 4:
            corner_score = 1.0  # Perfect rectangle
        elif corner_count == 3 or corner_count == 5:
            corner_score = 0.6  # Close to rectangle
        elif corner_count >= 6:
            corner_score = 0.3  # Too many corners (likely not a box)
        else:
            corner_score = 0.1  # Too few corners
        
        # Score based on how rectangular the shape is
        # Calculate how well the contour fits its bounding rectangle
        rect_area = box_width * box_height
        if rect_area > 0:
            extent = contour_area / rect_area  # How much of the rect is filled
            # Good rectangles have extent close to 1.0
            extent_score = min(1.0, extent * 1.2)  # Slightly favor higher extent
        else:
            extent_score = 0.0
        
        # Score based on aspect ratio (square-ish is better, but rectangles are fine)
        if 1.0 <= aspect_ratio <= 1.5:
            aspect_score = 1.0  # Square to slightly rectangular
        elif 1.5 < aspect_ratio <= 2.0:
            aspect_score = 0.9  # Rectangular
        elif 2.0 < aspect_ratio <= 2.5:
            aspect_score = 0.7  # Longer rectangle
        elif 2.5 < aspect_ratio <= 3.0:
            aspect_score = 0.5  # Very long rectangle
        else:
            aspect_score = 0.2  # Too elongated
        
        # Calculate edge density (how much of ROI has edges)
        edge_density = np.sum(combined_edges > 0) / (roi.shape[0] * roi.shape[1])
        edge_score = min(1.0, edge_density * 3.0)  # Normalize
        
        # Combine scores with weights favoring square/rectangular shapes
        # Corner count is most important, then extent, then aspect ratio, then edge density
        final_score = (corner_score * 0.4) + (extent_score * 0.3) + (aspect_score * 0.2) + (edge_score * 0.1)
        
        return min(1.0, final_score)
    
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
    
    def _is_package(self, class_id: int, class_name: str, bbox: Tuple[float, float, float, float] = None, 
                   frame: np.ndarray = None, confidence: float = 0.0) -> Tuple[bool, float]:
        """Determine if a detection is a package with enhanced heuristics.
        
        Args:
            class_id: COCO class ID
            class_name: COCO class name
            bbox: Optional bounding box (x1, y1, x2, y2) for shape-based detection
            frame: Optional frame for edge detection
            confidence: YOLO detection confidence
            
        Returns:
            Tuple of (is_package: bool, package_score: float)
            package_score is a combined confidence score (0.0-1.0)
        """
        # Exclude person
        if class_id in self.EXCLUDED_CLASS_IDS:
            return False, 0.0
        
        package_score = 0.0
        
        # Check if it's in our package class IDs (backpack, handbag, suitcase)
        if class_id in self.PACKAGE_CLASS_IDS:
            # Known package classes get base score from confidence
            package_score = max(package_score, confidence * 0.9)
            # Very lenient threshold for known package classes - accept almost anything
            if confidence >= self.confidence_threshold * 0.3:  # 70% more lenient (was 0.5)
                return True, package_score
        
        # Check class name for package-related keywords
        package_keywords = ['box', 'package', 'suitcase', 'bag', 'backpack', 'handbag', 'container', 
                           'carton', 'parcel', 'boxed', 'packaged', 'delivery', 'mail']
        class_name_lower = class_name.lower()
        if any(keyword in class_name_lower for keyword in package_keywords):
            package_score = max(package_score, confidence * 0.8)
            # Very lenient threshold for keyword matches
            if confidence >= self.confidence_threshold * 0.4:  # 60% more lenient (was 0.6)
                return True, package_score
        
        # Enhanced shape-based detection for delivery boxes/packages
        if bbox is not None and frame is not None:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            frame_height, frame_width = frame.shape[:2]
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                area = width * height
                frame_area = frame_width * frame_height
                area_ratio = area / frame_area if frame_area > 0 else 0
                
                # Position filtering: packages typically appear in lower portion of frame (ground level)
                center_y = (y1 + y2) / 2
                y_position_ratio = center_y / frame_height if frame_height > 0 else 0
                in_package_zone = y_position_ratio >= (1.0 - self.package_zone_y_percent)
                
                # Size filtering: exclude objects that are too small or too large (very lenient)
                min_area_ratio = 0.002  # 0.2% of frame (even more lenient)
                max_area_ratio = 0.50   # 50% of frame (more lenient)
                
                # Aspect ratio filtering: packages are typically 1.0-4.0 (very lenient)
                valid_aspect_ratio = 1.0 <= aspect_ratio <= 4.0  # Expanded from 3.0
                
                # Size check: reasonable size for a package
                valid_size = min_area_ratio <= area_ratio <= max_area_ratio
                
                # Simple fallback: if it's a reasonable rectangular object, consider it a package
                # This catches cases where edge detection might fail
                if valid_aspect_ratio and valid_size:
                    # Give base score just for being rectangular and reasonable size
                    base_rectangular_score = 0.4  # Increased from 0.3
                    
                    # Add score based on aspect ratio (square-ish is better)
                    if 1.0 <= aspect_ratio <= 2.0:
                        base_rectangular_score += 0.3  # Increased from 0.2
                    elif 2.0 < aspect_ratio <= 3.0:
                        base_rectangular_score += 0.2  # Increased from 0.1
                    elif 3.0 < aspect_ratio <= 4.0:
                        base_rectangular_score += 0.1  # New: allow very long rectangles
                    
                    # Add score for being in package zone
                    if in_package_zone:
                        base_rectangular_score += 0.2
                    else:
                        base_rectangular_score += 0.15  # Increased from 0.1
                    
                    # Combine with confidence - weight confidence heavily
                    if confidence > 0:
                        simple_score = (base_rectangular_score * 0.3) + (confidence * 0.7)  # More weight on confidence
                    else:
                        simple_score = base_rectangular_score
                    
                    # Very lenient threshold for simple detection - accept almost anything rectangular
                    if simple_score >= self.shape_confidence_threshold * 0.3:  # Even more lenient (was 0.5)
                        package_score = max(package_score, simple_score)
                        return True, package_score
                
                # Shape-based package detection (prioritize square/rectangular shapes)
                if valid_aspect_ratio and valid_size:
                    shape_score = 0.0
                    
                    # Base score from aspect ratio (square shapes get highest score)
                    if 1.0 <= aspect_ratio <= 1.3:
                        shape_score += 0.35  # Reduced from 0.45
                    elif 1.3 < aspect_ratio <= 1.6:
                        shape_score += 0.30  # Reduced from 0.40
                    elif 1.6 < aspect_ratio <= 2.0:
                        shape_score += 0.25  # Reduced from 0.30
                    elif 2.0 < aspect_ratio <= 2.5:
                        shape_score += 0.20  # Reduced from 0.20
                    elif 2.5 < aspect_ratio <= 3.0:
                        shape_score += 0.15  # Increased from 0.10
                    elif 3.0 < aspect_ratio <= 4.0:
                        shape_score += 0.10  # New: allow very long rectangles
                    
                    # Position score (packages on ground, but more lenient)
                    if in_package_zone:
                        shape_score += 0.25  # Reduced from 0.30
                    else:
                        shape_score += 0.20  # Increased from 0.15
                    
                    # Size score (medium-sized objects are more likely packages, more lenient)
                    if 0.01 <= area_ratio <= 0.15:  # 1-15% of frame
                        shape_score += 0.15  # Reduced from 0.20
                    elif 0.005 <= area_ratio < 0.01 or 0.15 < area_ratio <= 0.30:
                        shape_score += 0.10  # Same
                    elif 0.002 <= area_ratio < 0.005 or 0.30 < area_ratio <= 0.50:
                        shape_score += 0.05  # Expanded range
                    
                    # Edge detection score (box-like edges) - but don't require it
                    edge_score = self._detect_box_edges(frame, bbox)
                    shape_score += edge_score * 0.20  # Reduced from 0.40 - don't rely too much on edges
                    
                    # Combine with YOLO confidence if available (weight confidence more)
                    if confidence > 0:
                        combined_score = (shape_score * 0.40) + (confidence * 0.60)  # More weight on confidence
                    else:
                        combined_score = shape_score
                    
                    # Require minimum score for shape-based detection (very lenient threshold)
                    # Lower threshold by 70% for very lenient detection (was 50%)
                    adjusted_threshold = self.shape_confidence_threshold * 0.3
                    if combined_score >= adjusted_threshold:
                        package_score = max(package_score, combined_score)
                        return True, package_score
                
                # Ultra-simple fallback: accept ANY reasonable-sized rectangular object
                # This is a last resort to catch packages that pass all other checks
                if valid_aspect_ratio and valid_size and confidence > 0:
                    # Just check if it's rectangular and has some confidence
                    ultra_simple_score = confidence * 0.5  # Just use confidence
                    if ultra_simple_score >= self.shape_confidence_threshold * 0.2:  # Very low threshold
                        package_score = max(package_score, ultra_simple_score)
                        return True, package_score
        
        return False, package_score
    
    def _track_objects(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Track objects across frames using IoU matching with velocity tracking.
        
        Args:
            detections: List of detection dicts with 'type', 'bbox', 'confidence', 'package_score'
            
        Returns:
            Dictionary mapping object IDs to detection info with tracking data
        """
        current_detections: Dict[int, Dict] = {}
        
        # Match current detections with previous ones
        matched_previous = set()
        
        for det in detections:
            best_match_id = None
            best_iou = 0.4  # Improved IoU threshold for matching
            best_velocity = None
            
            # Calculate center of current detection
            current_center = self._calculate_bbox_center(det['bbox'])
            
            # Try to match with previous detections
            for prev_id, prev_data in self.previous_detections.items():
                if prev_id in matched_previous:
                    continue
                
                # Only match same type
                if prev_data['type'] != det['type']:
                    continue
                
                prev_bbox = prev_data['bbox']
                prev_center = prev_data.get('center', self._calculate_bbox_center(prev_bbox))
                
                iou = self._calculate_iou(det['bbox'], prev_bbox)
                
                # Calculate velocity (distance moved per frame)
                distance = self._calculate_distance(current_center, prev_center)
                
                # Prefer matches with high IoU and low velocity (stationary objects)
                # Packages should be relatively stationary
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = prev_id
                    best_velocity = distance
            
            # Assign ID
            if best_match_id is not None:
                det_id = best_match_id
                matched_previous.add(best_match_id)
                
                # Update tracking data
                if det_id in self.tracked_objects:
                    # Update existing tracked object
                    tracked = self.tracked_objects[det_id]
                    tracked['detection_count'] = tracked.get('detection_count', 0) + 1
                    tracked['last_seen'] = det.get('timestamp', datetime.now())
                    
                    # Track velocity
                    if 'velocities' not in tracked:
                        tracked['velocities'] = []
                    if best_velocity is not None:
                        tracked['velocities'].append(best_velocity)
                        # Keep only last 10 velocities
                        if len(tracked['velocities']) > 10:
                            tracked['velocities'] = tracked['velocities'][-10:]
                    
                    # Track positions
                    if 'positions' not in tracked:
                        tracked['positions'] = []
                    tracked['positions'].append(current_center)
                    if len(tracked['positions']) > 20:
                        tracked['positions'] = tracked['positions'][-20:]
                    
                    # Calculate average velocity
                    if tracked['velocities']:
                        tracked['avg_velocity'] = np.mean(tracked['velocities'])
                    else:
                        tracked['avg_velocity'] = 0.0
                else:
                    # Initialize tracking for matched object
                    self.tracked_objects[det_id] = {
                        'type': det['type'],
                        'detection_count': 1,
                        'first_seen': det.get('timestamp', datetime.now()),
                        'last_seen': det.get('timestamp', datetime.now()),
                        'velocities': [best_velocity] if best_velocity is not None else [],
                        'positions': [current_center],
                        'avg_velocity': best_velocity if best_velocity is not None else 0.0
                    }
            else:
                # New object
                det_id = self.next_object_id
                self.next_object_id += 1
                
                # Initialize tracking for new object
                self.tracked_objects[det_id] = {
                    'type': det['type'],
                    'detection_count': 1,
                    'first_seen': det.get('timestamp', datetime.now()),
                    'last_seen': det.get('timestamp', datetime.now()),
                    'velocities': [],
                    'positions': [current_center],
                    'avg_velocity': 0.0
                }
            
            # Add tracking info to detection
            det['object_id'] = det_id
            det['center'] = current_center
            if det_id in self.tracked_objects:
                det['detection_count'] = self.tracked_objects[det_id]['detection_count']
                det['avg_velocity'] = self.tracked_objects[det_id].get('avg_velocity', 0.0)
            
            current_detections[det_id] = det
        
        return current_detections
    
    def _update_confirmed_packages(self, current_detections: Dict[int, Dict], frame: np.ndarray = None):
        """Update confirmed packages based on temporal consistency and stationary detection.
        
        Args:
            current_detections: Current frame detections with IDs
            frame: Optional frame for package zone checking
        """
        # Velocity threshold for stationary objects (packages don't move much)
        stationary_velocity_threshold = 5.0  # pixels per frame
        
        # Check packages in current detections
        current_package_ids = {obj_id for obj_id, det in current_detections.items() 
                              if det['type'] == 'package'}
        
        # Update confirmed packages
        for obj_id in current_package_ids:
            det = current_detections[obj_id]
            detection_count = det.get('detection_count', 1)
            avg_velocity = det.get('avg_velocity', 0.0)
            bbox = det['bbox']
            
            # Check if package is in the package zone
            in_package_zone = True
            if frame is not None:
                x1, y1, x2, y2 = bbox
                frame_height = frame.shape[0]
                center_y = (y1 + y2) / 2
                y_position_ratio = center_y / frame_height if frame_height > 0 else 0
                in_package_zone = y_position_ratio >= (1.0 - self.package_zone_y_percent)
            
            # Check if package should be confirmed - more lenient requirements
            # Lower the frame requirement and velocity threshold
            min_frames = max(3, self.min_frames_for_confirmation // 2)  # Half the required frames
            if detection_count >= min_frames:
                # Package must be detected consistently, be relatively stationary, AND be in package zone
                # But be more lenient with velocity (packages might move slightly)
                if avg_velocity <= stationary_velocity_threshold * 2.0 and in_package_zone:  # Double velocity threshold
                    if obj_id not in self.confirmed_packages:
                        # New confirmed package - emit enter event
                        self.confirmed_packages[obj_id] = {
                            'bbox': bbox,
                            'detection_count': detection_count,
                            'first_seen': det.get('timestamp', datetime.now()),
                            'last_seen': det.get('timestamp', datetime.now()),
                            'avg_velocity': avg_velocity,
                            'package_score': det.get('package_score', det.get('confidence', 0.0)),
                            'event_emitted': False
                        }
                        if self.debug_mode:
                            print(f"[DEBUG] Package {obj_id} confirmed after {detection_count} frames "
                                  f"(velocity: {avg_velocity:.2f}, in_zone: {in_package_zone})")
                        
                        # Emit PACKAGE_ENTER event for newly confirmed package
                        event = DetectionEvent(
                            event_type=EventType.PACKAGE_ENTER,
                            object_type='package',
                            timestamp=self.confirmed_packages[obj_id]['first_seen'],
                            bounding_box=bbox,
                            confidence=self.confirmed_packages[obj_id]['package_score']
                        )
                        self.event_emitter.emit(event)
                        self.confirmed_packages[obj_id]['event_emitted'] = True
                        if self.debug_mode:
                            print(f"[DEBUG] Emitted PACKAGE_ENTER for newly confirmed package {obj_id}")
                    else:
                        # Update existing confirmed package (only if still in zone)
                        if in_package_zone:
                            self.confirmed_packages[obj_id]['last_seen'] = det.get('timestamp', datetime.now())
                            self.confirmed_packages[obj_id]['bbox'] = bbox
                            self.confirmed_packages[obj_id]['detection_count'] = detection_count
                        else:
                            # Package moved out of zone - remove it
                            if self.confirmed_packages[obj_id].get('event_emitted', False):
                                event = DetectionEvent(
                                    event_type=EventType.PACKAGE_LEAVE,
                                    object_type='package',
                                    timestamp=datetime.now(),
                                    bounding_box=bbox,
                                    confidence=0.0
                                )
                                self.event_emitter.emit(event)
                                if self.debug_mode:
                                    print(f"[DEBUG] Package {obj_id} left package zone - removed")
                            current_package_ids.discard(obj_id)  # Don't process as current package
                            if obj_id in self.confirmed_packages:
                                del self.confirmed_packages[obj_id]
        
        # Remove packages that are no longer detected (after grace period)
        confirmed_ids_to_remove = []
        for obj_id in self.confirmed_packages.keys():
            if obj_id not in current_package_ids:
                # Package disappeared - emit leave event and remove
                if self.confirmed_packages[obj_id].get('event_emitted', False):
                    event = DetectionEvent(
                        event_type=EventType.PACKAGE_LEAVE,
                        object_type='package',
                        timestamp=datetime.now(),
                        bounding_box=self.confirmed_packages[obj_id]['bbox'],
                        confidence=0.0
                    )
                    self.event_emitter.emit(event)
                    if self.debug_mode:
                        print(f"[DEBUG] Emitted PACKAGE_LEAVE for confirmed package {obj_id}")
                confirmed_ids_to_remove.append(obj_id)
        
        for obj_id in confirmed_ids_to_remove:
            del self.confirmed_packages[obj_id]
    
    def _learn_background(self, frame: np.ndarray):
        """Learn the background model from frames during initialization.
        
        Args:
            frame: Input frame (BGR format)
        """
        if self.background_learned:
            return
        
        # Process frame through background subtractor
        self.background_subtractor.apply(frame)
        self.background_learning_frame_count += 1
        
        # Accumulate frames for running average (more reliable than getBackgroundImage)
        if self.use_static_background:
            if self.background_accumulator is None:
                self.background_accumulator = frame.copy().astype(np.float32)
            else:
                # Running average: new_avg = (old_avg * (n-1) + new_frame) / n
                alpha = 1.0 / self.background_learning_frame_count
                self.background_accumulator = (1.0 - alpha) * self.background_accumulator + alpha * frame.astype(np.float32)
        
        # Store reference background frame for visualization
        if self.reference_background_frame is None:
            self.reference_background_frame = frame.copy()
        
        # Check if learning period is complete
        if self.background_learning_frame_count >= self.background_learning_frames:
            self.background_learned = True
            
            # Extract static background frame
            if self.use_static_background:
                # Use accumulated running average (more reliable)
                if self.background_accumulator is not None:
                    self.static_background_frame = self.background_accumulator.astype(np.uint8)
                else:
                    # Fallback: try to get background from MOG2
                    bg_model = self.background_subtractor.getBackgroundImage()
                    if bg_model is not None:
                        self.static_background_frame = bg_model.copy()
                    else:
                        # Final fallback: use current frame
                        self.static_background_frame = frame.copy()
                
                if self.debug_mode:
                    print(f"[DEBUG] Background learning complete after {self.background_learning_frame_count} frames")
                    print(f"[DEBUG] Using static background to prevent packages from being absorbed")
            else:
                if self.debug_mode:
                    print(f"[DEBUG] Background learning complete after {self.background_learning_frame_count} frames")
    
    def _get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get foreground mask from background subtraction.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Binary mask where foreground pixels are white (255) and background is black (0)
        """
        if not self.background_learned:
            # Return empty mask if background not learned yet
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        # Use static background if enabled (prevents packages from being absorbed)
        if self.use_static_background and self.static_background_frame is not None:
            # Convert frames to grayscale for comparison
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(self.static_background_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(frame_gray, bg_gray)
            
            # Threshold to get foreground mask
            _, fg_mask = cv2.threshold(diff, self.static_background_threshold, 255, cv2.THRESH_BINARY)
        else:
            # Use adaptive MOG2 (will continue learning and absorb stationary objects)
            fg_mask = self.background_subtractor.apply(frame)
            # Remove shadow pixels (MOG2 marks shadows as 127, we want only foreground as 255)
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        return fg_mask
    
    def _check_foreground_overlap(self, bbox: Tuple[float, float, float, float], 
                                  foreground_mask: np.ndarray) -> float:
        """Check how much of a bounding box overlaps with foreground regions.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            foreground_mask: Binary foreground mask
            
        Returns:
            Percentage of bbox area that is foreground (0.0-1.0)
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within frame bounds
        h, w = foreground_mask.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract bounding box region from foreground mask
        bbox_region = foreground_mask[y1:y2, x1:x2]
        
        if bbox_region.size == 0:
            return 0.0
        
        # Calculate percentage of foreground pixels
        foreground_pixels = np.sum(bbox_region > 0)
        total_pixels = bbox_region.size
        overlap_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return overlap_ratio
    
    def _detect_enter_leave(self, current_detections: Dict[int, Dict]):
        """Detect enter/leave events for persons only.
        Package events are handled in _update_confirmed_packages.
        
        Args:
            current_detections: Current frame detections with IDs
        """
        current_ids = set(current_detections.keys())
        previous_ids = set(self.previous_detections.keys())
        
        # Objects that entered (in current but not in previous)
        entered_ids = current_ids - previous_ids
        for obj_id in entered_ids:
            det = current_detections[obj_id]
            
            # Only handle person events here (package events handled in _update_confirmed_packages)
            if det['type'] == 'person':
                event_type = EventType.PERSON_ENTER
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
            prev_data = self.previous_detections[obj_id]
            prev_type = prev_data['type']
            prev_bbox = prev_data['bbox']
            
            # Only handle person events here (package events handled in _update_confirmed_packages)
            if prev_type == 'person':
                event_type = EventType.PERSON_LEAVE
                event = DetectionEvent(
                    event_type=event_type,
                    object_type=prev_type,
                    timestamp=datetime.now(),
                    bounding_box=prev_bbox,
                    confidence=0.0
                )
                self.event_emitter.emit(event)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame and detect objects.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, detections_list)
            detections_list contains dicts with 'type', 'bbox', 'confidence', 'class_id', 'package_score'
        """
        self.frame_number += 1
        
        # Learn background if still learning
        self._learn_background(frame)
        
        # Get foreground mask after background is learned
        foreground_mask = None
        if self.background_learned:
            foreground_mask = self._get_foreground_mask(frame)
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        # Draw background learning status
        if not self.background_learned:
            h, w = frame.shape[:2]
            progress = (self.background_learning_frame_count / self.background_learning_frames) * 100
            status_text = f"Learning background... {progress:.0f}% ({self.background_learning_frame_count}/{self.background_learning_frames})"
            cv2.putText(annotated_frame, status_text,
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw package zone for debug visualization
        if self.debug_mode:
            h, w = frame.shape[:2]
            zone_y = int(h * (1.0 - self.package_zone_y_percent))
            cv2.line(annotated_frame, (0, zone_y), (w, zone_y), (0, 255, 255), 1)
            cv2.putText(annotated_frame, "Package Zone", (10, zone_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw foreground mask overlay in debug mode
            if foreground_mask is not None:
                # Create colored overlay for foreground mask
                overlay = annotated_frame.copy()
                overlay[foreground_mask > 0] = [0, 255, 255]  # Cyan for foreground
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
        
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
                    package_score = 0.0
                else:
                    # Check if it's a package with enhanced detection
                    is_pkg, package_score = self._is_package(
                        class_id, class_name, bbox_coords, frame, confidence
                    )
                    if is_pkg:
                        obj_type = 'package'
                        
                        # Disable foreground filtering for now - it's blocking too many packages
                        # Only filter if overlap is extremely low (< 1%) - almost never filter
                        # if self.background_learned and foreground_mask is not None:
                        #     overlap_ratio = self._check_foreground_overlap(bbox_coords, foreground_mask)
                        #     # Only skip if overlap is extremely low (< 1%) - very lenient
                        #     if overlap_ratio < 0.01:
                        #         # Package detection has almost no foreground overlap - likely background
                        #         continue
                        # Otherwise, accept all packages regardless of foreground overlap
                    else:
                        # Skip other objects that don't match package criteria
                        continue
                
                detections.append({
                    'type': obj_type,
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'confidence': confidence,
                    'package_score': package_score,
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
                if obj_type == 'package':
                    score_to_show = package_score if package_score > 0 else confidence
                    label = f"{obj_type} {score_to_show:.2f}"
                else:
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
        
        # Update confirmed packages (temporal filtering) - pass frame for zone checking
        self._update_confirmed_packages(current_detections, frame)
        
        # Detect enter/leave events (only for confirmed packages)
        self._detect_enter_leave(current_detections)
        
        # Draw confirmed packages with special visualization
        for obj_id, confirmed_pkg in self.confirmed_packages.items():
            if obj_id in current_detections:
                x1, y1, x2, y2 = confirmed_pkg['bbox']
                # Draw thicker border for confirmed packages
                cv2.rectangle(annotated_frame,
                             (int(x1) - 2, int(y1) - 2),
                             (int(x2) + 2, int(y2) + 2),
                             (0, 255, 255), 3)
                # Add confirmation indicator
                cv2.putText(annotated_frame, "CONFIRMED",
                          (int(x1), int(y2) + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Update previous detections with full data structure
        self.previous_detections = {}
        for obj_id, det in current_detections.items():
            self.previous_detections[obj_id] = {
                'type': det['type'],
                'bbox': det['bbox'],
                'center': det.get('center', self._calculate_bbox_center(det['bbox'])),
                'timestamp': det.get('timestamp', datetime.now())
            }
        
        return annotated_frame, detections

