"""
Event system for detection enter/leave notifications.
"""
from enum import Enum
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


class EventType(Enum):
    """Types of detection events."""
    PERSON_ENTER = "person_enter"
    PERSON_LEAVE = "person_leave"
    PACKAGE_ENTER = "package_enter"
    PACKAGE_LEAVE = "package_leave"


@dataclass
class DetectionEvent:
    """Represents a detection event."""
    event_type: EventType
    object_type: str
    timestamp: datetime
    bounding_box: tuple  # (x1, y1, x2, y2)
    confidence: float
    
    def __str__(self):
        return f"{self.event_type.value} | {self.object_type} | {self.timestamp.strftime('%H:%M:%S')} | conf: {self.confidence:.2f}"


class DetectionEventEmitter:
    """Manages event callbacks and emits detection events."""
    
    def __init__(self):
        self._callbacks: Dict[EventType, List[Callable[[DetectionEvent], None]]] = {
            event_type: [] for event_type in EventType
        }
    
    def register_callback(self, event_type: EventType, callback: Callable[[DetectionEvent], None]):
        """Register a callback function for a specific event type.
        
        Args:
            event_type: The type of event to listen for
            callback: Function that takes a DetectionEvent and returns None
        """
        if callback not in self._callbacks[event_type]:
            self._callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: EventType, callback: Callable[[DetectionEvent], None]):
        """Unregister a callback function.
        
        Args:
            event_type: The type of event
            callback: The callback function to remove
        """
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
    
    def emit(self, event: DetectionEvent):
        """Emit an event to all registered callbacks.
        
        Args:
            event: The DetectionEvent to emit
        """
        for callback in self._callbacks[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in callback for {event.event_type.value}: {e}")
    
    def register_all_events_callback(self, callback: Callable[[DetectionEvent], None]):
        """Register a callback for all event types.
        
        Args:
            callback: Function that takes a DetectionEvent and returns None
        """
        for event_type in EventType:
            self.register_callback(event_type, callback)

