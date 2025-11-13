"""YOLO model loader"""

import os
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False

# Global YOLO model instance
_yolo_model = None
_yolo_model_path = None


def get_yolo_model_path(project_dir: str) -> str:
    """Get the path to the YOLO model file"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_dir = sys._MEIPASS
        model_path = os.path.join(base_dir, "yolov8n.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(project_dir, "yolov8n.pt")
    else:
        # Running as script
        model_path = os.path.join(project_dir, "yolov8n.pt")
    return model_path


def get_yolo_model(project_dir: str = None, device: str = None) -> object:
    """Get or initialize the YOLO model"""
    global _yolo_model, _yolo_model_path
    
    if not YOLO_AVAILABLE:
        return None
    
    # If project_dir not provided, try to detect it
    if project_dir is None:
        if getattr(sys, 'frozen', False):
            project_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to project root
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(project_dir)))
    
    model_path = get_yolo_model_path(project_dir)
    
    # Return cached model if path hasn't changed
    if _yolo_model is not None and _yolo_model_path == model_path:
        return _yolo_model
    
    # Load device if not provided
    if device is None:
        from .device_detection import get_device
        device = get_device()
    
    if os.path.exists(model_path):
        try:
            _yolo_model = YOLO(model_path)
            _yolo_model_path = model_path
            try:
                _yolo_model.to(device)
                print(f"YOLO model loaded and using device: {device}")
            except Exception:
                print(f"YOLO model loaded (device auto-detected)")
            return _yolo_model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return None
    
    return None

