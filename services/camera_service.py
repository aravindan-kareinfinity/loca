"""Camera configuration service"""

import os
import json
import uuid
from typing import List, Optional, Dict

# Handle both relative and absolute imports
try:
    from ..models.camera import Camera
except ImportError:
    from models.camera import Camera


class CameraService:
    """Service for managing camera configurations"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Ensure config file exists with default structure"""
        if not os.path.exists(self.config_path):
            default_config = {
                "name": "People Flow",
                "port": 8000,
                "debug": True,
                "cameras": [],
                "showwebcam": False
            }
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=2)
            except OSError:
                pass
    
    def read_config(self) -> dict:
        """Read configuration from file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "cameras" not in data or not isinstance(data["cameras"], list):
                    data["cameras"] = []
                if "showwebcam" not in data:
                    data["showwebcam"] = False
                return data
        except (OSError, json.JSONDecodeError):
            return {"name": "People Flow", "port": 8000, "debug": True, "cameras": [], "showwebcam": False}
    
    def write_config(self, data: dict) -> None:
        """Write configuration to file"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
    
    def get_all_cameras(self) -> List[dict]:
        """Get all cameras from config"""
        cfg = self.read_config()
        return cfg.get("cameras", [])
    
    def get_camera_by_id(self, camera_id: str) -> Optional[dict]:
        """Get a camera by ID"""
        cameras = self.get_all_cameras()
        for cam in cameras:
            if cam.get("idCode") == camera_id:
                return cam
        return None
    
    def add_camera(self, camera_data: dict) -> Camera:
        """Add a new camera"""
        generated_id = f"CAM-{uuid.uuid4().hex[:12].upper()}"
        id_code = str(camera_data.get("idCode", "")).strip() or generated_id
        
        cam = Camera(
            idCode=id_code,
            name=str(camera_data.get("name", "")).strip(),
            location=str(camera_data.get("location", "")).strip(),
            description=(camera_data.get("description") or None),
            ipOrHost=str(camera_data.get("ipOrHost", "")).strip(),
            port=(int(camera_data["port"]) if str(camera_data.get("port", "")).strip().isdigit() else None),
            protocol=str(camera_data.get("protocol", "rtsp")).lower(),
            streamPath=(str(camera_data.get("streamPath", "")).strip() or None),
            username=(str(camera_data.get("username")) if camera_data.get("username") else None),
            password=(str(camera_data.get("password")) if camera_data.get("password") else None),
            macAddress=(str(camera_data.get("macAddress")) if camera_data.get("macAddress") else None),
            mode=str(camera_data.get("mode", "Recognize")),
            active=bool(camera_data.get("active", True)),
            lineEnabled=bool(camera_data.get("lineEnabled", False)),
            lineX1=(int(camera_data["lineX1"]) if str(camera_data.get("lineX1", "")).strip().isdigit() else None),
            lineY1=(int(camera_data["lineY1"]) if str(camera_data.get("lineY1", "")).strip().isdigit() else None),
            lineX2=(int(camera_data["lineX2"]) if str(camera_data.get("lineX2", "")).strip().isdigit() else None),
            lineY2=(int(camera_data["lineY2"]) if str(camera_data.get("lineY2", "")).strip().isdigit() else None),
            lineDirection=str(camera_data.get("lineDirection", "AtoBIsIn")),
            lineCountMode=str(camera_data.get("lineCountMode", "BOTH")).upper()
        )
        
        if not cam.idCode:
            raise ValueError("idCode required")
        if not cam.name:
            raise ValueError("name required")
        
        cfg = self.read_config()
        cams = cfg.get("cameras", [])
        
        if any(c.get("idCode") == cam.idCode for c in cams):
            raise ValueError("idCode already exists")
        
        cams.append(cam.to_dict())
        cfg["cameras"] = cams
        self.write_config(cfg)
        
        return cam
    
    def update_camera(self, camera_id: str, update_data: dict) -> Camera:
        """Update an existing camera"""
        cfg = self.read_config()
        cams = cfg.get("cameras", [])
        
        for idx, existing in enumerate(cams):
            if existing.get("idCode") == camera_id:
                updated = existing.copy()
                fields = [
                    "name", "location", "description", "ipOrHost", "port",
                    "protocol", "streamPath", "username", "password", "macAddress",
                    "mode", "active", "lineEnabled", "lineX1", "lineY1", "lineX2", "lineY2", 
                    "lineDirection", "lineCountMode"
                ]
                
                for f in fields:
                    if f in update_data and update_data[f] is not None:
                        updated[f] = update_data[f]
                
                cam = Camera(
                    idCode=updated.get("idCode", camera_id),
                    name=updated.get("name", existing.get("name", "")),
                    location=updated.get("location", existing.get("location", "")),
                    description=updated.get("description"),
                    ipOrHost=updated.get("ipOrHost", ""),
                    port=int(updated["port"]) if str(updated.get("port", "")).isdigit() else None,
                    protocol=str(updated.get("protocol", "rtsp")).lower(),
                    streamPath=updated.get("streamPath"),
                    username=updated.get("username"),
                    password=updated.get("password"),
                    macAddress=updated.get("macAddress"),
                    mode=updated.get("mode", "Recognize"),
                    active=bool(updated.get("active", True)),
                    lineEnabled=bool(updated.get("lineEnabled", False)),
                    lineX1=int(updated["lineX1"]) if str(updated.get("lineX1", "")).isdigit() else None,
                    lineY1=int(updated["lineY1"]) if str(updated.get("lineY1", "")).isdigit() else None,
                    lineX2=int(updated["lineX2"]) if str(updated.get("lineX2", "")).isdigit() else None,
                    lineY2=int(updated["lineY2"]) if str(updated.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(updated.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(updated.get("lineCountMode", "BOTH")).upper()
                )
                
                updated = cam.to_dict()
                cams[idx] = updated
                cfg["cameras"] = cams
                self.write_config(cfg)
                
                return cam
        
        raise ValueError("Camera not found")
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera"""
        cfg = self.read_config()
        cams = cfg.get("cameras", [])
        new_cams = [c for c in cams if c.get("idCode") != camera_id]
        
        if len(new_cams) == len(cams):
            return False
        
        cfg["cameras"] = new_cams
        self.write_config(cfg)
        return True
    
    def dict_to_camera(self, cam_dict: dict) -> Camera:
        """Convert dictionary to Camera object"""
        from urllib.parse import unquote
        
        return Camera(
            idCode=cam_dict.get("idCode", ""),
            name=cam_dict.get("name", ""),
            location=cam_dict.get("location", ""),
            description=cam_dict.get("description"),
            ipOrHost=cam_dict.get("ipOrHost", ""),
            port=int(cam_dict["port"]) if str(cam_dict.get("port", "")).isdigit() else None,
            protocol=str(cam_dict.get("protocol", "rtsp")).lower(),
            streamPath=cam_dict.get("streamPath"),
            username=cam_dict.get("username"),
            password=cam_dict.get("password") or (unquote(cam_dict.get("passwordEnc", "")) if cam_dict.get("passwordEnc") else None),
            macAddress=cam_dict.get("macAddress"),
            mode=cam_dict.get("mode", "Recognize"),
            active=True,
            lineEnabled=bool(cam_dict.get("lineEnabled", False)),
            lineX1=int(cam_dict["lineX1"]) if str(cam_dict.get("lineX1", "")).isdigit() else None,
            lineY1=int(cam_dict["lineY1"]) if str(cam_dict.get("lineY1", "")).isdigit() else None,
            lineX2=int(cam_dict["lineX2"]) if str(cam_dict.get("lineX2", "")).isdigit() else None,
            lineY2=int(cam_dict["lineY2"]) if str(cam_dict.get("lineY2", "")).isdigit() else None,
            lineDirection=str(cam_dict.get("lineDirection", "AtoBIsIn")),
            lineCountMode=str(cam_dict.get("lineCountMode", "BOTH")).upper()
        )

