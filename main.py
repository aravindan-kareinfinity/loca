from flask import Flask, send_from_directory, request, jsonify
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime
import uuid
import threading
import time
from typing import Dict

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import pickle
except Exception:
    pickle = None
try:
    import face_recognition
except Exception:
    face_recognition = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    import requests
except Exception:
    requests = None
import re


ProtocolType = Literal["rtsp", "onvif", "http", "https", "rtmp"]
ModeType = Literal["Recognize", "Identify", "Recognize and Identify"]


# Known camera vendors to help classify devices discovered on the LAN
KNOWN_CAMERA_VENDORS = [
    'axis', 'hikvision', 'dahua', 'uniview', 'cp plus',
    'bosch', 'hanwha', 'sony', 'avtech', 'arecont', 'vivotek', 'mobotix',
]


def get_mac_vendor(mac_address):
    """Lookup vendor name for a MAC address using macvendors API.
    Returns 'Unknown' on error.
    """
    if requests is None:
        return "Unknown"
    try:
        url = f"https://api.macvendors.com/{mac_address}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and response.text:
            return response.text.strip()
    except Exception:
        pass
    return "Unknown"


def find_lan_devices():
    """Discover devices using ARP table. Returns list of (ip, mac, vendor)."""
    try:
        output = os.popen("arp -a").read()
    except Exception:
        output = ""
    devices = re.findall(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f\-:]{17})", output, re.I)
    results = []
    for ip_address, mac in devices:
        normalized_mac = mac.replace('-', ':').lower()
        vendor = get_mac_vendor(normalized_mac)
        results.append((ip_address, normalized_mac, vendor))
    return results


@dataclass
class Camera:
    idCode: str
    name: str
    location: str
    description: Optional[str] = None
    ipOrHost: str = ""
    port: Optional[int] = None
    protocol: ProtocolType = "rtsp"
    streamPath: Optional[str] = None
    username: Optional[str] = None
    passwordEnc: Optional[str] = None  # store encrypted/placeholder, not plain
    macAddress: Optional[str] = None
    mode: ModeType = "Recognize"
    active: bool = True
    createdAt: datetime = field(default_factory=datetime.utcnow)
    updatedAt: datetime = field(default_factory=datetime.utcnow)
    # Tripwire settings (screen coordinates in stream frame)
    lineEnabled: bool = False
    lineX1: Optional[int] = None
    lineY1: Optional[int] = None
    lineX2: Optional[int] = None
    lineY2: Optional[int] = None
    # Direction setting: 'AtoBIsIn' or 'BtoAIsIn' (A is (x1,y1), B is (x2,y2))
    lineDirection: str = "AtoBIsIn"
    # Count mode: 'BOTH' | 'IN' | 'OUT'
    lineCountMode: str = "BOTH"

    @property
    def streamingUrl(self) -> str:
        # Return full URL including password for the UI as requested
        return self.connection_url()

    def connection_url(self) -> str:
        """Builds the actual URL for connecting to the camera feed.
        This includes the plaintext credential (passwordEnc) if present.
        """
        host = (self.ipOrHost or "").strip()
        if not host:
            return ""
        proto = (self.protocol or "rtsp").lower().rstrip(":")
        auth = ""
        if self.username:
            auth = self.username
            if self.passwordEnc:
                auth = f"{auth}:{self.passwordEnc}"
            auth += "@"
        port = f":{self.port}" if self.port else ""
        path = f"/{(self.streamPath or '').lstrip('/')}" if self.streamPath else ""
        return f"{proto}://{auth}{host}{port}{path}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["streamingUrl"] = self.streamingUrl
        # Do not leak passwordEnc if not desired; it's kept as is here
        # Convert datetimes to isoformat
        data["createdAt"] = self.createdAt.isoformat()
        data["updatedAt"] = self.updatedAt.isoformat()
        return data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# main.py is already inside the @project directory, so serve from BASE_DIR directly
PROJECT_DIR = BASE_DIR
FACE_DB_PATH = os.path.join(PROJECT_DIR, "face_database.pkl")
PRESENCE_LOG_PATH = os.path.join(PROJECT_DIR, "presence_log.json")
PERSON_IMAGES_DIR = os.path.join(PROJECT_DIR, "person_images")
YOLO_WEIGHTS = os.path.join(PROJECT_DIR, "yolov8n.pt")

app = Flask(__name__, static_folder=PROJECT_DIR, static_url_path="/")


# In-memory camera workers registry
class CameraWorker:
    def __init__(self, cam: 'Camera'):
        self.cam = cam
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.mode = cam.mode
        self.recognizer = None
        self.frame_counter = 0  # Track frame number for skipping
        self.line_config = {
            "enabled": bool(getattr(cam, 'lineEnabled', False)),
            "x1": getattr(cam, 'lineX1', None),
            "y1": getattr(cam, 'lineY1', None),
            "x2": getattr(cam, 'lineX2', None),
            "y2": getattr(cam, 'lineY2', None),
            "direction": getattr(cam, 'lineDirection', 'AtoBIsIn'),
            "countMode": getattr(cam, 'lineCountMode', 'BOTH'),
            "cameraId": cam.idCode
        }

    def start(self, force_restart=False):
        # If thread is alive and not forcing restart, keep existing thread
        if self.thread and self.thread.is_alive() and not force_restart:
            return
        # Stop existing thread if alive (for restart scenario)
        if force_restart and self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _encode_jpeg(self, frame):
        if cv2 is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buf.tobytes()

    def _process_recognize(self, frame):
        # Placeholder for recognition-only rendering
        if cv2 is None:
            return frame
        # Run lightweight full pipeline if available to produce overlays and logs
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Recognize mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return frame

    def _process_detect_train(self, frame):
        if cv2 is None:
            return frame
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Detect/Train mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        return frame

    def _process_full(self, frame):
        if cv2 is None:
            return frame
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Full mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        return frame

    def _ensure_recognizer(self):
        if self.recognizer is not None:
            return
        if cv2 is None or face_recognition is None or np is None:
            return
        try:
            self.recognizer = FaceRecognitionSystem()
        except Exception:
            self.recognizer = None

    def _run(self):
        if cv2 is None:
            # Without OpenCV we cannot stream frames; loop to keep thread alive but idle
            while not self.stop_event.is_set():
                time.sleep(0.5)
            return
        url = self.cam.connection_url()
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame_counter = 0  # Reset counter on start
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue
            
            # Process every 2nd frame (frames 0, 2, 4, ...)
            # This reduces processing by 50% while maintaining smooth streaming
            should_process = (self.frame_counter % 2 == 0)
            
            if should_process:
                # Full processing: detection, tracking, recognition
                mode = (self.mode or 'Recognize')
                if mode == 'Recognize':
                    frame = self._process_recognize(frame)
                elif mode == 'Identify':
                    frame = self._process_detect_train(frame)
                else:
                    frame = self._process_full(frame)

                jpeg = self._encode_jpeg(frame)
                if jpeg is not None:
                    with self.lock:
                        self.latest_jpeg = jpeg
            # For skipped frames, we still read from camera to stay current,
            # but reuse the last processed frame (latest_jpeg already contains it)
            # This maintains smooth video streaming while reducing processing load
            
            self.frame_counter += 1
            # Throttle
            time.sleep(0.03)
        cap.release()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg


workers: Dict[str, CameraWorker] = {}

def start_worker_for_camera(cam: 'Camera'):
    # Check if worker exists - if it does, we should stop it first when updating
    # For new cameras, create new worker; for updates, create fresh worker
    worker = workers.get(cam.idCode)
    if worker is None:
        # Create new worker for new camera
        worker = CameraWorker(cam)
        workers[cam.idCode] = worker
    else:
        # Worker exists - update settings but will restart if thread is not alive
        worker.cam = cam  # Update camera object with new settings
    
    # Update all worker settings
    worker.mode = cam.mode
    worker.line_config = {
        "enabled": bool(getattr(cam, 'lineEnabled', False)),
        "x1": getattr(cam, 'lineX1', None),
        "y1": getattr(cam, 'lineY1', None),
        "x2": getattr(cam, 'lineX2', None),
        "y2": getattr(cam, 'lineY2', None),
        "direction": getattr(cam, 'lineDirection', 'AtoBIsIn'),
        "countMode": getattr(cam, 'lineCountMode', 'BOTH'),
        "cameraId": cam.idCode
    }
    worker.start()

def stop_worker(id_code: str):
    worker = workers.get(id_code)
    if worker:
        worker.stop()
        workers.pop(id_code, None)


def start_webcam_worker():
    # Create a pseudo camera for webcam testing
    cam = Camera(
        idCode="WEBCAM",
        name="Webcam",
        location="Local",
        ipOrHost="",
        protocol="rtsp",  # not used for webcam
        streamPath=None,
        active=True,
        mode="Recognize"
    )
    worker = workers.get("WEBCAM")
    if worker is None:
        worker = CameraWorker(cam)
        workers["WEBCAM"] = worker
    worker.mode = cam.mode
    # Override run to open device 0
    def run_webcam():
        if cv2 is None:
            while not worker.stop_event.is_set():
                time.sleep(0.5)
            return
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Ensure recognizer exists for webcam worker
        worker._ensure_recognizer()
        worker.frame_counter = 0  # Reset counter on start
        while not worker.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            
            # Process every 2nd frame (frames 0, 2, 4, ...)
            # This reduces processing by 50% while maintaining smooth streaming
            should_process = (worker.frame_counter % 3 == 0)
            
            if should_process:
                # Full processing: detection, tracking, recognition
                mode = (worker.mode or 'Recognize')
                if mode == 'Recognize':
                    frame = worker._process_recognize(frame)
                elif mode == 'Identify':
                    frame = worker._process_detect_train(frame)
                else:
                    frame = worker._process_full(frame)
                jpeg = worker._encode_jpeg(frame)
                if jpeg is not None:
                    with worker.lock:
                        worker.latest_jpeg = jpeg
            # For skipped frames, we still read from camera to stay current,
            # but reuse the last processed frame (latest_jpeg already contains it)
            # This maintains smooth video streaming while reducing processing load
            
            worker.frame_counter += 1
            time.sleep(0.03)
        cap.release()
    worker.stop_event.clear()
    worker.thread = threading.Thread(target=run_webcam, daemon=True)
    worker.thread.start()


@app.get("/")
def serve_root():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.get("/<path:path>")
def serve_assets(path: str):
    # Serves any additional assets referenced by index.html, if present
    return send_from_directory(PROJECT_DIR, path)


@app.get('/api/stream/<id_code>')
def stream_camera(id_code: str):
    from flask import Response
    worker = workers.get(id_code)
    if worker is None:
        return jsonify({"error": "stream not available"}), 404

    def gen():
        boundary = b'--frame\r\n'
        while True:
            jpeg = worker.get_jpeg()
            if jpeg is not None:
                yield boundary
                yield b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- Face Recognition System (ported, PROJECT_DIR storage) ----------------
class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.unknown_faces_buffer = {}
        self.person_id_to_name = {}
        self.face_encoding_cache = {}
        self.creation_cooldown = {}
        self.next_track_id = 0
        self.tracks = {}
        self.max_age = 2
        self.next_person_id = 1
        self.active_person_counts = {}
        # Tripwire state per track id
        self.track_side = {}
        self.track_counted_in = {}
        # Face tracking cache: store encodings per track to avoid re-identification
        self.track_encoding_cache = {}  # {track_id: encoding}
        self.track_verification_counter = {}  # {track_id: frame_count} for periodic re-verification
        self.track_identification_confidence = {}  # {track_id: distance} - stores last identification distance
        # Bounding box movement tracking: skip face detection if bbox hasn't moved much
        self.track_last_bbox_center = {}  # {track_id: (cx, cy)} - stores last bbox center
        self.bbox_movement_threshold = 0.10  # 10% of bbox size - minimum movement to trigger face detection
        # Current camera_id for presence events
        self.current_camera_id = None
        # DeepSORT-style appearance features: extract once per track when person identified
        self.track_appearance_features = {}  # {track_id: feature_vector} - appearance features for matching
        self.track_feature_extracted = {}  # {track_id: bool} - flag to track if features extracted once

        os.makedirs(PERSON_IMAGES_DIR, exist_ok=True)
        self._load_face_database()
        self.presence_events = self._load_presence_log()

        # YOLO person detector
        self.yolo_model = None
        if YOLO is not None and os.path.exists(YOLO_WEIGHTS):
            try:
                self.yolo_model = YOLO(YOLO_WEIGHTS)
            except Exception:
                self.yolo_model = None

        # Initialize mapping for known faces
        for i, name in enumerate(self.known_face_names):
            person_id = i + 1
            self.person_id_to_name[person_id] = name
            if i < len(self.known_face_encodings):
                self.face_encoding_cache[person_id] = self.known_face_encodings[i]
        self.next_person_id = len(self.known_face_names) + 1

    # ----- Persistence -----
    def _load_face_database(self):
        if pickle is None:
            return
        if os.path.exists(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
            except Exception:
                self.known_face_encodings = []
                self.known_face_names = []

    def _save_face_database(self):
        if pickle is None:
            return
        try:
            with open(FACE_DB_PATH, 'wb') as f:
                pickle.dump({'encodings': self.known_face_encodings, 'names': self.known_face_names}, f)
        except Exception:
            pass

    def reload_face_database_names(self):
        # Reload only names from disk; keep encodings as is if lengths match
        if pickle is None:
            return
        try:
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    names = data.get('names', [])
                    if isinstance(names, list) and len(names) == len(self.known_face_encodings):
                        self.known_face_names = names
                        # Rebuild mapping
                        self.person_id_to_name = {i + 1: nm for i, nm in enumerate(self.known_face_names)}
                        # Update existing tracks with new names so bounding boxes show updated names immediately
                        for tid, track_info in self.tracks.items():
                            pid = track_info.get('person_id')
                            if pid is not None and pid in self.person_id_to_name:
                                new_name = self.person_id_to_name[pid]
                                track_info['name'] = new_name
        except Exception:
            pass

    def _load_presence_log(self):
        try:
            if os.path.exists(PRESENCE_LOG_PATH):
                with open(PRESENCE_LOG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception:
            return []
        return []

    def _save_presence_log(self):
        try:
            with open(PRESENCE_LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.presence_events, f, indent=2)
        except Exception:
            pass

    def _append_presence_event(self, person_id, name, event, image_path=None, camera_id=None):
        record = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "person_id": person_id,
            "name": name,
            "event": event
        }
        if image_path:
            record["image_path"] = os.path.relpath(image_path, PROJECT_DIR)
        if camera_id:
            record["camera_id"] = camera_id
        self.presence_events.append(record)
        self._save_presence_log()

    # ----- Helpers -----
    def _ensure_person_folder(self, person_id):
        folder = os.path.join(PERSON_IMAGES_DIR, f"person_{person_id}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _save_person_image(self, person_id, face_image):
        try:
            folder = self._ensure_person_folder(person_id)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(folder, f"{ts}.jpg")
            cv2.imwrite(path, face_image)
            return path
        except Exception:
            return None

    # ----- Detection & Tracking -----
    def _detect_people(self, frame):
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, classes=[0], verbose=False, conf=0.5)
                detections = []
                for result in results:
                    if getattr(result, 'boxes', None) is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0].cpu().numpy())
                            if cls == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                detections.append([x1, y1, x2, y2, conf])
                return detections
            except Exception:
                pass
        # Fallback via face detection approximating full body
        if face_recognition is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        detections = []
        for top, right, bottom, left in face_locations:
            h = bottom - top
            w = right - left
            x1 = max(0, left - w//2)
            y1 = max(0, top - h)
            x2 = min(frame.shape[1], right + w//2)
            y2 = min(frame.shape[0], bottom + h//2)
            detections.append([x1, y1, x2, y2, 0.8])
        return detections

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _track(self, detections, frame=None):
        """Enhanced tracking with DeepSORT-style appearance features.
        Once a person is identified, their appearance features are extracted once
        and used for matching in subsequent frames.
        """
        prev = self.tracks
        current = {}
        for tid in prev:
            prev[tid]['matched'] = False
        
        # Extract appearance features for new detections if frame provided
        detection_features = []
        if frame is not None and cv2 is not None:
            for x1, y1, x2, y2, conf in detections:
                feat = self._extract_appearance_features(frame, (x1, y1, x2, y2))
                detection_features.append(feat)
        else:
            detection_features = [None] * len(detections)
        
        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            bbox = (x1, y1, x2, y2)
            best_id = None
            best_score = -1.0
            det_feat = detection_features[idx] if idx < len(detection_features) else None
            
            # First pass: IoU matching
            for tid, tr in prev.items():
                iou = self._iou(bbox, tr['bbox'])
                
                # Combined score: IoU + appearance similarity (for identified persons)
                score = iou
                
                # If track has appearance features and person is identified, use feature matching
                if tid in self.track_appearance_features and det_feat is not None:
                    stored_feat = self.track_appearance_features[tid]
                    if stored_feat is not None:
                        feat_sim = self._cosine_similarity(det_feat, stored_feat)
                        # Weighted combination: IoU (60%) + Feature similarity (40%)
                        score = 0.6 * iou + 0.4 * feat_sim
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_id = tid
            
            if best_id is not None:
                # Match found - transfer identity and features
                prev_track = prev[best_id]
                current[best_id] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'person_id': prev_track.get('person_id'),  # Bind identity permanently
                    'name': prev_track.get('name', 'Unknown'),  # Bind name permanently
                    'matched': True,
                    'counted': prev_track.get('counted', False),
                    'image_saved': prev_track.get('image_saved', False)
                }
                # Preserve appearance features if already extracted
                if best_id in self.track_appearance_features:
                    # Optionally update features periodically, but keep original if person identified
                    if best_id not in self.track_feature_extracted or not self.track_feature_extracted.get(best_id, False):
                        if det_feat is not None:
                            self.track_appearance_features[best_id] = det_feat
                prev[best_id]['matched'] = True
            else:
                # New track
                tid = self.next_track_id
                self.next_track_id += 1
                current[tid] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'person_id': None,
                    'name': 'Unknown',
                    'matched': True,
                    'counted': False,
                    'image_saved': False
                }
                # Store appearance features for new track
                if det_feat is not None:
                    self.track_appearance_features[tid] = det_feat
                    self.track_feature_extracted[tid] = False  # Will be marked True when identified
        
        for tid, tr in prev.items():
            if not tr.get('matched', False):
                tr['last_seen'] += 1
                if tr['last_seen'] < self.max_age:
                    current[tid] = tr
        # exits
        removed = set(prev.keys()) - set(current.keys())
        for rid in removed:
            tr = prev.get(rid)
            if tr and tr.get('person_id') is not None and tr.get('counted', False):
                self._on_exit(tr)
            # Clean up face tracking cache and appearance features for removed tracks
            self.track_encoding_cache.pop(rid, None)
            self.track_verification_counter.pop(rid, None)
            self.track_identification_confidence.pop(rid, None)
            self.track_last_bbox_center.pop(rid, None)
            self.track_appearance_features.pop(rid, None)
            self.track_feature_extracted.pop(rid, None)
        self.tracks = current
        return current

    def _detect_faces_in_bbox(self, frame, bbox):
        if face_recognition is None:
            return []
        x1, y1, x2, y2 = map(int, bbox)
        m = 20
        x1 = max(0, x1 - m)
        y1 = max(0, y1 - m)
        x2 = min(frame.shape[1], x2 + m)
        y2 = min(frame.shape[0], y2 + m)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)
        faces = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            faces.append({
                'face_location': (top + y1, right + x1, bottom + y1, left + x1),
                'encoding': enc
            })
        return faces

    @staticmethod
    def _face_quality_ok(face_image):
        try:
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return False
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = float(np.mean(gray))
            return lap > 50 and 30 < bright < 220
        except Exception:
            return False

    def _identify(self, encoding):
        if not self.known_face_encodings or face_recognition is None:
            return None, float('inf')
        dists = face_recognition.face_distance(self.known_face_encodings, encoding)
        idx = int(np.argmin(dists))
        if dists[idx] < 0.6:
            return self.known_face_names[idx], float(dists[idx])
        return None, float(dists[idx])

    def _on_identity(self, track_id, person_id, name, face_image=None):
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        if not tr.get('counted', False):
            count = self.active_person_counts.get(person_id, 0)
            image_path = None
            if count == 0 and face_image is not None and face_image.size != 0:
                image_path = self._save_person_image(person_id, face_image)
            self._append_presence_event(person_id, name, 'enter', image_path, camera_id=self.current_camera_id)
            self.active_person_counts[person_id] = count + 1
            tr['counted'] = True
        if face_image is not None and not tr.get('image_saved', False):
            p = self._save_person_image(person_id, face_image)
            if p:
                tr['image_saved'] = True

    def _on_exit(self, track):
        pid = track.get('person_id')
        name = track.get('name', f'Person_{pid}') if pid is not None else 'Unknown'
        if pid is None:
            return
        cnt = self.active_person_counts.get(pid, 0)
        if cnt > 0:
            cnt -= 1
            self.active_person_counts[pid] = cnt
            if cnt == 0:
                self._append_presence_event(pid, name, 'exit', camera_id=self.current_camera_id)

    def _crop_face(self, frame, loc):
        top, right, bottom, left = loc
        e = 10
        top = max(0, top - e)
        left = max(0, left - e)
        bottom = min(frame.shape[0], bottom + e)
        right = min(frame.shape[1], right + e)
        return frame[top:bottom, left:right]

    def _extract_appearance_features(self, frame, bbox):
        """Extract appearance features from bbox region for DeepSORT-style matching.
        Uses color histogram and simple texture features as lightweight alternative.
        Returns normalized feature vector.
        """
        if cv2 is None or np is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            
            # Resize to fixed size for consistent feature extraction
            roi_resized = cv2.resize(roi, (64, 128)) if roi.shape[0] > 0 and roi.shape[1] > 0 else roi
            
            # Extract color histogram features (HSV)
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # Normalize histograms
            hist_h = hist_h / (np.sum(hist_h) + 1e-8)
            hist_s = hist_s / (np.sum(hist_s) + 1e-8)
            hist_v = hist_v / (np.sum(hist_v) + 1e-8)
            
            # Combine features
            features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            return features
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a, b):
        """Calculate cosine similarity between two feature vectors."""
        if a is None or b is None:
            return 0.0
        if np is None:
            return 0.0
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        except Exception:
            return 0.0

    def process_frame(self, frame, line_config=None):
        # Store current camera_id for use in presence events
        self.current_camera_id = line_config.get('cameraId') if line_config else None
        dets = self._detect_people(frame)
        tracks = self._track(dets, frame)  # Pass frame for appearance feature extraction
        # Draw tripwire line and side labels if enabled
        if line_config and line_config.get('enabled') and all(line_config.get(k) is not None for k in ('x1','y1','x2','y2')):
            x1, y1, x2, y2 = int(line_config['x1']), int(line_config['y1']), int(line_config['x2']), int(line_config['y2'])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            # Compute midpoint and perpendicular for label placement
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            vx, vy = (x2 - x1), (y2 - y1)
            # Perpendicular vector (left of A->B is (+nx, +ny))
            nx, ny = -vy, vx
            norm = (nx**2 + ny**2) ** 0.5 or 1.0
            nx, ny = nx / norm, ny / norm
            offset = 30  # pixels away from line
            # Determine which side is IN according to direction
            direction = line_config.get('direction', 'AtoBIsIn')
            count_mode = line_config.get('countMode', 'BOTH')
            in_left = (direction == 'AtoBIsIn')
            # Label positions
            in_pos = (int(mx + (nx if in_left else -nx) * offset), int(my + (ny if in_left else -ny) * offset))
            out_pos = (int(mx + (-nx if in_left else nx) * offset), int(my + (-ny if in_left else ny) * offset))
            def draw_label(img, pos, text, color_bg, color_fg=(255,255,255)):
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2
                (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
                x, y = pos
                x = max(0, min(img.shape[1] - tw - 16, x - tw // 2))
                y = max(th + 10, min(img.shape[0] - 6, y))
                cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 4), color_bg, -1)
                cv2.putText(img, text, (x, y - 4), font, scale, color_fg, thickness, cv2.LINE_AA)
            # Draw according to count mode
            if count_mode in ('BOTH', 'IN'):
                draw_label(frame, in_pos, 'IN', (32, 114, 187))
            if count_mode in ('BOTH', 'OUT'):
                draw_label(frame, out_pos, 'OUT', (227, 151, 99))
        for tid, info in tracks.items():
            bbox = info['bbox']
            person_name = info.get('name', 'Unknown')
            color = (0, 0, 255) if person_name == 'Unknown' else (0, 255, 0)
            
            # Optimization #2: Bounding Box Movement Tracking
            # Calculate current bbox center
            x1, y1, x2, y2 = map(float, bbox)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Check if bbox has moved significantly
            last_center = self.track_last_bbox_center.get(tid)
            bbox_moved_significantly = True
            if last_center is not None and bbox_width > 0 and bbox_height > 0:
                dx = abs(current_center[0] - last_center[0])
                dy = abs(current_center[1] - last_center[1])
                # Movement threshold: 10% of bbox size
                threshold_x = bbox_width * self.bbox_movement_threshold
                threshold_y = bbox_height * self.bbox_movement_threshold
                bbox_moved_significantly = (dx > threshold_x) or (dy > threshold_y)
            
            # Update last bbox center
            self.track_last_bbox_center[tid] = current_center
            
            # Optimization #3: Confidence-Based Adaptive Verification Interval
            # Get last identification confidence (distance)
            last_confidence = self.track_identification_confidence.get(tid, 1.0)
            
            # Determine verification interval based on confidence
            # Lower distance = higher confidence
            if last_confidence < 0.4:
                verification_interval = 10  # Very high confidence: verify every 10 frames
            elif last_confidence < 0.5:
                verification_interval = 5   # High confidence: verify every 5 frames (default)
            elif last_confidence < 0.6:
                verification_interval = 3   # Medium confidence: verify every 3 frames
            else:
                verification_interval = 2   # Low confidence: verify every 2 frames
            
            # Face tracking cache optimization: skip face detection for known tracks
            # unless periodic verification is due OR bbox moved significantly
            should_detect_faces = True
            has_person_id = (self.tracks[tid].get('person_id') is not None)
            
            if has_person_id:
                # Known track: check if verification is due
                verification_count = self.track_verification_counter.get(tid, 0)
                
                # Skip face detection if:
                # 1. Verification counter not reached (periodic check)
                # 2. AND bbox hasn't moved significantly (movement-based check)
                if verification_count < verification_interval and not bbox_moved_significantly:
                    # Skip face detection - reuse cached identity
                    should_detect_faces = False
                    self.track_verification_counter[tid] = verification_count + 1
                    # Use cached name and person_id from track info
                    person_name = self.tracks[tid].get('name', 'Unknown')
                    color = (0, 255, 0)  # Green for known person
                else:
                    # Periodic verification OR significant movement: reset counter and do full detection
                    self.track_verification_counter[tid] = 0
            
            if should_detect_faces:
                # Perform face detection and identification
                faces = self._detect_faces_in_bbox(frame, bbox)
                if faces:
                    for face in faces:
                        enc = face['encoding']
                        loc = face['face_location']
                        face_img = self._crop_face(frame, loc)
                        name, dist = self._identify(enc)
                        # Store identification confidence for adaptive verification interval
                        self.track_identification_confidence[tid] = dist
                        
                        if name:
                            if self.tracks[tid].get('person_id') is None:
                                # find id
                                pid = None
                                for pid_k, nm in self.person_id_to_name.items():
                                    if nm == name:
                                        pid = pid_k
                                        break
                                if pid is None:
                                    pid = self.next_person_id
                                    self.next_person_id += 1
                                    self.person_id_to_name[pid] = name
                                # ONE TIME: Bind person identity to track permanently
                                self.tracks[tid]['person_id'] = pid
                                self.tracks[tid]['name'] = name
                                self.face_encoding_cache[pid] = enc
                                # Cache encoding for this track
                                self.track_encoding_cache[tid] = enc
                                self.track_verification_counter[tid] = 0
                                # ONE TIME: Extract and bind appearance features for DeepSORT matching
                                bbox = self.tracks[tid].get('bbox')
                                if bbox and not self.track_feature_extracted.get(tid, False):
                                    feat = self._extract_appearance_features(frame, bbox)
                                    if feat is not None:
                                        self.track_appearance_features[tid] = feat
                                        self.track_feature_extracted[tid] = True  # Mark as extracted once
                                self._on_identity(tid, pid, name, face_img)
                            else:
                                # Known track: update cache and reset verification counter
                                # Identity already bound - just verify periodically
                                self.track_encoding_cache[tid] = enc
                                self.track_verification_counter[tid] = 0
                            person_name = name
                            color = (0, 255, 0)
                            t, r, b, l = loc
                            cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)
                        else:
                            # unknown buffer
                            buf = self.unknown_faces_buffer.setdefault(tid, [])
                            now = time.time()
                            prev_t = self.creation_cooldown.get(tid, 0)
                            if now - prev_t > 2.0 and self._face_quality_ok(face_img):
                                buf.append({'encoding': enc, 'image': face_img, 'ts': now})
                                if len(buf) >= 3:
                                    self._create_new_person_from_track(tid)
                                    self.creation_cooldown[tid] = now
                                    person_name = self.tracks[tid]['name']
            
            # Clean up cache for removed tracks (will be cleaned when track is removed)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {person_name}"
            if person_name == 'Unknown' and tid in self.unknown_faces_buffer:
                label += f" {len(self.unknown_faces_buffer[tid])}/3"
            if info.get('last_seen', 0) > 0:
                label += " [soon gone]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tripwire crossing detection (centroid vs directed line side change)
            if line_config and line_config.get('enabled') and all(line_config.get(k) is not None for k in ('x1','y1','x2','y2')):
                lx1, ly1, lx2, ly2 = int(line_config['x1']), int(line_config['y1']), int(line_config['x2']), int(line_config['y2'])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                side = self._point_side(cx, cy, lx1, ly1, lx2, ly2)
                prev_side = self.track_side.get(tid)
                self.track_side[tid] = side
                # Only count when identity known and we switched sides
                if prev_side is not None and prev_side != 0 and side != 0 and np.sign(prev_side) != np.sign(side):
                    # Determine direction: right->left (prev<0 to >0) or left->right (prev>0 to <0)
                    crossed_right_to_left = (prev_side < 0 and side > 0)
                    # Map to IN/OUT based on configured direction
                    direction = line_config.get('direction', 'AtoBIsIn')
                    event = 'enter' if (direction == 'AtoBIsIn' and crossed_right_to_left) or (direction == 'BtoAIsIn' and not crossed_right_to_left) else 'exit'
                    # Apply count mode filter
                    count_mode = line_config.get('countMode', 'BOTH')
                    if (count_mode == 'IN' and event != 'enter') or (count_mode == 'OUT' and event != 'exit'):
                        continue
                    pid = self.tracks[tid].get('person_id')
                    name = self.tracks[tid].get('name', f'Person_{pid}') if pid is not None else 'Unknown'
                    if pid is not None:
                        snapshot = frame[max(0, cy-60):min(frame.shape[0], cy+60), max(0, cx-60):min(frame.shape[1], cx+60)]
                        img_path = self._save_person_image(pid, snapshot) if snapshot.size != 0 else None
                        self._append_presence_event(pid, name, event, img_path, camera_id=line_config.get('cameraId'))

    @staticmethod
    def _point_side(px, py, x1, y1, x2, y2):
        # Returns signed area factor: >0 left of line A->B, <0 right, 0 on line
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    def _create_new_person_from_track(self, tid):
        items = self.unknown_faces_buffer.get(tid) or []
        if not items:
            return
        enc = items[0]['encoding']
        imgs = [i.get('image') for i in items if i.get('image') is not None]
        img = imgs[-1] if imgs else None
        pid = self.next_person_id
        name = f"Person_{pid}"
        self.tracks[tid]['person_id'] = pid
        self.tracks[tid]['name'] = name
        self.person_id_to_name[pid] = name
        self.face_encoding_cache[pid] = enc
        # Cache encoding for this track to enable face tracking cache optimization
        self.track_encoding_cache[tid] = enc
        self.track_verification_counter[tid] = 0
        self.known_face_encodings.append(enc)
        self.known_face_names.append(name)
        self.next_person_id += 1
        self._save_face_database()
        try:
            del self.unknown_faces_buffer[tid]
        except Exception:
            pass
        self._on_identity(tid, pid, name, img)


# ---------------- Presence API ----------------
@app.get('/api/presence')
def api_presence():
    try:
        if os.path.exists(PRESENCE_LOG_PATH):
            with open(PRESENCE_LOG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return jsonify(data)
    except Exception:
        pass
    return jsonify([])

@app.get('/api/people')
def api_people():
    # Return known people with ids, names, and image counts
    people = []
    try:
        encs, names = [], []
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, 'rb') as f:
                data = pickle.load(f)
                encs = data.get('encodings', [])
                names = data.get('names', [])
        for i, nm in enumerate(names):
            person_id = i + 1
            folder = os.path.join(PERSON_IMAGES_DIR, f'person_{person_id}')
            img_count = 0
            if os.path.isdir(folder):
                img_count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            people.append({'person_id': person_id, 'name': nm, 'images': img_count})
    except Exception:
        pass
    return jsonify(people)

@app.put('/api/people/<int:person_id>')
def api_rename_person(person_id: int):
    payload = request.get_json(force=True, silent=True) or {}
    new_name = str(payload.get('name', '')).strip()
    if not new_name:
        return jsonify({'error': 'name required'}), 400
    # Update name in face database
    try:
        if not os.path.exists(FACE_DB_PATH):
            return jsonify({'error': 'no face database'}), 400
        with open(FACE_DB_PATH, 'rb') as f:
            data = pickle.load(f)
        names = data.get('names', [])
        if person_id < 1 or person_id > len(names):
            return jsonify({'error': 'invalid person_id'}), 400
        names[person_id - 1] = new_name
        data['names'] = names
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        return jsonify({'error': 'failed to update'}), 500
    # Refresh all running recognizers to pick up new names
    try:
        for w in workers.values():
            if getattr(w, 'recognizer', None):
                w.recognizer.reload_face_database_names()
    except Exception:
        pass
    return jsonify({'ok': True, 'person_id': person_id, 'name': new_name})


@app.get('/api/discover/cameras')
def discover_only_cameras_api():
    """API endpoint to discover and return only likely camera devices."""
    devices = find_lan_devices()
    cameras_only = []
    for ip_address, mac, vendor in devices:
        vendor_lower = vendor.lower()
        if any(v in vendor_lower for v in KNOWN_CAMERA_VENDORS):
            cameras_only.append({
                'ip_address': ip_address,
                'mac_address': mac,
                'vendor': vendor,
            })
    return jsonify(cameras_only)


@app.get('/api/discover')
def discover_cameras_api():
    """API endpoint to discover devices on the LAN and flag likely cameras.
    Also checks if cameras are already added to the system.
    Returns cameras with 'status' field: 'already_added' or 'can_add'.
    """
    devices = find_lan_devices()
    
    # Load existing cameras from config to check for duplicates
    existing_cameras = []
    try:
        config_path = os.path.join(PROJECT_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                existing_cameras = cfg.get("cameras", [])
    except Exception:
        pass
    
    discovered = []
    for ip_address, mac, vendor in devices:
        vendor_lower = vendor.lower()
        is_camera = any(v in vendor_lower for v in KNOWN_CAMERA_VENDORS)
        
        # Check if this device is already added (by IP or MAC)
        status = "can_add"
        existing_camera_id = None
        
        if is_camera:
            for existing in existing_cameras:
                existing_ip = (existing.get("ipOrHost") or "").strip().lower()
                existing_mac = (existing.get("macAddress") or "").strip().lower()
                
                if (existing_ip and existing_ip == ip_address.lower()) or \
                   (existing_mac and existing_mac and existing_mac == mac):
                    status = "already_added"
                    existing_camera_id = existing.get("idCode")
                    break
        
        discovered.append({
            'ip_address': ip_address,
            'mac_address': mac,
            'vendor': vendor,
            'is_camera': is_camera,
            'status': status,
            'existing_camera_id': existing_camera_id,
        })
    
    return jsonify(discovered)


if __name__ == "__main__":
    # Ensure a default config.json exists alongside index.html
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "name": "People Flow",
            "port": 8000,
            "debug": True,
            "cameras": [],
            "showwebcam": False
        }
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
        except OSError:
            pass

    # Helpers to read/write cameras
    def read_config() -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "cameras" not in data or not isinstance(data["cameras"], list):
                    data["cameras"] = []
                if "showwebcam" not in data:
                    data["showwebcam"] = False
                return data
        except (OSError, json.JSONDecodeError):
            return {"name": "People Flow", "port": 8000, "debug": True, "cameras": []}

    def write_config(data: dict) -> None:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    @app.get("/api/cameras")
    def api_get_cameras():
        cfg = read_config()
        return jsonify(cfg.get("cameras", []))

    @app.get("/api/config")
    def api_get_config():
        return jsonify(read_config())

    @app.put("/api/config")
    def api_put_config():
        payload = request.get_json(force=True, silent=True) or {}
        cfg = read_config()
        if "showwebcam" in payload:
            cfg["showwebcam"] = bool(payload["showwebcam"])
            write_config(cfg)
            # Start/stop webcam worker accordingly
            if cfg["showwebcam"]:
                start_webcam_worker()
            else:
                stop_worker("WEBCAM")
        return jsonify(cfg)

    @app.post("/api/cameras")
    def api_add_camera():
        payload = request.get_json(force=True, silent=True) or {}
        # Minimal validation and normalization
        generated_id = f"CAM-{uuid.uuid4().hex[:12].upper()}"
        id_code = str(payload.get("idCode", "")).strip() or generated_id
        cam = Camera(
            idCode=id_code,
            name=str(payload.get("name", "")).strip(),
            location=str(payload.get("location", "")).strip(),
            description=(payload.get("description") or None),
            ipOrHost=str(payload.get("ipOrHost", "")).strip(),
            port=(int(payload["port"]) if str(payload.get("port", "")).strip().isdigit() else None),
            protocol=str(payload.get("protocol", "rtsp")).lower(),
            streamPath=(str(payload.get("streamPath", "")).strip() or None),
            username=(str(payload.get("username")) if payload.get("username") else None),
            passwordEnc=(str(payload.get("password")) if payload.get("password") else None),
            macAddress=(str(payload.get("macAddress")) if payload.get("macAddress") else None),
            mode=str(payload.get("mode", "Recognize")),
            active=bool(payload.get("active", True)),
            lineEnabled=bool(payload.get("lineEnabled", False)),
            lineX1=(int(payload["lineX1"]) if str(payload.get("lineX1", "")).strip().isdigit() else None),
            lineY1=(int(payload["lineY1"]) if str(payload.get("lineY1", "")).strip().isdigit() else None),
            lineX2=(int(payload["lineX2"]) if str(payload.get("lineX2", "")).strip().isdigit() else None),
            lineY2=(int(payload["lineY2"]) if str(payload.get("lineY2", "")).strip().isdigit() else None),
            lineDirection=str(payload.get("lineDirection", "AtoBIsIn")),
            lineCountMode=str(payload.get("lineCountMode", "BOTH")).upper()
        )
        if not cam.idCode:
            return jsonify({"error": "idCode required"}), 400
        if not cam.name:
            return jsonify({"error": "name required"}), 400
        cfg = read_config()
        cams = cfg.get("cameras", [])
        # Enforce unique idCode
        if any(c.get("idCode") == cam.idCode for c in cams):
            return jsonify({"error": "idCode already exists"}), 400
        cams.append(cam.to_dict())
        cfg["cameras"] = cams
        write_config(cfg)
        # Start worker if active
        if cam.active:
            start_worker_for_camera(cam)
        return jsonify(cam.to_dict()), 201

    @app.put("/api/cameras/<id_code>")
    def api_update_camera(id_code: str):
        payload = request.get_json(force=True, silent=True) or {}
        cfg = read_config()
        cams = cfg.get("cameras", [])
        for idx, existing in enumerate(cams):
            if existing.get("idCode") == id_code:
                # Merge updates
                updated = existing.copy()
                fields = [
                    "name", "location", "description", "ipOrHost", "port",
                    "protocol", "streamPath", "username", "macAddress",
                    "mode", "active", "lineEnabled", "lineX1", "lineY1", "lineX2", "lineY2", "lineDirection", "lineCountMode"
                ]
                for f in fields:
                    if f in payload and payload[f] is not None:
                        updated[f] = payload[f]
                # Update password if provided
                if payload.get("password"):
                    updated["passwordEnc"] = str(payload.get("password"))
                # Recompute streamingUrl via model for consistency
                cam = Camera(
                    idCode=updated.get("idCode", id_code),
                    name=updated.get("name", existing.get("name", "")),
                    location=updated.get("location", existing.get("location", "")),
                    description=updated.get("description"),
                    ipOrHost=updated.get("ipOrHost", ""),
                    port=int(updated["port"]) if str(updated.get("port", "")).isdigit() else None,
                    protocol=str(updated.get("protocol", "rtsp")).lower(),
                    streamPath=updated.get("streamPath"),
                    username=updated.get("username"),
                    passwordEnc=updated.get("passwordEnc"),
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
                write_config(cfg)
                # Always refresh worker to ensure new settings take effect
                # Stop existing worker if it exists
                existing_worker = workers.get(id_code)
                if existing_worker:
                    existing_worker.stop()
                    # Don't remove from workers dict, we'll update it
                
                # Start/stop worker according to active flag
                if updated.get("active", True):
                    # Update camera object in existing worker or create new one
                    if existing_worker:
                        # Update existing worker with new camera settings
                        existing_worker.cam = cam
                        existing_worker.mode = cam.mode
                        existing_worker.line_config = {
                            "enabled": bool(cam.lineEnabled),
                            "x1": cam.lineX1,
                            "y1": cam.lineY1,
                            "x2": cam.lineX2,
                            "y2": cam.lineY2,
                            "direction": cam.lineDirection,
                            "countMode": cam.lineCountMode,
                            "cameraId": cam.idCode
                        }
                        # Force restart to refresh video connection with new settings
                        existing_worker.start(force_restart=True)
                    else:
                        # Create new worker
                        start_worker_for_camera(cam)
                else:
                    # Camera deactivated - remove worker
                    stop_worker(id_code)
                return jsonify(updated)
        return jsonify({"error": "Camera not found"}), 404

    @app.post("/api/cameras/<id_code>/preview/start")
    def api_start_preview(id_code: str):
        # Start a worker thread for this camera regardless of active flag
        cfg = read_config()
        cams = cfg.get("cameras", [])
        for c in cams:
            if c.get("idCode") == id_code:
                cam = Camera(
                    idCode=c.get("idCode"),
                    name=c.get("name", ""),
                    location=c.get("location", ""),
                    description=c.get("description"),
                    ipOrHost=c.get("ipOrHost", ""),
                    port=int(c["port"]) if str(c.get("port", "")).isdigit() else None,
                    protocol=str(c.get("protocol", "rtsp")).lower(),
                    streamPath=c.get("streamPath"),
                    username=c.get("username"),
                    passwordEnc=c.get("passwordEnc"),
                    macAddress=c.get("macAddress"),
                    mode=c.get("mode", "Recognize"),
                    active=True,
                    lineEnabled=bool(c.get("lineEnabled", False)),
                    lineX1=int(c["lineX1"]) if str(c.get("lineX1", "")).isdigit() else None,
                    lineY1=int(c["lineY1"]) if str(c.get("lineY1", "")).isdigit() else None,
                    lineX2=int(c["lineX2"]) if str(c.get("lineX2", "")).isdigit() else None,
                    lineY2=int(c["lineY2"]) if str(c.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(c.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(c.get("lineCountMode", "BOTH")).upper()
                )
                start_worker_for_camera(cam)
                return jsonify({"ok": True})
        return jsonify({"error": "Camera not found"}), 404

    @app.post("/api/cameras/<id_code>/preview/stop")
    def api_stop_preview(id_code: str):
        stop_worker(id_code)
        return jsonify({"ok": True})

    @app.delete("/api/cameras/<id_code>")
    def api_delete_camera(id_code: str):
        cfg = read_config()
        cams = cfg.get("cameras", [])
        new_cams = [c for c in cams if c.get("idCode") != id_code]
        if len(new_cams) == len(cams):
            return jsonify({"error": "Camera not found"}), 404
        cfg["cameras"] = new_cams
        write_config(cfg)
        # Stop worker if running
        stop_worker(id_code)
        return jsonify({"ok": True})

    # Start workers for active cameras on server start
    try:
        boot_cfg = read_config()
        for c in boot_cfg.get("cameras", []):
            if c.get("active"):
                cam = Camera(
                    idCode=c.get("idCode"),
                    name=c.get("name", ""),
                    location=c.get("location", ""),
                    description=c.get("description"),
                    ipOrHost=c.get("ipOrHost", ""),
                    port=int(c["port"]) if str(c.get("port", "")).isdigit() else None,
                    protocol=str(c.get("protocol", "rtsp")).lower(),
                    streamPath=c.get("streamPath"),
                    username=c.get("username"),
                    passwordEnc=c.get("passwordEnc"),
                    macAddress=c.get("macAddress"),
                    mode=c.get("mode", "Recognize"),
                    active=True,
                    lineEnabled=bool(c.get("lineEnabled", False)),
                    lineX1=int(c["lineX1"]) if str(c.get("lineX1", "")).isdigit() else None,
                    lineY1=int(c["lineY1"]) if str(c.get("lineY1", "")).isdigit() else None,
                    lineX2=int(c["lineX2"]) if str(c.get("lineX2", "")).isdigit() else None,
                    lineY2=int(c["lineY2"]) if str(c.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(c.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(c.get("lineCountMode", "BOTH")).upper()
                )
                start_worker_for_camera(cam)
        if boot_cfg.get("showwebcam"):
            start_webcam_worker()
    except Exception:
        pass

    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)


